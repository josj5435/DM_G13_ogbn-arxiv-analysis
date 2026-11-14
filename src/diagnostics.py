import argparse,gzip,csv
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT,read_json,save_json,save_table,sha,load_data,load_config,environment
from .evaluate import rank_metrics,rank_order
from .preprocess import eval_baskets

def rank_difference_cases(rows, n=10):
    """Diagnostic ranks only: equal scores share average ranks, including PageRank.

    Keep every case tied at the nth absolute gap; node IDs never choose membership.
    The stored ordinal ranks and the official NDCG tie convention remain untouched.
    """
    if n < 1: raise ValueError('n must be positive')
    result=rows.copy()
    result['indegree_average_rank']=result.in_degree.rank(method='average',ascending=False)
    result['pagerank_average_rank']=result.pagerank.rank(method='average',ascending=False)
    result['rank_difference_ordinal']=result.indegree_rank-result.pagerank_rank
    result['rank_difference']=result.indegree_average_rank-result.pagerank_average_rank
    result['rank_difference_abs']=result.rank_difference.abs()
    if result.empty: return result
    threshold=result.rank_difference_abs.nlargest(min(n,len(result))).iloc[-1]
    return result[result.rank_difference_abs>=threshold].sort_values(
        ['rank_difference_abs','rank_difference'],ascending=[False,False]).reset_index(drop=True)


def write_rank_difference_diagnostics(root):
    """Regenerate just the diagnostic cases from packaged scores, without raw data."""
    out=root/'diagnostics'
    joined=pd.read_parquet(out/'influence_cluster_join.parquet')
    cases=rank_difference_cases(joined)
    save_table(out/'rank_difference_cases.csv',cases)
    return cases

def map_titles(out):
    mapping=pd.read_csv(ROOT/'data'/'raw'/'arxiv'/'mapping'/'nodeidx2paperid.csv.gz')
    mapping.columns=['node_id','mag_id']; records=[]; duplicate_ids=0; seen=set()
    textpath=ROOT/'data'/'titleabs.tsv.gz'
    if textpath.exists():
        with gzip.open(textpath,'rt',encoding='utf-8') as f:
            for line in f:
                pieces=line.rstrip('\n').split('\t',2)
                if len(pieces)<2 or not pieces[0].isdigit(): continue
                mid=int(pieces[0])
                if mid in seen: duplicate_ids+=1; continue
                seen.add(mid); records.append(dict(mag_id=mid,title=pieces[1]))
        titles=mapping.merge(pd.DataFrame(records),on='mag_id',how='left',validate='one_to_one')
    else: titles=mapping.assign(title='')
    save_table(out/'titles.parquet',titles)
    save_json(out/'title_mapping.json',dict(n_nodes=len(mapping),matched=int(titles.title.notna().sum()),matched_fraction=float(titles.title.notna().mean()),duplicate_metadata_ids=duplicate_ids,join_key='MAG ID -> original node_id',metadata_license='Original titles/abstracts remain attributed to their source papers; no blanket relicensing assumed'))
    return titles

def diagnostics(root):
    data=load_data(); final=root/'final'; out=root/'diagnostics'; out.mkdir(exist_ok=True)
    frozen=read_json(root/'frozen.json'); sel=frozen['selection']['clusters']; tag=f'k{sel["k"]}_d{sel["d"]}_s0'
    search=pd.read_csv(root/'validation'/'clustering_metrics.csv')
    search['eligible']=(search.d==128)|(search.explained_variance>=frozen['config']['clusters']['variance_min'])
    search['selected']=(search.k==sel['k'])&(search.d==sel['d'])
    search['reason']=np.where(~search.eligible,'training cumulative explained variance < 0.90',np.where(search.selected,'within best S - 0.01; then smallest dimension, smallest k','eligible but not chosen by frozen silhouette/tie rule'))
    save_table(out/'validation_cluster_selection_audit.csv',search)
    assign=np.load(final/'models'/f'{tag}_assignments.npz')['assignment']; titles=map_titles(out)
    pair_rows=pd.read_parquet(final/'eval_pairs.parquet')
    pair_rows['cluster_a']=assign[pair_rows.node_a.to_numpy()]; pair_rows['cluster_b']=assign[pair_rows.node_b.to_numpy()]
    pair_rows['same_cluster']=pair_rows.cluster_a==pair_rows.cluster_b
    pair_rows['model_id']=tag
    save_table(out/'selected_model_pair_annotations.parquet',pair_rows)
    pr=pd.read_parquet(final/'pagerank_scores.parquet'); pr['cluster_id']=assign[pr.node_id.to_numpy()]; pr['label']=data['labels'][pr.node_id.to_numpy()]; pr['year']=data['years'][pr.node_id.to_numpy()]
    joined=pr.merge(titles,on='node_id',how='left',validate='one_to_one'); assert len(joined)==len(pr)
    save_table(out/'influence_cluster_join.parquet',joined)
    save_table(out/'cluster_representatives.csv',joined.sort_values(['cluster_id','pagerank','node_id'],ascending=[True,False,True]).groupby('cluster_id').head(5))
    save_table(out/'rank_difference_cases.csv',rank_difference_cases(joined))
    top_pr=set(pr.nsmallest(100,'pagerank_rank').node_id); top_in=set(pr.nsmallest(100,'indegree_rank').node_id)
    save_json(out/'top100_overlap.json',dict(intersection=len(top_pr&top_in),fraction=len(top_pr&top_in)/100))
    ranking_groups=[]
    for field in ['year','label']:
        for group,rows in pr.groupby(field):
            for method,score in [('B1_in_degree','in_degree'),('A1_PageRank','pagerank')]:
                ranking_groups.append(dict(group_type=field,group=group,method=method,**rank_metrics(rows[score].to_numpy(),rows.future_citations.to_numpy(),rows.node_id.to_numpy())))
    save_table(out/'pagerank_by_year_label.csv',ranking_groups)
    pairs=pd.read_csv(final/'frequent_pairs.csv'); pairs['cluster_a']=assign[pairs.node_a]; pairs['cluster_b']=assign[pairs.node_b]; pairs['same_cluster']=pairs.cluster_a==pairs.cluster_b
    save_table(out/'frequent_pairs_clusters.csv',pairs)
    baskets,edges=eval_baskets(data['edges'],data['years'],2017); valid=data['feature_mask'][edges].all(axis=1)&np.array([bool(baskets[int(s)]) for s in edges[:,0]])
    e=edges[valid]; edgeframe=pd.DataFrame(dict(source=e[:,0],target=e[:,1],same_cluster=assign[e[:,0]]==assign[e[:,1]],source_label=data['labels'][e[:,0]],target_year=data['years'][e[:,1]],source_ref_bin=np.digitize([len(baskets[int(s)]) for s in e[:,0]],[5,10,20])))
    groups=[]
    for field in ['source_label','target_year','source_ref_bin']:
        for group,g in edgeframe.groupby(field): groups.append(dict(group_type=field,group=group,same_fraction=float(g.same_cluster.mean()),n_edges=len(g)))
    save_table(out/'citation_patterns_by_group.csv',groups)
    all_edges=data['edges']; cross=np.zeros((40,40),dtype=np.int64); np.add.at(cross,(data['labels'][all_edges[:,0]].astype(int),data['labels'][all_edges[:,1]].astype(int)),1)
    save_table(out/'subject_citation_matrix.csv',pd.DataFrame(cross,columns=[f'target_{i}' for i in range(40)]).assign(source_label=np.arange(40)))
    sizes=[]
    for cutoff in [2016,2017]:
        ids=np.load(ROOT/'outputs'/'snapshots'/f'{cutoff}.npz')['nodes']; deg=pd.read_csv(ROOT/'outputs'/'tables'/f'degrees_{cutoff}.csv')
        for column in ['in_degree','out_degree']:
            a=deg[column].to_numpy(); sizes.append(dict(cutoff=cutoff,measure=column,n=len(a),mean=a.mean(),median=np.median(a),q90=np.quantile(a,.9),q99=np.quantile(a,.99),max=a.max(),zeros=int((a==0).sum())))
    save_table(out/'degree_summary.csv',sizes)
    qm=pd.read_parquet(final/'query_metrics.parquet'); pivot=qm.pivot(index=['mask_seed','query_id'],columns='method',values=['recall10','ndcg10'])
    predictions=pd.read_parquet(final/'query_predictions.parquet')
    fills=predictions[predictions.method=='A3'].groupby(['mask_seed','query_id']).fallback.any()
    available=qm[qm.method=='A3'].set_index(['mask_seed','query_id']).query_has_applicable_rule
    cross=[]
    for seed in sorted(qm.mask_seed.unique()):
        flags=available.loc[seed]; fallback=fills.loc[seed].reindex(flags.index,fill_value=False)
        for rule in [False,True]:
            for fill in [False,True]:
                count=int(((flags==rule)&(fallback==fill)).sum())
                cross.append(dict(mask_seed=int(seed),query_has_applicable_rule=rule,a3_fallback_used=fill,n_queries=count,n_eligible_queries=len(flags),fraction=count/len(flags)))
    save_table(out/'a3_rule_and_fallback.csv',cross)
    diffs=[]
    for method in ['A2','A3']:
        for metric in ['recall10','ndcg10']:
            delta=pivot[metric,method]-pivot[metric,'B2']
            for seed,g in delta.groupby(level=0): diffs.append(dict(method=method,baseline='B2',metric=metric,mask_seed=seed,paired_mean_difference=g.mean(),n_queries=len(g),positive_queries=int((g>0).sum()),negative_queries=int((g<0).sum())))
    save_table(out/'apriori_paired_differences.csv',diffs)
    qs=pd.read_parquet(ROOT/'outputs'/'splits'/'queries_2017_0.parquet')
    failures=qm[(qm.method=='A2')&(qm.mask_seed==0)].merge(qs,on=['query_id','mask_seed'],validate='one_to_one')
    save_table(out/'rule_failure_analysis.parquet',failures)
    group_metrics=[]
    for ref_bin,g in failures.groupby(pd.cut(failures.n_refs,bins=[4,9,19,np.inf],labels=['5-9','10-19','20+']),observed=True):
        group_metrics.append(dict(reference_bin=str(ref_bin),n_queries=len(g),recall10=g.recall10.mean(),rule_coverage=g.rule_coverage.mean(),empty_rate=g['empty'].mean(),hidden_not_in_rules=int(g.n_hidden_not_in_rules.sum()),hidden_total=int(g.n_hidden.sum())))
    save_table(out/'apriori_by_reference_count.csv',group_metrics)
    metrics={}
    for name in ['pagerank','apriori','clustering']:
        df=pd.read_csv(final/f'{name}_metrics.csv')
        if name=='clustering': df=df[(df.k==sel['k'])&(df.d.isin([sel['d'],128]))&(df.method=='kmeans')]
        metrics[name]=df.to_dict('records')
    save_json(root/'metrics.json',dict(run_id=root.name,cutoff=2017,evaluation_year=2018,selection=frozen['selection'],frozen_hash=sha(root/'frozen.json'),metrics=metrics,limitations=['publication-year retrospective approximation','embedding pretraining date not guaranteed','internal citations only','five seeds indicate initialization/masking variation, not independent datasets']))
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run-id',default='arxiv_v2_20261106_review_20261107')
    p.add_argument('--rank-difference-only',action='store_true')
    args=p.parse_args(); root=ROOT/'outputs'/args.run_id
    if args.rank_difference_only: write_rank_difference_diagnostics(root)
    else: diagnostics(root)
