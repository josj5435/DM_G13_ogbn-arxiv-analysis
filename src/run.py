import argparse,copy,sys,traceback
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .common import ROOT,load_data,load_config,save_json,read_json,save_table,sha,array_hash,environment,Resource,now
from .preprocess import build_snapshot,build_baskets,eval_baskets
from .pagerank import compute_pagerank
from .evaluate import rank_metrics,rank_order
from .apriori import mine_pairs,make_rules,evaluate_rules
from .clusters import run_clusters,select_configuration
from .pairs import fixed_pairs

def frozen_path(cfg): return ROOT/'outputs'/cfg['run_id']/'frozen.json'

def run_pagerank(data,nodes,edges,qe,cfg,out,stage,alpha=None):
    n=len(data['years']); counts=np.bincount(qe[:,1],minlength=n)[nodes]
    with Resource() as baseline_res: indegree=np.bincount(edges[:,1],minlength=n)[nodes]
    base=dict(method='B1_in_degree',alpha=np.nan,**rank_metrics(indegree,counts,nodes),**baseline_res.fields())
    rows=[base]; scores=[]
    for a in cfg['pagerank']['alphas'] if alpha is None else [alpha]:
        with Resource() as r: p,meta=compute_pagerank(nodes,edges,a,cfg['pagerank']['tol'],cfg['pagerank']['max_iter'])
        rows.append(dict(method='A1_PageRank',alpha=a,**rank_metrics(p,counts,nodes),**meta,**r.fields()))
        ranks=np.empty(len(nodes),int); ranks[rank_order(p,nodes)]=np.arange(1,len(nodes)+1)
        branks=np.empty(len(nodes),int); branks[rank_order(indegree,nodes)]=np.arange(1,len(nodes)+1)
        scores.append(pd.DataFrame(dict(node_id=nodes,pagerank=p,in_degree=indegree,future_citations=counts,pagerank_rank=ranks,indegree_rank=branks,alpha=a)))
    for row in rows: row.update(stage=stage,cutoff=cfg['cutoff'],evaluation_year=cfg['cutoff']+1)
    save_table(out/'pagerank_metrics.csv',rows); save_table(out/'pagerank_scores.parquet',pd.concat(scores,ignore_index=True))
    return rows

def run_apriori(data,nodes,edges,cfg,out,stage,selection=None):
    baskets=build_baskets(nodes,edges); cutoff=cfg['cutoff']; acfg=cfg['apriori']; pilot=[]; candidates={}
    # All expansion feasibility decisions finish before any 2017 recommendation score.
    for mc in (acfg['basic_counts']+acfg['expansion_counts']) if selection is None else [selection['min_count']]:
        try:
            with Resource() as r: counts,pairs,meta=mine_pairs(baskets,mc,acfg['pilot_seconds'],acfg['pilot_rss_bytes'])
            candidates[mc]=(counts,pairs,meta); pilot.append(dict(**meta,**r.fields(),included=True,decided_at=now(),reason='train-only pair counting fits fixed resource budget'))
        except (MemoryError,TimeoutError) as e:
            pilot.append(dict(min_count=mc,included=False,decided_at=now(),reason=str(e)))
            if selection is not None: raise
    if not candidates: raise RuntimeError('No feasible Apriori candidates')
    save_table(out/'apriori_pilot.csv',pilot)
    query_sets={seed:pd.read_parquet(ROOT/'outputs'/'splits'/f'queries_{cutoff}_{seed}.parquet').to_dict('records') for seed in cfg['queries']['seeds']}
    allrows=[]; search=[]; final_per=[]; predictions=[]
    for mc,(counts,pairs,meta) in candidates.items():
        for confidence in acfg['confidences'] if selection is None else [selection['confidence']]:
            rules=make_rules(counts,pairs,len(baskets),mc,confidence)
            for seed,queries in query_sets.items():
                with Resource() as r: metrics,per,pred=evaluate_rules(rules,counts,nodes,queries,stage=='final')
                for row in metrics: allrows.append(dict(row,stage=stage,cutoff=cutoff,evaluation_year=cutoff+1,mask_seed=seed,min_count=mc,confidence=confidence,n_rules=len(rules),evaluation_seconds=r.seconds,n_eval_year_papers=int((data['years']==cutoff+1).sum()),n_excluded_queries=int((data['years']==cutoff+1).sum())-len(queries),query_inclusion_rate=len(queries)/int((data['years']==cutoff+1).sum()),min_internal_past_references=cfg['queries']['min_refs']))
                if stage=='final': final_per.append(per); predictions.extend(pred)
            a2=[r for r in allrows if r['method']=='A2' and r['min_count']==mc and r['confidence']==confidence]
            search.append(dict(algorithm='apriori',min_count=mc,confidence=confidence,recall10=np.mean([r['recall10'] for r in a2]),ndcg10=np.mean([r['ndcg10'] for r in a2]),n_rules=len(rules)))
            print(f'{stage} Apriori count={mc}, confidence={confidence}: A2 Recall={search[-1]["recall10"]:.6f}; rules={len(rules)}',flush=True)
            if stage=='final':
                save_table(out/'association_rules.csv',pd.DataFrame(rules,columns=['antecedent','consequent','support_count','support','confidence','lift']))
                save_table(out/'frequent_items.csv',[dict(node_id=n,support_count=c,support=c/len(baskets)) for n,c in sorted(counts.items()) if c>=mc])
                save_table(out/'frequent_pairs.csv',[dict(node_a=a,node_b=b,support_count=c) for (a,b),c in sorted(pairs.items())])
                save_json(out/'apriori_training.json',meta)
    save_table(out/'apriori_metrics.csv',allrows)
    if stage=='final':
        save_table(out/'query_metrics.parquet',pd.concat(final_per,ignore_index=True)); save_table(out/'query_predictions.parquet',predictions)
    return search

def run(stage,config_path):
    cfg=load_config(config_path); root=ROOT/'outputs'/cfg['run_id']; root.mkdir(exist_ok=True)
    out=root/stage
    if (out/'COMPLETE.json').exists(): raise RuntimeError('Completed run is immutable; choose a new run_id')
    out.mkdir(exist_ok=True); data=load_data(); freeze=frozen_path(cfg)
    if stage=='final':
        frozen=read_json(freeze); expected=freeze.with_suffix('.sha256').read_text().strip()
        assert sha(freeze)==expected,'Frozen configuration hash mismatch'
        assert frozen['processed_manifest_hash']==sha(ROOT/'data'/'processed'/'manifest.json')
        assert frozen['input_edges_hash']==array_hash(data['edges'])
        for name,digest in frozen['split_hashes'].items():
            assert sha(ROOT/'outputs'/'splits'/name)==digest, f'Query mask changed: {name}'
        validation_cfg=load_config(ROOT/'configs'/'validation.yaml')
        assert frozen['validation_config_hash']==sha(ROOT/'configs'/'validation.yaml')
        # final yaml only selects the saved validation configuration and final year.
        cfg=copy.deepcopy(frozen['config']); cfg['cutoff']=2017; cfg['evaluation_year']=2018
        selection=frozen['selection']
    else:
        if freeze.exists(): raise RuntimeError('Frozen settings already exist; use a new run_id')
        selection=None
    save_json(out/'execution_config.json',cfg); save_json(out/'environment.json',environment())
    nodes,edges=build_snapshot(data['edges'],data['years'],cfg['cutoff']); baskets,qe=eval_baskets(data['edges'],data['years'],cfg['cutoff'])
    pairs=None
    if stage=='final':
        # Pair pool is generated before final models, independently of assignments.
        pairs,q,meta=fixed_pairs(baskets,data['feature_mask'],cfg['clusters']['pair_max'],cfg['clusters']['pair_seed'])
        save_table(out/'eval_pairs.parquet',pairs); save_json(out/'eval_pairs_meta.json',dict(strata=meta,n_population=len(q),n_pairs=len(pairs),sha256=sha(out/'eval_pairs.parquet'),seed=cfg['clusters']['pair_seed']))
    with threadpool_limits(limits=cfg['threads']):
        pr=run_pagerank(data,nodes,edges,qe,cfg,out,stage,None if selection is None else selection['alpha'])
        ap=run_apriori(data,nodes,edges,cfg,out,stage,None if selection is None else selection['apriori'])
        km=run_clusters(data,cfg['cutoff'],out,cfg['clusters'],stage,None if selection is None else selection['clusters'],pairs,baskets,qe)
    search=[dict(algorithm='pagerank',**x) for x in pr if x['method']=='A1_PageRank']+ap+[dict(algorithm='kmeans',**x) for x in km]
    save_table(out/'validation_search.csv' if stage=='validation' else out/'experiment_registry.csv',search)
    if stage=='validation':
        prs=[x for x in pr if x['method']=='A1_PageRank']; best=max(x['ndcg100'] for x in prs)
        alpha=sorted([x for x in prs if x['ndcg100']>=best-1e-12],key=lambda x:(abs(x['alpha']-.85),x['alpha']))[0]['alpha']
        best_ap=sorted(ap,key=lambda x:(-x['recall10'],-x['ndcg10'],-x['min_count'],-x['confidence']))[0]
        best_km=select_configuration(km,cfg['clusters'])
        selection=dict(alpha=alpha,apriori={k:best_ap[k] for k in ['min_count','confidence']},clusters={k:best_km[k] for k in ['k','d']})
        save_json(freeze,dict(selection=selection,config=cfg,frozen_at=now(),validation_config_hash=sha(config_path),processed_manifest_hash=sha(ROOT/'data'/'processed'/'manifest.json'),split_hashes={p.name:sha(p) for p in sorted((ROOT/'outputs'/'splits').glob('*.parquet'))},input_edges_hash=array_hash(data['edges']),criterion='2017 only: PR NDCG; A2 five-mask mean Recall then NDCG/count/conf; shared-128D silhouette with variance filter'))
        freeze.with_suffix('.sha256').write_text(sha(freeze),encoding='utf-8')
        print('FROZEN:',selection,flush=True)
    save_json(out/'COMPLETE.json',dict(stage=stage,completed_at=now(),frozen_hash=sha(freeze),selection=selection))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['validation','final'],required=True); parser.add_argument('--config',required=True); args=parser.parse_args()
    try: run(args.stage,args.config)
    except BaseException as e:
        save_json(ROOT/'outputs'/'logs'/f'error_{args.stage}.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc())); raise
