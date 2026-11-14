"""Recalculate selected final models and audit immutable saved outputs."""
import argparse,itertools
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from .common import ROOT,load_data,read_json,sha,save_json,save_table,array_hash,now
from .preprocess import build_snapshot,eval_baskets,build_baskets,make_queries,clean_edges
from .pagerank import compute_pagerank
from .apriori import mine_pairs,make_rules,evaluate_rules
from .clusters import normalize_features
from .pairs import fixed_pairs
from .make_figures import select_graph_targets,extract_display_subgraph

def verify(root, report):
    data=load_data(); final=root/'final'; frozen=read_json(root/'frozen.json'); cfg=frozen['config']; selection=frozen['selection']; checks=[]
    def check(name,condition,**details):
        checks.append(dict(check=name,passed=bool(condition),**details)); save_json(report,dict(status='running',checks=checks))
        if not condition: raise AssertionError(name)
    before=sha(root/'frozen.json'); check('frozen_hash',before==(root/'frozen.sha256').read_text().strip())
    check('complete_final_exists',(final/'COMPLETE.json').exists())
    for name,h in frozen['split_hashes'].items(): check(f'split_hash_{name}',sha(ROOT/'outputs'/'splits'/name)==h)
    original=np.load(ROOT/'data'/'original_arrays'/'edges.npy'); recleaned,stats=clean_edges(original,data['years'])
    check('preprocessing_exact_replay',np.array_equal(recleaned,data['edges']))
    nodes,edges=build_snapshot(data['edges'],data['years'],2017); baskets,qe=eval_baskets(data['edges'],data['years'],2017)
    check('future_edges_excluded',np.all(data['years'][edges]<=2017))
    old_scores=pd.read_parquet(final/'pagerank_scores.parquet').sort_values('node_id')
    p,meta=compute_pagerank(nodes,edges,selection['alpha'])
    check('pagerank_full_snapshot_replay',np.allclose(p,old_scores.pagerank,rtol=0,atol=1e-14),maximum_absolute_difference=float(np.max(np.abs(p-old_scores.pagerank.to_numpy()))),**meta)
    for seed in range(5):
        old=pd.read_parquet(ROOT/'outputs'/'splits'/f'queries_2017_{seed}.parquet').to_dict('records'); new=make_queries(baskets,seed)
        check(f'query_mask_replay_{seed}',len(old)==len(new) and all(a['query_id']==b['query_id'] and list(a['observed'])==b['observed'] and list(a['hidden'])==b['hidden'] for a,b in zip(old,new)))
    pairs,ids,pmeta=fixed_pairs(baskets,data['feature_mask']); old_pairs=pd.read_parquet(final/'eval_pairs.parquet')
    check('fixed_pair_replay',pairs.equals(old_pairs),n_pairs=len(pairs))
    with threadpool_limits(limits=cfg['threads']):
        train=np.flatnonzero(data['feature_mask']&data['year_mask']&(data['years']<=2017)); evaluation=np.flatnonzero(data['feature_mask']&(data['years']==2018))
        x=normalize_features(data['features'],train); y=normalize_features(data['features'],evaluation)
        saved_pca=joblib.load(final/'models'/'pca.joblib'); pca=PCA(n_components=128,svd_solver='full',whiten=False).fit(x)
        check('train_only_pca_replay',np.allclose(pca.mean_,saved_pca.mean_,rtol=0,atol=1e-12) and np.allclose(pca.components_,saved_pca.components_,rtol=0,atol=1e-10))
        k=selection['clusters']['k']; d=selection['clusters']['d']; tx=x if d==128 else pca.transform(x)[:,:d]; ty=y if d==128 else pca.transform(y)[:,:d]
        km=KMeans(n_clusters=k,init='k-means++',n_init=10,max_iter=300,tol=1e-4,random_state=0,algorithm='lloyd').fit(tx)
        tag=f'k{k}_d{d}_s0'; saved=joblib.load(final/'models'/f'{tag}.joblib'); saved_a=np.load(final/'models'/f'{tag}_assignments.npz')['assignment']
        check('selected_kmeans_replay',np.allclose(km.cluster_centers_,saved.cluster_centers_,atol=1e-10,rtol=0) and np.array_equal(km.predict(ty),saved_a[evaluation]))
        # Perturb only future features/labels/references: the training matrices remain identical.
        altered_features=np.array(data['features']); altered_features[evaluation]=0
        check('future_feature_perturbation_preserves_train',np.array_equal(normalize_features(altered_features,train),x))
        changed_edges=data['edges'][data['years'][data['edges'][:,0]]!=2018]
        check('future_reference_perturbation_preserves_snapshot',np.array_equal(build_snapshot(changed_edges,data['years'],2017)[1],edges))
        check('future_labels_not_model_input',np.all(data['years'][train]<=2017) and np.intersect1d(train,evaluation).size==0)
        counts,l2,apmeta=mine_pairs(build_baskets(nodes,edges),selection['apriori']['min_count'])
        rules=make_rules(counts,l2,len(nodes),selection['apriori']['min_count'],selection['apriori']['confidence'])
        old_rules=pd.read_csv(final/'association_rules.csv'); replay=pd.DataFrame(rules)
        check('apriori_exact_rule_replay',len(replay)==len(old_rules) and np.allclose(replay.to_numpy(),old_rules.to_numpy(),atol=1e-12,rtol=1e-12),n_rules=len(rules))
        allmetrics=[]
        for seed in range(5):
            metrics,_,_=evaluate_rules(rules,counts,nodes,make_queries(baskets,seed))
            allmetrics.extend(dict(m,mask_seed=seed) for m in metrics)
        replay=pd.DataFrame(allmetrics).sort_values(['mask_seed','method']); old=pd.read_csv(final/'apriori_metrics.csv').sort_values(['mask_seed','method'])
        columns=['recall10','ndcg10','precision10','rule_coverage','query_has_applicable_rule_rate','catalog_coverage','empty']
        check('all_five_masks_recommendation_replay',np.allclose(replay[columns],old[columns],atol=1e-14,rtol=0,equal_nan=True))
    predictions=pd.read_parquet(final/'query_predictions.parquet')
    check('recommendation_maximum_ten',bool((predictions.groupby(['query_id','mask_seed','method']).size()<=10).all()))
    check('recommendation_candidate_valid',np.isin(predictions.candidate.to_numpy(),nodes).all())
    overlap=0
    for seed in range(5):
        qs={q['query_id']:set(q['observed']) for q in make_queries(baskets,seed)}
        pp=predictions[predictions.mask_seed==seed]
        overlap+=sum(int(v) in qs[int(q)] for q,v in zip(pp.query_id,pp.candidate))
    check('observed_references_never_recommended',overlap==0)
    import networkx as nx
    g=nx.DiGraph(); g.add_nodes_from(nodes.tolist()); g.add_edges_from(edges.tolist())
    selected=select_graph_targets(nodes,edges,saved_a,cfg['visualization']); stored=read_json(root/'visualization'/'selection.json')
    check('visual_selection_replay',selected['clusters']==stored['clusters'] and selected['papers']==stored['papers'])
    for mode,targets in [('cluster',selected['clusters']),('paper',selected['papers'])]:
        for target in targets:
            chosen,shown,meta=extract_display_subgraph(g,saved_a,mode,target,cfg['visualization']); folder=root/'visualization'/f'{mode}_{target}'
            ns=pd.read_csv(folder/'nodes.csv'); es=pd.read_csv(folder/'edges.csv')
            check(f'{mode}_{target}_subset_replay',chosen==ns.original_node_id.tolist() and shown==list(es.itertuples(index=False,name=None)) and len(chosen)<=80 and len(shown)<=250 and all(g.has_edge(a,b) for a,b in shown))
            graph=nx.DiGraph(); graph.add_nodes_from(chosen); graph.add_edges_from(shown); pos=nx.spring_layout(graph.to_undirected(),seed=44,iterations=100)
            oldpos=pd.read_csv(folder/'positions.csv').set_index('original_node_id')
            check(f'{mode}_{target}_layout_replay',all(np.allclose(pos[n],oldpos.loc[n,['x','y']].to_numpy(),atol=1e-12,rtol=0) for n in chosen))
    check('frozen_hash_unchanged',sha(root/'frozen.json')==before)
    save_json(report,dict(status='passed',completed_at=now(),n_checks=len(checks),checks=checks,scope='Separate-process replay of preprocessing, PageRank, PCA, selected k-means seed 0, Apriori rules/all five masks, pair pool, graph selection and layouts. Sensitivity model fits were not all rerun.'))
    print(f'Verification: {len(checks)} checks passed',flush=True)

if __name__=='__main__':
    from .common import REFERENCE_RUN
    p=argparse.ArgumentParser(); p.add_argument('--run-id',default=REFERENCE_RUN)
    p.add_argument('--report',type=Path,default=ROOT/'outputs/checks/model_replay.json')
    args=p.parse_args()
    if args.report.exists(): raise FileExistsError('Choose a new --report path to preserve the previous check')
    verify(ROOT/'outputs'/args.run_id,args.report)
