"""Separately measure B2/A2/A3 inference on identical complete query sets."""
import itertools
import numpy as np
import pandas as pd
from .common import ROOT,read_json,load_data,Resource,save_table
from .preprocess import build_snapshot,build_baskets,eval_baskets,make_queries
from .apriori import mine_pairs,make_rules,rule_index
from .evaluate import rank_order,recommendation_metrics

def benchmark(root):
    data=load_data(); frozen=read_json(root/'frozen.json'); ac=frozen['selection']['apriori']
    nodes,edges=build_snapshot(data['edges'],data['years'],2017); baskets,_=eval_baskets(data['edges'],data['years'],2017)
    with Resource() as popfit:
        counts=np.bincount(edges[:,1],minlength=len(data['years'])); popularity=nodes[rank_order(counts[nodes],nodes)]
    with Resource() as rulefit:
        counter,pairs,meta=mine_pairs(build_baskets(nodes,edges),ac['min_count']); rules=make_rules(counter,pairs,len(nodes),ac['min_count'],ac['confidence']); index=rule_index(rules)
    expected=pd.read_csv(root/'final'/'apriori_metrics.csv'); rows=[]
    for seed in range(5):
        queries=make_queries(baskets,seed)
        for method in ['B2','A2','A3']:
            metrics=[]
            with Resource() as r:
                for q in queries:
                    excluded=set(q['observed'])|{q['query_id']}; scores={}
                    if method!='B2':
                        for a in q['observed']:
                            for v,s in index.get(a,[]):
                                if v not in excluded: scores[v]=max(scores.get(v,0),s)
                        pred=sorted(scores,key=lambda v:(-scores[v],-counts[v],v))[:10]
                    else: pred=[]
                    if method in ['B2','A3']:
                        for v in popularity:
                            if len(pred)>=10: break
                            if v not in excluded and v not in pred: pred.append(int(v))
                    metrics.append(recommendation_metrics(pred,q['hidden']))
            means=pd.DataFrame(metrics).mean(); target=expected[(expected.method==method)&(expected.mask_seed==seed)].iloc[0]
            assert np.allclose(means.to_numpy(),target[means.index].to_numpy(dtype=float),rtol=0,atol=1e-14)
            fit=popfit.seconds if method=='B2' else rulefit.seconds+(popfit.seconds if method=='A3' else 0)
            rows.append(dict(method=method,mask_seed=seed,n_queries=len(queries),n_candidates=len(nodes),training_seconds=fit,inference_and_metric_seconds=r.seconds,total_seconds=fit+r.seconds,peak_rss_bytes=r.peak,metric_replay_match=True,scope='separate whole-method inference plus identical metric evaluation; training measured once and shared across masks'))
    save_table(root/'final'/'recommendation_resource_comparison.csv',rows)

if __name__=='__main__':
    import argparse
    from .common import REFERENCE_RUN
    parser=argparse.ArgumentParser(); parser.add_argument('--run-id',default=REFERENCE_RUN)
    args=parser.parse_args(); benchmark(ROOT/'outputs'/args.run_id)
