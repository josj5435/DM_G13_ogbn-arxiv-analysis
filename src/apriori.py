from collections import Counter,defaultdict
from itertools import combinations
import time
import numpy as np
import psutil
from .evaluate import recommendation_metrics,rank_order

def mine_pairs(baskets,min_count,timeout=1800,memory_limit=8*1024**3):
    start=time.perf_counter(); counts=Counter(x for b in baskets.values() for x in set(b)); l1={x for x,c in counts.items() if c>=min_count}
    pairs=Counter(); operations=0
    for i,b in enumerate(baskets.values()):
        kept=sorted(set(b)&l1); operations+=len(kept)*(len(kept)-1)//2
        pairs.update(combinations(kept,2))
        if i%1000==0:
            if time.perf_counter()-start>timeout: raise TimeoutError('Apriori train-only budget exceeded')
            if psutil.Process().memory_info().rss>memory_limit: raise MemoryError('Apriori train-only RSS budget exceeded')
    l2={p:c for p,c in pairs.items() if c>=min_count}
    meta=dict(M=len(baskets),empty_baskets=sum(not b for b in baskets.values()),L1=len(l1),C2_theoretical=len(l1)*(len(l1)-1)//2,C2_observed=len(pairs),L2=len(l2),pair_operations=operations,min_count=min_count)
    return counts,l2,meta

def make_rules(counts,pairs,m,min_count,min_confidence):
    result=[]
    for (a,b),c in sorted(pairs.items()):
        if c<min_count: continue
        for x,y in [(a,b),(b,a)]:
            conf=c/counts[x]
            if conf>=min_confidence: result.append(dict(antecedent=x,consequent=y,support_count=c,support=c/m,confidence=conf,lift=conf/(counts[y]/m)))
    return result

def rule_index(rules):
    index=defaultdict(list)
    for r in rules: index[r['antecedent']].append((r['consequent'],r['confidence']))
    return index

def predict_references(index,observed,popularity,counts,query_id,k=10):
    exclude=set(observed)|{query_id}; scores={}; applied=0
    for a in observed:
        for v,score in index.get(a,[]):
            if v not in exclude:
                applied+=1; scores[v]=max(scores.get(v,0),score)
    pure=sorted(scores,key=lambda v:(-scores[v],-counts.get(v,0),v))[:k]
    popular=[]
    for v in popularity:
        if int(v) not in exclude: popular.append(int(v))
        if len(popular)==k: break
    hybrid=list(pure)
    for v in popular:
        if len(hybrid)>=k: break
        if v not in hybrid: hybrid.append(v)
    return dict(B2=popular,A2=pure,A3=hybrid),scores,applied

def evaluate_rules(rules,counts,nodes,queries,save_predictions=False):
    idx=rule_index(rules); pop=nodes[rank_order(np.array([counts.get(int(n),0) for n in nodes]),nodes)]
    per=[]; predictions=[]; catalog={m:set() for m in ['B2','A2','A3']}
    rule_targets={r['consequent'] for r in rules}
    for q in queries:
        methods,scores,applied=predict_references(idx,q['observed'],pop,counts,q['query_id'])
        for method,pred in methods.items():
            metrics=recommendation_metrics(pred,q['hidden'])
            per.append(dict(query_id=q['query_id'],mask_seed=q['mask_seed'],method=method,**metrics,rule_coverage=(int(bool(scores)) if method in ('A2','A3') else float('nan')),query_has_applicable_rule=bool(scores),empty=int(not pred),applied_rules=applied,n_rule_candidates=len(scores),n_hidden=len(q['hidden']),n_hidden_not_in_rules=sum(h not in rule_targets for h in q['hidden'])))
            catalog[method].update(pred)
            if save_predictions:
                for rank,v in enumerate(pred,1): predictions.append(dict(query_id=q['query_id'],mask_seed=q['mask_seed'],method=method,rank=rank,candidate=v,score=scores.get(v,0) if method!='B2' else counts.get(v,0),training_count=counts.get(v,0),hit=v in q['hidden'],fallback=method=='A3' and v not in scores,tie_rule='score DESC; training_count DESC; node_id ASC'))
    import pandas as pd
    frame=pd.DataFrame(per)
    rows=[]
    for method,g in frame.groupby('method'):
        rows.append(dict(method=method,**{c:float(g[c].mean()) for c in ['recall10','ndcg10','precision10','rule_coverage','empty']},query_has_applicable_rule_rate=float(g.query_has_applicable_rule.mean()),catalog_coverage=len(catalog[method])/len(nodes),n_queries=len(g),n_candidates=len(nodes)))
    return rows,frame,predictions
