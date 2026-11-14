"""Model-independent stratified unordered pairs, with inclusion weights."""
import numpy as np
from .common import save_table,save_json,sha
from .evaluate import pair_summary

def bins_for(ids,baskets): return np.digitize([len(baskets[int(i)]) for i in ids],[5,10,20])

def fixed_pairs(baskets,feature_mask,max_pairs=100000,seed=202):
    ids=np.array(sorted(q for q,b in baskets.items() if b and feature_mask[q]),dtype=np.int64)
    groups=[ids[bins_for(ids,baskets)==i] for i in range(4)]
    strata=[(i,j) for i in range(4) for j in range(i,4)]
    populations=[len(groups[i])*(len(groups[i])-1)//2 if i==j else len(groups[i])*len(groups[j]) for i,j in strata]
    total=min(max_pairs,sum(populations)); counts=[min(p,total//len(strata)) for p in populations]
    while sum(counts)<total:
        for z,p in enumerate(populations):
            if counts[z]<p: counts[z]+=1
            if sum(counts)==total: break
    rng=np.random.default_rng(seed); rows=[]; meta=[]
    for (i,j),population,sample in zip(strata,populations,counts):
        if not sample: continue
        # Rejection sampling on node pairs is uniform within the unordered stratum.
        chosen=set()
        while len(chosen)<sample:
            batch=max(100,(sample-len(chosen))*2)
            aa=rng.choice(groups[i],batch); bb=rng.choice(groups[j],batch)
            for a,b in zip(aa,bb):
                if a==b: continue
                chosen.add((min(int(a),int(b)),max(int(a),int(b))))
                if len(chosen)==sample: break
        for a,b in sorted(chosen):
            ba=set(baskets[a]); bb=set(baskets[b]); jac=len(ba&bb)/len(ba|bb)
            rows.append(dict(node_a=a,node_b=b,stratum=f'{i}-{j}',population=population,sample_size=sample,inclusion_probability=sample/population,weight=population/sample,jaccard=jac))
        meta.append(dict(stratum=f'{i}-{j}',population=population,sample_size=sample,inclusion_probability=sample/population))
    import pandas as pd
    return pd.DataFrame(rows),ids,meta

def external_patterns(pairs,eval_ids,train_ids,assignment,baskets,eval_edges,feature_mask,seed):
    actual=pair_summary(pairs,assignment)
    result=[dict(kind='bibliographic',baseline='observed',random_seed=-1,**actual)]
    rng=np.random.default_rng(seed)
    q=np.array(sorted(x for x,b in baskets.items() if b and feature_mask[x]))
    valid=feature_mask[eval_edges].all(axis=1)&np.isin(eval_edges[:,0],q)
    e=eval_edges[valid]
    strata=bins_for(q,baskets)
    same=assignment[e[:,0]]==assignment[e[:,1]]
    result.append(dict(kind='direct_citation',baseline='observed',random_seed=-1,same_fraction=float(same.mean()),n_edges=len(e),excluded_edges=len(eval_edges)-len(e)))
    for rseed in range(5):
        rng=np.random.default_rng(rseed)
        perm=assignment.copy(); perm[eval_ids]=rng.permutation(assignment[eval_ids])
        assert np.array_equal(np.sort(perm[eval_ids]),np.sort(assignment[eval_ids]))
        result.append(dict(kind='bibliographic',baseline='size_preserving_random',random_seed=rseed,**pair_summary(pairs,perm)))
        perm=assignment.copy()
        for stratum in range(4):
            nodes=q[strata==stratum]; perm[nodes]=rng.permutation(assignment[nodes])
        result.append(dict(kind='direct_citation',baseline='ref_bin_source_shuffle',random_seed=rseed,same_fraction=float((perm[e[:,0]]==perm[e[:,1]]).mean()),n_edges=len(e),excluded_edges=len(eval_edges)-len(e)))
    return result
