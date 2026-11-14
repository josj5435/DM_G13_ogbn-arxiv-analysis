import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score

def rank_order(scores,ids): return np.lexsort((ids,-np.asarray(scores,dtype=float)))

def linear_ndcg(relevance,scores,ids,k=100):
    r=np.asarray(relevance,float); k=min(k,len(r))
    discount=1/np.log2(np.arange(k)+2)
    ideal=float(np.dot(np.sort(r)[::-1][:k],discount))
    return float(np.dot(r[rank_order(scores,ids)[:k]],discount)/ideal) if ideal>0 else float('nan')

def rank_metrics(scores,counts,ids):
    rho=float(spearmanr(scores,counts).statistic) if np.ptp(scores)>0 and np.ptp(counts)>0 else float('nan')
    return dict(ndcg100=linear_ndcg(np.log2(1+counts),scores,ids),spearman=rho,n_candidates=len(ids),k=min(100,len(ids)))

def recommendation_metrics(prediction,hidden,k=10):
    hidden=set(hidden); hits=np.array([int(x in hidden) for x in prediction[:k]])
    ideal=np.sum(1/np.log2(np.arange(min(k,len(hidden)))+2))
    return dict(recall10=float(hits.sum()/len(hidden)),precision10=float(hits.sum()/k),ndcg10=float(np.sum(hits/np.log2(np.arange(len(hits))+2))/ideal))

def common_silhouette(x,labels):
    n=len(np.unique(labels))
    if n<2 or n>=len(labels): return float('nan')
    return float(silhouette_score(x,labels,metric='euclidean'))

def weighted_group(values,weights):
    if len(values)==0: return dict(mean=float('nan'),positive=float('nan'),n=0,effective_n=0,status='invalid')
    return dict(mean=float(np.average(values,weights=weights)),positive=float(np.average(values>0,weights=weights)),n=len(values),effective_n=float(weights.sum()**2/np.square(weights).sum()),status='low_support' if len(values)<100 else 'ok')

def pair_summary(pairs,labels):
    a=pairs.node_a.to_numpy(); b=pairs.node_b.to_numpy(); same=labels[a]==labels[b]
    j=pairs.jaccard.to_numpy(); w=pairs.weight.to_numpy(); result={}
    for tag,m in [('same',same),('different',~same)]:
        result.update({f'{tag}_{k}':v for k,v in weighted_group(j[m],w[m]).items()})
        result[f'{tag}_unweighted_mean']=float(j[m].mean()) if m.any() else float('nan')
    result['jaccard_statistic']='inverse_probability_weighted_mean_of_ordinary_set_jaccard'
    result['jaccard_difference']=result['same_mean']-result['different_mean']
    result['same_weighted_fraction']=float(np.average(same,weights=w))
    return result
