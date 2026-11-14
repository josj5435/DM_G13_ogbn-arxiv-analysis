import inspect
import numpy as np
import networkx as nx
import pandas as pd
import pytest
from sklearn.metrics import ndcg_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from src.pagerank import compute_pagerank
from src.evaluate import linear_ndcg,rank_order,recommendation_metrics,pair_summary
from src.apriori import mine_pairs,make_rules,evaluate_rules,predict_references,rule_index
from src.preprocess import clean_edges,build_snapshot,build_baskets,make_queries
from src.clusters import select_configuration,normalize_features
from src.pairs import fixed_pairs
from src.common import array_hash

def test_pagerank_hand_dangling():
    p,m=compute_pagerank(np.arange(2),np.array([[0,1]]),tol=1e-12)
    np.testing.assert_allclose(p,[.35087719298,.64912280702],atol=1e-10)
    assert m['fixed_point_residual']<1e-12 and abs(p.sum()-1)<1e-12

def test_pagerank_networkx_isolates_direction():
    nodes=np.arange(5); edges=np.array([[0,1],[1,2],[2,1],[3,2]])
    g=nx.DiGraph(); g.add_nodes_from(nodes); g.add_edges_from(edges)
    p,m=compute_pagerank(nodes,edges,tol=1e-12)
    other=nx.pagerank(g,tol=1e-12/len(nodes),max_iter=1000)
    np.testing.assert_allclose(p,[other[i] for i in nodes],atol=1e-12)
    assert p[1]>p[0] and p[4]>0

def test_pagerank_failure():
    with pytest.raises(RuntimeError): compute_pagerank(np.arange(2),np.array([[0,1]]),max_iter=1)

def test_ndcg_hand_linear_gain():
    r=np.array([2,1,0]); scores=np.array([1,2,0]); ids=np.arange(3)
    expected=(1+2/np.log2(3))/(2+1/np.log2(3))
    assert linear_ndcg(r,scores,ids,2)==pytest.approx(expected)
    assert linear_ndcg(r,scores,ids,2)==pytest.approx(ndcg_score(r[None,:],scores[None,:],k=2))

def test_ndcg_ties_and_zero():
    ids=np.array([5,2,8]); scores=np.ones(3); r=np.array([2,1,0])
    assert rank_order(scores,ids).tolist()==[1,0,2]
    distinct=np.empty(3); distinct[rank_order(scores,ids)]=[3,2,1]
    assert linear_ndcg(r,scores,ids,2)==pytest.approx(ndcg_score(r[None,:],distinct[None,:],k=2))
    assert np.isnan(linear_ndcg(np.zeros(3),scores,ids))

def test_apriori_hand_empty_basket():
    b={0:(1,2),1:(1,2),2:(1,),3:()}
    counts,pairs,meta=mine_pairs(b,2)
    assert counts[1]==3 and counts[2]==2 and pairs[(1,2)]==2
    assert meta['M']==4 and meta['empty_baskets']==1 and meta['L1']==2
    r=make_rules(counts,pairs,4,2,.1); a=next(x for x in r if x['antecedent']==1); c=next(x for x in r if x['antecedent']==2)
    assert a['support']==.5 and a['confidence']==pytest.approx(2/3)
    assert c['confidence']==1 and c['lift']==pytest.approx(4/3)
    assert make_rules(counts,pairs,3,2,.1)[0]['support']!=a['support']

def test_apriori_exhaustive_pruning():
    from itertools import combinations
    rng=np.random.default_rng(10); baskets={i:tuple(sorted(rng.choice(12,rng.integers(0,8),replace=False))) for i in range(100)}
    c,p,m=mine_pairs(baskets,9)
    brute={pair:sum(set(pair)<=set(b) for b in baskets.values()) for pair in combinations(range(12),2)}
    assert p=={pair:count for pair,count in brute.items() if count>=9}

def test_mask_recommendations_missing_truth():
    b={20:tuple(range(10)),21:(1,2),22:tuple(range(1,7))}; q=make_queries(b,2)
    assert len(q)==2 and all(not set(x['observed'])&set(x['hidden']) for x in q)
    assert q==make_queries(b,2)
    empty=recommendation_metrics([], [100,101]); assert empty=={'recall10':0.,'precision10':0.,'ndcg10':0.}
    short=recommendation_metrics([100], [100,101]); assert short['recall10']==.5 and short['precision10']==.1
    rules=[dict(antecedent=1,consequent=3,confidence=.3),dict(antecedent=2,consequent=3,confidence=.9)]
    methods,scores,applied=predict_references(rule_index(rules),[1,2],np.arange(6),{3:10},20)
    assert methods['A2']==[3] and scores[3]==.9 and not set(methods['A3'])&{1,2}

def test_temporal_leakage_and_cleaning():
    years=np.array([2015,2016,2017,2018,2017])
    edges=np.array([[0,0],[1,0],[1,0],[2,1],[3,0],[0,3],[99,0]])
    e,stats=clean_edges(edges,years); nodes,se=build_snapshot(e,years,2016)
    assert se.tolist()==[[1,0]] and nodes.tolist()==[0,1]
    assert stats['self_loops']==1 and stats['duplicate_edges']==1 and stats['reversed_year_edges']==1
    changed=np.vstack([edges,[3,1]]); e2,_=clean_edges(changed,years)
    np.testing.assert_equal(build_snapshot(e2,years,2016)[1],se)
    assert len(build_baskets(nodes,se))==2

def test_hybrid_never_exceeds_top_k():
    rules=[dict(antecedent=1,consequent=i,confidence=.5) for i in range(20,35)]
    methods,_,_=predict_references(rule_index(rules),[1],np.arange(50),{},100)
    assert len(methods['A2'])==10 and len(methods['A3'])==10
    assert methods['A3']==methods['A2']

def test_selection_without_labels():
    rows=[dict(k=20,d=128,silhouette=.1,explained_variance=1),dict(k=40,d=32,silhouette=.095,explained_variance=.91),dict(k=20,d=16,silhouette=.2,explained_variance=.89),dict(k=20,d=64,silhouette=.099,explained_variance=.95)]
    selected=select_configuration(rows,dict(variance_min=.9,silhouette_tolerance=.01))
    assert selected['d']==32 and selected['k']==40

def test_train_only_pca_and_future_labels():
    x=np.random.default_rng(1).normal(size=(30,8)); train=np.arange(20); eval_ids=np.arange(20,30)
    tx=normalize_features(x,train); pca=PCA().fit(tx); km=KMeans(n_clusters=3,random_state=0,n_init=2).fit(pca.transform(tx))
    before=array_hash(pca.components_)+array_hash(km.cluster_centers_)
    x[eval_ids]*=100; km.predict(pca.transform(normalize_features(x,eval_ids)))
    assert before==array_hash(pca.components_)+array_hash(km.cluster_centers_)

def test_fixed_pair_population_and_null():
    baskets={i:tuple(range(i%5+1)) for i in range(30)}
    pairs,ids,meta=fixed_pairs(baskets,np.ones(30,bool),max_pairs=150,seed=9)
    p2,_,_=fixed_pairs(baskets,np.ones(30,bool),max_pairs=150,seed=9)
    pd.testing.assert_frame_equal(pairs,p2); assert not pairs[['node_a','node_b']].duplicated().any()
    assert np.all(pairs.node_a<pairs.node_b) and len(pairs)==150
    labels=np.arange(30)%3; shuffled=np.random.default_rng(0).permutation(labels)
    a=pair_summary(pairs,labels); b=pair_summary(pairs,shuffled)
    assert a['same_n']+a['different_n']==len(pairs) and b['same_n']+b['different_n']==len(pairs)
    pd.testing.assert_frame_equal(pairs,p2)
