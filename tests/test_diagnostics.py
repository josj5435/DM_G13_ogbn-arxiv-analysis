import numpy as np
import pandas as pd
import pytest
from src.diagnostics import rank_difference_cases
from src.evaluate import rank_order, linear_ndcg

def sample():
    return pd.DataFrame(dict(node_id=[99,1,50,2],record=['a','b','c','d'],
        in_degree=[4,4,1,0],pagerank=[.1,.4,.3,.2],
        indegree_rank=[2,1,3,4],pagerank_rank=[4,1,2,3]))


def test_average_rank_and_original_columns_preserved():
    original=sample(); result=rank_difference_cases(original,4).set_index('record')
    assert result.loc['a','indegree_average_rank']==result.loc['b','indegree_average_rank']==1.5
    assert result.loc['a','rank_difference']==-2.5
    assert result.loc['a','indegree_rank']==2
    assert 'rank_difference' not in original


def test_case_membership_invariant_to_node_ids_and_row_order():
    data=sample(); expected=set(rank_difference_cases(data,2).record)
    data['node_id']=[1,99,2,50]
    # Even the official ordinal ranks can change without affecting diagnostics.
    data['indegree_rank']=[1,2,3,4]
    assert set(rank_difference_cases(data.sample(frac=1,random_state=7),2).record)==expected


def test_boundary_ties_are_all_kept_including_pagerank_ties():
    data=pd.DataFrame(dict(node_id=[9,1,5],in_degree=[1,1,1],pagerank=[.5,.5,.5],
                           indegree_rank=[3,1,2],pagerank_rank=[3,1,2]))
    result=rank_difference_cases(data,1)
    assert len(result)==3
    assert (result.indegree_average_rank==2).all()
    assert (result.pagerank_average_rank==2).all()
    assert (result.rank_difference==0).all()


def test_official_ndcg_still_uses_node_id_tie_break():
    scores=np.array([1.,1.,0.]); ids=np.array([20,10,30]); relevance=np.array([1.,3.,0.])
    assert rank_order(scores,ids).tolist()==[1,0,2]
    assert linear_ndcg(relevance,scores,ids,k=1)==1.
    assert linear_ndcg(relevance,scores,np.array([10,20,30]),k=1)==pytest.approx(1/3)
