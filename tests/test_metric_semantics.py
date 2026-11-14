import numpy as np
import pandas as pd
import pytest
import uuid
from src.apriori import evaluate_rules
from src.common import ROOT,save_table
from src.evaluate import pair_summary

def fixture_inputs():
    rules=[dict(antecedent=1,consequent=2,confidence=.7)]
    counts={1:4,2:3,3:2}; nodes=np.arange(7)
    queries=[dict(query_id=10,mask_seed=0,observed=[1],hidden=[2,6]),
             dict(query_id=11,mask_seed=0,observed=[3],hidden=[6]),
             dict(query_id=12,mask_seed=0,observed=[1],hidden=[6])]
    return rules,counts,nodes,queries

def test_rule_coverage_is_not_defined_for_popularity():
    rows,per,_=evaluate_rules(*fixture_inputs())
    metrics={r['method']:r for r in rows}
    assert np.isnan(metrics['B2']['rule_coverage'])
    assert per.loc[per.method=='B2','rule_coverage'].isna().all()
    for method in ['A2','A3']:
        assert metrics[method]['rule_coverage']==pytest.approx(2/3)
    for method in ['B2','A2','A3']:
        assert metrics[method]['query_has_applicable_rule_rate']==pytest.approx(2/3)
        assert per.loc[per.method==method,'query_has_applicable_rule'].tolist()==[True,False,True]

def test_rule_availability_does_not_imply_hit():
    rows,per,_=evaluate_rules(*fixture_inputs())
    missed=per[(per.query_id==12)&(per.method=='A2')].iloc[0]
    assert missed.query_has_applicable_rule and missed.rule_coverage==1 and missed.recall10==0
    a2=next(r for r in rows if r['method']=='A2')
    assert a2['recall10']==pytest.approx(1/6)
    assert a2['empty']==pytest.approx(1/3)

def test_fallback_does_not_increase_rule_coverage():
    _,counts,nodes,queries=fixture_inputs()
    rows,per,_=evaluate_rules([],counts,nodes,queries)
    metrics={r['method']:r for r in rows}
    assert np.isnan(metrics['B2']['rule_coverage'])
    assert metrics['A2']['rule_coverage']==metrics['A3']['rule_coverage']==0
    assert metrics['A2']['empty']==1 and metrics['A3']['empty']==0
    assert not per.query_has_applicable_rule.any()

def test_csv_explicit_na_and_roundtrip():
    rows,_,_=evaluate_rules(*fixture_inputs())
    path=ROOT/'outputs'/'semantic_test_artifacts'/f'metrics_{uuid.uuid4().hex}.csv'
    save_table(path,rows)
    assert 'N/A' in path.read_text(encoding='utf-8-sig')
    parsed=pd.read_csv(path)
    assert parsed.loc[parsed.method=='B2','rule_coverage'].isna().all()
    assert parsed.loc[parsed.method=='A2','rule_coverage'].iloc[0]==pytest.approx(2/3)

def test_jaccard_weights_apply_to_pair_means():
    pairs=pd.DataFrame(dict(node_a=[0,0,0],node_b=[1,2,3],jaccard=[.2,.8,.5],weight=[9.,1.,1.]))
    before=pairs.copy(deep=True)
    result=pair_summary(pairs,np.array([0,0,0,1]))
    assert result['same_mean']==pytest.approx(.26)
    assert result['same_unweighted_mean']==pytest.approx(.5)
    assert result['jaccard_difference']==pytest.approx(-.24)
    assert result['jaccard_statistic']=='inverse_probability_weighted_mean_of_ordinary_set_jaccard'
    pd.testing.assert_frame_equal(pairs,before)
