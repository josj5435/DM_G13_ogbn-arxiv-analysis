"""Check the submitted files and reported metrics without downloading raw data."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .common import ROOT, REFERENCE_RUN, read_json, sha
from .submission_manifest import verify_manifest
from .diagnostics import rank_difference_cases
from .evaluate import rank_metrics


def check_submission():
    integrity=verify_manifest(ROOT)
    root=ROOT/'outputs'/REFERENCE_RUN; final=root/'final'
    frozen=read_json(root/'frozen.json')
    assert sha(root/'frozen.json')==(root/'frozen.sha256').read_text().strip()
    assert sha(ROOT/'configs/validation.yaml')==frozen['validation_config_hash']
    assert sha(ROOT/'outputs/preprocessing.json')==frozen['processed_manifest_hash']
    for name,digest in frozen['split_hashes'].items():
        assert sha(ROOT/'outputs/splits'/name)==digest,name
    assert frozen['selection']==dict(alpha=.7,apriori=dict(min_count=20,confidence=.1),clusters=dict(k=20,d=128))
    assert read_json(final/'COMPLETE.json')['frozen_hash']==sha(root/'frozen.json')
    pr=pd.read_csv(final/'pagerank_metrics.csv').set_index('method')
    scores=pd.read_parquet(final/'pagerank_scores.parquet')
    for method,column in [('B1_in_degree','in_degree'),('A1_PageRank','pagerank')]:
        metrics=rank_metrics(scores[column].to_numpy(),scores.future_citations.to_numpy(),scores.node_id.to_numpy())
        for name in ['ndcg100','spearman']: assert np.isclose(metrics[name],pr.loc[method,name],atol=1e-14,rtol=0)
    assert np.isclose(pr.loc['A1_PageRank','ndcg100'],.597558,atol=5e-7,rtol=0)
    assert np.isclose(pr.loc['B1_in_degree','ndcg100'],.924337,atol=5e-7,rtol=0)
    ap=pd.read_csv(final/'apriori_metrics.csv'); per=pd.read_parquet(final/'query_metrics.parquet')
    assert ap.loc[ap.method=='B2','rule_coverage'].isna().all()
    assert per.loc[per.method=='B2','rule_coverage'].isna().all()
    assert set(ap.n_queries)=={15072} and set(ap.n_eval_year_papers)=={29799}
    assert np.allclose(ap.query_inclusion_rate,15072/29799)
    columns=['recall10','ndcg10','precision10','empty','rule_coverage','query_has_applicable_rule_rate']
    for (seed,method),g in per.groupby(['mask_seed','method']):
        row=ap[(ap.mask_seed==seed)&(ap.method==method)].iloc[0]
        for column in columns:
            actual=g['query_has_applicable_rule' if column.endswith('_rate') else column].mean()
            assert np.isclose(actual,row[column],atol=1e-14,rtol=0,equal_nan=True)
    expected={'A2':.189744,'A3':.192853,'B2':.083361}
    for method,value in expected.items(): assert np.isclose(ap[ap.method==method].recall10.mean(),value,atol=5e-7,rtol=0)
    js=pd.DataFrame(read_json(root/'metrics.json')['metrics']['apriori']).sort_values(['mask_seed','method'])
    csv=ap.sort_values(['mask_seed','method'])
    assert np.allclose(js[columns].astype(float),csv[columns].astype(float),atol=1e-14,rtol=0,equal_nan=True)
    predictions=pd.read_parquet(final/'query_predictions.parquet')
    assert predictions['rank'].between(1,10).all()
    assert not predictions.duplicated(['mask_seed','query_id','method','candidate']).any()
    nodes=np.load(ROOT/'outputs/snapshots/2017.npz')['nodes']
    assert np.isin(predictions.candidate,nodes).all()
    for seed in range(5):
        queries=pd.read_parquet(ROOT/'outputs/splits'/f'queries_2017_{seed}.parquet')
        observed={q.query_id:set(q.observed) for q in queries.itertuples()}
        pp=predictions[predictions.mask_seed==seed]
        assert all(c!=q and c not in observed[q] for q,c in zip(pp.query_id,pp.candidate))
    a3=predictions[predictions.method=='A3'].groupby(['mask_seed','query_id']).fallback.any()
    flags=per[per.method=='A3'].set_index(['mask_seed','query_id']).query_has_applicable_rule
    cross=pd.read_csv(root/'diagnostics/a3_rule_and_fallback.csv')
    for row in cross.itertuples():
        rule=flags.loc[row.mask_seed]; fill=a3.loc[row.mask_seed].reindex(rule.index,fill_value=False)
        assert ((rule==row.query_has_applicable_rule)&(fill==row.a3_fallback_used)).sum()==row.n_queries
    km=pd.read_csv(final/'clustering_metrics.csv'); selected=km[(km.method=='kmeans')&(km.k==20)&(km.d==128)]
    assert set(selected.seed)==set(range(5))
    assert np.isclose(selected.silhouette.mean(),.024570,atol=5e-7,rtol=0)
    assert np.isclose(selected.ari.mean(),.133399,atol=5e-7,rtol=0)
    # Load every retained fitted model, without refitting it.
    model_counts={}
    for stage,count in [('validation',12),('final',30)]:
        models=sorted((root/stage/'models').glob('k*.joblib')); assert len(models)==count
        for model in models:
            estimator=joblib.load(model)
            assert np.isfinite(estimator.cluster_centers_).all()
        assert joblib.load(root/stage/'models/pca.joblib').components_.shape==(128,128)
        model_counts[stage]=len(models)
    cases=pd.read_csv(root/'diagnostics/rank_difference_cases.csv')
    expected_cases=rank_difference_cases(scores)
    assert set(cases.node_id)==set(expected_cases.node_id)
    assert np.array_equal(cases.set_index('node_id').rank_difference.sort_index(),expected_cases.set_index('node_id').rank_difference.sort_index())
    n_png=len(list((root/'figures').rglob('*.png'))); n_svg=len(list((root/'figures').rglob('*.svg')))
    assert n_png==n_svg==15
    return dict(status='passed',verified_files=integrity['hashed_files'],relative_links=integrity['relative_links'],frozen_config_and_splits_preserved=True,
        official_metrics_match=True,pagerank_metric_recalculation=True,apriori_query_aggregate_json_consistent=True,
        prediction_rows_checked=len(predictions),model_files_loaded=model_counts,figures=dict(png=n_png,svg=n_svg),
        source_and_artifact_integrity=True,scope='Read-only delivered-file verification; no model fitting or new data acquisition')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.parse_args()
    print(json.dumps(check_submission(),ensure_ascii=False,indent=2),flush=True)
