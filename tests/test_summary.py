import json
import uuid
import numpy as np
import pandas as pd
import pytest
from src.common import ROOT
from src.summarize import summarize


@pytest.fixture
def run_folder():
    root = ROOT/'outputs/semantic_test_artifacts'/uuid.uuid4().hex
    (root/'final').mkdir(parents=True); (root/'validation').mkdir()
    frozen = dict(selection=dict(alpha=.7, apriori=dict(min_count=20, confidence=.1), clusters=dict(k=20, d=128)),
                  config=dict(cutoff=2016, evaluation_year=2017, clusters=dict(variance_min=.9), queries=dict(min_refs=5, k=10)))
    (root/'frozen.json').write_text(json.dumps(frozen), encoding='utf-8')
    pd.DataFrame([dict(method=m, mask_seed=seed, recall10=r, ndcg10=r/2, empty=.25, rule_coverage=.75,
                       n_queries=8, n_eval_year_papers=10, cutoff=2017, evaluation_year=2018)
                  for m in ['B2', 'A2', 'A3'] for seed, r in enumerate([.2, .4])]).to_csv(root/'final/apriori_metrics.csv', index=False)
    pd.DataFrame([dict(method='kmeans', k=20, d=128, seed=i, silhouette=.03, ari=.14, pipeline_seconds=t)
                  for i, t in enumerate([2., 4., 12.])]).to_csv(root/'final/clustering_metrics.csv', index=False)
    pd.DataFrame([dict(method='A1_PageRank', ndcg100=.6, spearman=.2, n_candidates=10),
                  dict(method='B1_in_degree', ndcg100=.9, spearman=.3, n_candidates=10)]).to_csv(root/'final/pagerank_metrics.csv', index=False)
    pd.DataFrame([dict(d=64, explained_variance=.8)]).to_csv(root/'validation/dimension_diagnostics.csv', index=False)
    pd.DataFrame([dict(k=20, d=128, seed=seed, kind='bibliographic', baseline=b, same_mean=a,
                       different_mean=c, jaccard_difference=a-c)
                  for seed in [0, 1] for b, a, c in [('observed', .01, .003), ('size_preserving_random', .004, .0039)]]).to_csv(root/'final/cluster_citation_patterns.csv', index=False)
    (root/'final/eval_pairs_meta.json').write_text(json.dumps(dict(n_pairs=100)), encoding='utf-8')
    return root


def render(root):
    path = root/'summary.md'; summarize(root, path)
    return path.read_text(encoding='utf-8')


def test_summary_tracks_changed_run_measurements(run_folder):
    before = render(run_folder)
    for text in ['**25.0%**', '**75.0%**', '12.000초', '2.000~4.000초', '0.300000 ± 0.141421', '| 같은 군집 | 0.010000 |', '| 크기 보존 무작위 배정의 차이 | 0.000100 |']:
        assert text in before
    p = run_folder/'final/apriori_metrics.csv'; ap = pd.read_csv(p)
    ap.loc[ap.method == 'A2', ['empty', 'rule_coverage']] = [.4, .6]; ap.to_csv(p, index=False)
    p = run_folder/'final/clustering_metrics.csv'; km = pd.read_csv(p); km.pipeline_seconds = [5., 9., 30.]; km.to_csv(p, index=False)
    after = render(run_folder)
    assert '**40.0%**' in after and '**60.0%**' in after
    assert '30.000초' in after and '5.000~9.000초' in after
    assert '**25.0%**' not in after and '1,297' not in after


def test_missing_measurements_are_na_not_zero_or_reference_values(run_folder):
    p = run_folder/'final/apriori_metrics.csv'; ap = pd.read_csv(p)
    ap = ap.drop(columns=['empty', 'rule_coverage']); ap.to_csv(p, index=False)
    p = run_folder/'final/clustering_metrics.csv'; km = pd.read_csv(p)
    km.pipeline_seconds = np.nan; km.to_csv(p, index=False)
    (run_folder/'validation/dimension_diagnostics.csv').unlink()
    (run_folder/'final/cluster_citation_patterns.csv').unlink()
    text = render(run_folder)
    assert '빈 추천률은 **N/A**' in text and 'RuleCoverage **N/A**' in text
    assert '원본 128차원 실행시간은 N/A (기록 없음)' in text
    assert '| 같은 군집 | N/A |' in text and 'N/A (설명분산 기록 없음)' in text
    assert '17.9%' not in text and '82.1%' not in text and '1,297' not in text


def test_runtime_aggregation_uses_sample_statistics(run_folder):
    text = render(run_folder)
    # [2, 4, 12]: sample SD sqrt(28), median 4, mean 6.
    assert '| 128 | 3 | 6.000 | 5.292 | 4.000 | 2.000 | 12.000 |' in text
