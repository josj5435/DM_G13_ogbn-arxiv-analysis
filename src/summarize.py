"""Generate a report handoff from saved measurements, without fitting models."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, REFERENCE_RUN, read_json


def number(value, spec='.6f'):
    return format(value, spec) if pd.notna(value) and np.isfinite(value) else 'N/A'


def values(frame, column):
    return pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors='coerce').replace([np.inf, -np.inf], np.nan)


def unique_value(frame, column):
    found = values(frame, column).dropna().unique()
    return float(found[0]) if len(found) == 1 else np.nan


def read_table(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def subset(frame, **filters):
    result = frame
    for column, value in filters.items():
        if column not in result:
            return result.iloc[:0]
        result = result[result[column] == value]
    return result


def table(frame):
    clean = frame.astype(object).where(frame.notna(), 'N/A')
    return '| '+' | '.join(map(str, clean.columns))+' |\n| '+' | '.join(['---']*len(clean.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str, row))+' |' for row in clean.itertuples(index=False, name=None))


def summarize(root, destination):
    root = Path(root); final = root/'final'
    frozen = read_json(root/'frozen.json'); selection = frozen['selection']; cfg = frozen['config']
    pr = read_table(final/'pagerank_metrics.csv'); ap = read_table(final/'apriori_metrics.csv')
    km = read_table(final/'clustering_metrics.csv')
    k, d = selection['clusters']['k'], selection['clusters']['d']
    selected = subset(km, method='kmeans', k=k, d=d)
    dimensions = subset(km, method='kmeans', k=k)
    dim_rows = []; time_rows = []
    for dim in sorted(values(dimensions, 'd').dropna().unique()):
        group = subset(dimensions, d=dim); times = values(group, 'pipeline_seconds').dropna()
        dim_rows.append(dict(d=int(dim), silhouette=number(values(group, 'silhouette').mean()), ari=number(values(group, 'ari').mean())))
        time_rows.append(dict(d=int(dim), n=len(times), mean=number(times.mean(), '.3f'), std=number(times.std(), '.3f'), median=number(times.median(), '.3f'), min=number(times.min(), '.3f'), max=number(times.max(), '.3f')))
    apr = []
    for method in ['B2', 'A2', 'A3']:
        group = subset(ap, method=method); recall = values(group, 'recall10')
        apr.append(dict(방법=method, Recall10=f'{number(recall.mean())} ± {number(recall.std())}', NDCG10=number(values(group, 'ndcg10').mean()), RuleCoverage='N/A' if method == 'B2' else number(values(group, 'rule_coverage').mean(), '.2%'), 빈추천=number(values(group, 'empty').mean(), '.2%')))
    eligible = unique_value(ap, 'n_queries'); total = unique_value(ap, 'n_eval_year_papers')
    inclusion = eligible/total if pd.notna(total) and total > 0 else np.nan
    year = unique_value(ap, 'evaluation_year'); cutoff = unique_value(ap, 'cutoff')
    mask_seeds = values(ap, 'mask_seed').dropna().unique()
    seed_text = ', '.join(str(int(seed)) for seed in sorted(mask_seeds)) or 'N/A'
    a2 = subset(ap, method='A2')
    empty = number(values(a2, 'empty').mean(), '.1%'); coverage = number(values(a2, 'rule_coverage').mean(), '.1%')
    pats = subset(read_table(final/'cluster_citation_patterns.csv'), k=k, d=d)
    observed = subset(pats, kind='bibliographic', baseline='observed')
    random = subset(pats, kind='bibliographic', baseline='size_preserving_random')
    random_by_seed = random.groupby('seed').jaccard_difference.mean() if {'seed', 'jaccard_difference'} <= set(random.columns) else pd.Series(dtype=float)
    jac_same = number(values(observed, 'same_mean').mean()); jac_other = number(values(observed, 'different_mean').mean())
    jac_delta = number(values(observed, 'jaccard_difference').mean()); jac_random = number(random_by_seed.mean())
    direct = number(values(subset(pats, kind='direct_citation', baseline='observed'), 'same_fraction').mean(), '.2%')
    direct_random = number(values(subset(pats, kind='direct_citation', baseline='ref_bin_source_shuffle'), 'same_fraction').mean(), '.2%')
    pair_meta = read_json(final/'eval_pairs_meta.json') if (final/'eval_pairs_meta.json').exists() else {}
    n_pairs = number(pair_meta.get('n_pairs', np.nan), ',.0f')
    diag = read_table(root/'validation/dimension_diagnostics.csv')
    variance = unique_value(subset(diag, d=64), 'explained_variance')
    threshold = cfg['clusters']['variance_min']
    variance_status = 'N/A (설명분산 기록 없음)' if pd.isna(variance) else ('충족' if variance >= threshold else '미달')
    pca_choice = 'PCA를 사용하지 않는 원본 특징' if d == 128 else f'PCA {d}차원 특징'
    original_times = values(subset(dimensions, d=128), 'pipeline_seconds').dropna()
    if original_times.empty:
        timing_text = '원본 128차원 실행시간은 N/A (기록 없음)입니다.'
    else:
        rest = original_times.drop(original_times.idxmax())
        timing_text = f'원본 128차원에서 기록된 실행 {len(original_times)}회의 최댓값은 {number(original_times.max(), ",.3f")}초입니다. 최댓값 한 건을 제외한 {len(rest)}회의 범위는 {number(rest.min(), ".3f")}~{number(rest.max(), ".3f")}초입니다.'
    pr_value = values(subset(pr, method='A1_PageRank'), 'ndcg100').mean()
    baseline_value = values(subset(pr, method='B1_in_degree'), 'ndcg100').mean()
    comparison = 'N/A (순위 지표 기록 없음)' if pd.isna(pr_value) or pd.isna(baseline_value) else ('낮습니다' if pr_value < baseline_value else '높습니다' if pr_value > baseline_value else '같습니다')
    network_count = len(list((root/'figures/networks').glob('*.png')))
    network_label = f'인용망 그림 {network_count}개' if network_count else '인용망 그림 N/A (파일 없음)'
    text = f'''# 보고서·발표 담당자용 결과 요약

연구 질문은 (1) 과거 인용망의 PageRank가 다음 해 인용을 예측하는지, (2) 내용 군집과 실제 인용 패턴이 관련되는지, (3) 참고문헌의 연관규칙으로 숨긴 참고문헌을 복원할 수 있는지입니다. 주요 알고리즘은 PageRank·k-means·Apriori이며 PCA는 보조 차원 비교입니다. 서로 다른 과제의 지표를 하나의 정확도로 합산하지 않습니다.

Run: `{root.name}`. 설정 선택은 ≤{cfg['cutoff']}→{cfg['evaluation_year']}, 최종은 ≤{number(cutoff, '.0f')}→{number(year, '.0f')}입니다. α={selection['alpha']}, 최소 support count={selection['apriori']['min_count']}, confidence={selection['apriori']['confidence']}, k={k}, d={d}. 사전 후보 범위의 선택 결과이며 전역 최적을 주장하지 않습니다.

## 1. 인용 순위

{table(pr.reindex(columns=['method', 'ndcg100', 'spearman', 'n_candidates']).round(6))}

PageRank의 NDCG@100은 In-degree보다 {comparison}. 미래 인용은 관심도의 대리 지표이며 연구 품질의 정답이 아닙니다. 한 데이터의 결과로 보편적인 우열을 결론내리지 않습니다. 공식 NDCG는 score 내림차순·node ID 오름차순이며, 순위 차이 사례에만 Average Rank를 사용합니다.

## 2. 참고문헌 복원

평가 대상은 {number(year, '.0f')}년 {number(total, ',.0f')}편 중 데이터 내부 과거 참고문헌≥{cfg['queries']['min_refs']}인 **{number(eligible, ',.0f')}편({number(inclusion, '.1%')})**입니다. 나머지 {number(total-eligible, ',.0f')}편은 사전 제외 기준에 해당합니다. 아래 수치는 전체 논문·외부 참고문헌·참고문헌이 없는 신규 논문으로 일반화할 수 없습니다.

{table(pd.DataFrame(apr))}

±는 기록된 마스크 seed ({seed_text}) 사이의 표본 표준편차입니다. 독립 데이터셋 신뢰구간이 아닙니다. 순수 Apriori의 빈 추천률은 **{empty}**이며 빈 목록도 0점으로 평가 평균에 포함합니다. 숨긴 비빈발 정답도 Recall 분모에 남깁니다. RuleCoverage **{coverage}**는 규칙 후보가 있는 질의 비율로, 정답 적중률이 아닙니다. B2는 규칙을 사용하지 않아 N/A입니다. A3는 규칙 추천이 {cfg['queries']['k']}개보다 적으면 인기순으로 보충하므로 규칙이 있는 질의에서도 fallback을 사용할 수 있습니다. [규칙·fallback 구분](../outputs/{root.name}/diagnostics/a3_rule_and_fallback.csv)을 참조하세요.

## 3. 군집과 PCA

선택 모델 k={k}, d={d}의 Silhouette 평균은 **{number(values(selected, 'silhouette').mean())}**, ARI 평균은 **{number(values(selected, 'ari').mean())}**입니다. Silhouette이 0에 가까우면 군집 분리가 약하며 낮은 ARI는 분야 라벨과의 일치도가 제한적임을 뜻합니다. k-means는 내용 기반 군집화이며 그래프 커뮤니티 탐지가 아닙니다.

{table(pd.DataFrame(dim_rows, columns=['d', 'silhouette', 'ari']))}

PCA64의 검증 누적 설명분산은 {number(variance, '.2%')}이고 사전 기준 {number(threshold, '.0%')}의 충족 여부는 **{variance_status}**입니다. 선택된 구성은 {pca_choice}입니다. 다른 차원은 사전 정의된 민감도 비교이며 최종 점수만으로 PCA의 개선 효과를 판단하지 않습니다.

### 참고문헌 Jaccard와 직접 인용

| 포함확률 역수가중 평균 Jaccard | 값 |
| --- | ---: |
| 같은 군집 | {jac_same} |
| 다른 군집 | {jac_other} |
| 같은 군집 − 다른 군집 | {jac_delta} |
| 크기 보존 무작위 배정의 차이 | {jac_random} |

각 쌍은 일반 집합 Jaccard `J=|A∩B|/|A∪B|`이고 그룹 평균에만 `Σ(wJ)/Σw`, `w=1/쌍 포함확률`을 적용합니다. 항목별 가중치를 쓰는 Weighted Jaccard Similarity와 다릅니다. 참고문헌 길이 교차층에서 군집을 보지 않고 고정한 {n_pairs}쌍을 공유합니다. 무작위 기준은 평가 노드 전체의 군집 라벨을 섞어 군집 크기를 보존하며, 같은 쌍과 가중치에 적용합니다. 기록된 {len(random)}개 무작위 배정 결과를 모델 seed별로 먼저 평균한 뒤 {len(random_by_seed)}개 seed를 평균했습니다.

Jaccard는 0~1 범위이므로 차이의 절댓값을 함께 해석해야 합니다. 관측된 군집·인용 패턴의 관련성을 검토하는 진단이며 인과관계나 통계적 유의성을 입증하지 않습니다. 같은 군집 직접 인용 비율은 {direct}, 참고문헌 길이층 내 source 군집을 섞는 별도의 무작위 기준은 {direct_random}입니다.

### 실행 비용

{table(pd.DataFrame(time_rows, columns=['d', 'n', 'mean', 'std', 'median', 'min', 'max']))}

위 값은 기록된 pipeline wall time(초)이며 n은 유효 측정 수, std는 표본 표준편차입니다. {timing_text} 공유 데스크톱 부하를 통제하지 못했으므로 평균만으로 PCA의 속도 배수를 주장하지 않습니다. 누락된 값은 N/A로 표시하며 0이나 다른 Run의 값으로 대체하지 않습니다.

## 4. 표·그림과 해석 범위

결과는 `outputs/{root.name}/`에 있으며 아래 링크로 확인할 수 있습니다. 모든 그림은 PNG/SVG를 함께 제공합니다.

| 보고서 내용 | 표 | 그림 |
| --- | --- | --- |
| 데이터·정리 통계 | [전처리 통계](../outputs/tables/preprocessing_stats.csv) | [데이터 개요](../outputs/{root.name}/figures/eda/dataset_overview.png) |
| PageRank와 Baseline | [순위 평가](../outputs/{root.name}/final/pagerank_metrics.csv) | [비교 그림](../outputs/{root.name}/figures/metrics/pagerank_vs_indegree.png) |
| Apriori와 보충 | [추천 평가](../outputs/{root.name}/final/apriori_metrics.csv) | [비교 그림](../outputs/{root.name}/figures/metrics/apriori_vs_popularity.png) |
| 선택·설명분산 | [검증 군집](../outputs/{root.name}/validation/clustering_metrics.csv), [차원 진단](../outputs/{root.name}/validation/dimension_diagnostics.csv) | [설명분산·선택](../outputs/{root.name}/figures/metrics/validation_and_variance.png) |
| 군집 품질·무작위 비교 | [군집 평가](../outputs/{root.name}/final/clustering_metrics.csv) | [군집 품질](../outputs/{root.name}/figures/metrics/final_clustering.png), [무작위 기준](../outputs/{root.name}/figures/metrics/clustering_random_baseline.png) |
| 인용 패턴 | [인용 패턴 평가](../outputs/{root.name}/final/cluster_citation_patterns.csv) | [비교 그림](../outputs/{root.name}/figures/metrics/citation_pattern_baselines.png) |
| 관계 예시 | [노드·간선·좌표](../outputs/{root.name}/visualization/) | [{network_label}](../outputs/{root.name}/figures/networks/) |

원본 MAG 임베딩이 과거 시점에 이용 가능했는지는 보장되지 않습니다. train-only fit만으로 완전한 미래 정보 차단을 보장할 수 없습니다. seed 간 변동은 초기화/마스킹 변동이며 독립 환경에서 전체 실험의 재현성을 보장하지 않습니다. 현재 실행 가능한 검증 범위는 [검증 요약](../VALIDATION.md)에 있습니다.

[방법과 강의 대응](../docs/METHODS.md), [번호형 출처](../docs/REFERENCES.md), [AI 사용 공개](../docs/AI_USE.md)를 함께 참조하세요. 보고서는 8~10쪽·10pt·줄 간격 1.5·IEEE식 번호 인용, 발표는 5분/30MB의 수업 가이드 조건을 따릅니다.
'''
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding='utf-8')
    return destination


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run-id', default=REFERENCE_RUN)
    p.add_argument('--output', type=Path, default=ROOT/'handoff/RESULTS_SUMMARY_KO.md'); a = p.parse_args()
    summarize(ROOT/'outputs'/a.run_id, a.output)
    print('Result summary written.')
