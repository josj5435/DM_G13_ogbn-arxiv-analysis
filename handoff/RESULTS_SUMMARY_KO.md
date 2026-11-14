# 보고서·발표 담당자용 결과 요약

연구 질문은 (1) 과거 인용망의 PageRank가 다음 해 인용을 예측하는지, (2) 내용 군집과 실제 인용 패턴이 관련되는지, (3) 참고문헌의 연관규칙으로 숨긴 참고문헌을 복원할 수 있는지입니다. 주요 알고리즘은 PageRank·k-means·Apriori이며 PCA는 보조 차원 비교입니다. 서로 다른 과제의 지표를 하나의 정확도로 합산하지 않습니다.

Run: `arxiv_v2_20261106_review_20261107`. 설정 선택은 ≤2016→2017, 최종은 ≤2017→2018입니다. α=0.7, 최소 support count=20, confidence=0.1, k=20, d=128. 사전 후보 범위의 선택 결과이며 전역 최적을 주장하지 않습니다.

## 1. 인용 순위

| method | ndcg100 | spearman | n_candidates |
| --- | --- | --- | --- |
| B1_in_degree | 0.924337 | 0.35585 | 90941 |
| A1_PageRank | 0.597558 | 0.241163 | 90941 |

PageRank의 NDCG@100은 In-degree보다 낮습니다. 미래 인용은 관심도의 대리 지표이며 연구 품질의 정답이 아닙니다. 한 데이터의 결과로 보편적인 우열을 결론내리지 않습니다. 공식 NDCG는 score 내림차순·node ID 오름차순이며, 순위 차이 사례에만 Average Rank를 사용합니다.

## 2. 참고문헌 복원

평가 대상은 2018년 29,799편 중 데이터 내부 과거 참고문헌≥5인 **15,072편(50.6%)**입니다. 나머지 14,727편은 사전 제외 기준에 해당합니다. 아래 수치는 전체 논문·외부 참고문헌·참고문헌이 없는 신규 논문으로 일반화할 수 없습니다.

| 방법 | Recall10 | NDCG10 | RuleCoverage | 빈추천 |
| --- | --- | --- | --- | --- |
| B2 | 0.083361 ± 0.002154 | 0.059101 | N/A | 0.00% |
| A2 | 0.189744 ± 0.001072 | 0.145552 | 82.11% | 17.89% |
| A3 | 0.192853 ± 0.001598 | 0.147028 | 82.11% | 0.00% |

±는 기록된 마스크 seed (0, 1, 2, 3, 4) 사이의 표본 표준편차입니다. 독립 데이터셋 신뢰구간이 아닙니다. 순수 Apriori의 빈 추천률은 **17.9%**이며 빈 목록도 0점으로 평가 평균에 포함합니다. 숨긴 비빈발 정답도 Recall 분모에 남깁니다. RuleCoverage **82.1%**는 규칙 후보가 있는 질의 비율로, 정답 적중률이 아닙니다. B2는 규칙을 사용하지 않아 N/A입니다. A3는 규칙 추천이 10개보다 적으면 인기순으로 보충하므로 규칙이 있는 질의에서도 fallback을 사용할 수 있습니다. [규칙·fallback 구분](../outputs/arxiv_v2_20261106_review_20261107/diagnostics/a3_rule_and_fallback.csv)을 참조하세요.

## 3. 군집과 PCA

선택 모델 k=20, d=128의 Silhouette 평균은 **0.024570**, ARI 평균은 **0.133399**입니다. Silhouette이 0에 가까우면 군집 분리가 약하며 낮은 ARI는 분야 라벨과의 일치도가 제한적임을 뜻합니다. k-means는 내용 기반 군집화이며 그래프 커뮤니티 탐지가 아닙니다.

| d | silhouette | ari |
| --- | --- | --- |
| 16 | 0.021142 | 0.125092 |
| 32 | 0.024094 | 0.132819 |
| 64 | 0.023870 | 0.135768 |
| 128 | 0.024570 | 0.133399 |

PCA64의 검증 누적 설명분산은 83.18%이고 사전 기준 90%의 충족 여부는 **미달**입니다. 선택된 구성은 PCA를 사용하지 않는 원본 특징입니다. 다른 차원은 사전 정의된 민감도 비교이며 최종 점수만으로 PCA의 개선 효과를 판단하지 않습니다.

### 참고문헌 Jaccard와 직접 인용

| 포함확률 역수가중 평균 Jaccard | 값 |
| --- | ---: |
| 같은 군집 | 0.007340 |
| 다른 군집 | 0.003004 |
| 같은 군집 − 다른 군집 | 0.004336 |
| 크기 보존 무작위 배정의 차이 | 0.000074 |

각 쌍은 일반 집합 Jaccard `J=|A∩B|/|A∪B|`이고 그룹 평균에만 `Σ(wJ)/Σw`, `w=1/쌍 포함확률`을 적용합니다. 항목별 가중치를 쓰는 Weighted Jaccard Similarity와 다릅니다. 참고문헌 길이 교차층에서 군집을 보지 않고 고정한 100,000쌍을 공유합니다. 무작위 기준은 평가 노드 전체의 군집 라벨을 섞어 군집 크기를 보존하며, 같은 쌍과 가중치에 적용합니다. 기록된 25개 무작위 배정 결과를 모델 seed별로 먼저 평균한 뒤 5개 seed를 평균했습니다.

Jaccard는 0~1 범위이므로 차이의 절댓값을 함께 해석해야 합니다. 관측된 군집·인용 패턴의 관련성을 검토하는 진단이며 인과관계나 통계적 유의성을 입증하지 않습니다. 같은 군집 직접 인용 비율은 23.46%, 참고문헌 길이층 내 source 군집을 섞는 별도의 무작위 기준은 9.10%입니다.

### 실행 비용

| d | n | mean | std | median | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 5 | 4.888 | 0.417 | 4.936 | 4.417 | 5.398 |
| 32 | 5 | 7.790 | 0.805 | 7.628 | 6.880 | 9.082 |
| 64 | 5 | 12.925 | 0.982 | 12.960 | 11.602 | 14.335 |
| 128 | 5 | 275.769 | 571.117 | 20.750 | 16.165 | 1297.385 |

위 값은 기록된 pipeline wall time(초)이며 n은 유효 측정 수, std는 표본 표준편차입니다. 원본 128차원에서 기록된 실행 5회의 최댓값은 1,297.385초입니다. 최댓값 한 건을 제외한 4회의 범위는 16.165~27.258초입니다. 공유 데스크톱 부하를 통제하지 못했으므로 평균만으로 PCA의 속도 배수를 주장하지 않습니다. 누락된 값은 N/A로 표시하며 0이나 다른 Run의 값으로 대체하지 않습니다.

## 4. 표·그림과 해석 범위

결과는 `outputs/arxiv_v2_20261106_review_20261107/`에 있으며 아래 링크로 확인할 수 있습니다. 모든 그림은 PNG/SVG를 함께 제공합니다.

| 보고서 내용 | 표 | 그림 |
| --- | --- | --- |
| 데이터·정리 통계 | [전처리 통계](../outputs/tables/preprocessing_stats.csv) | [데이터 개요](../outputs/arxiv_v2_20261106_review_20261107/figures/eda/dataset_overview.png) |
| PageRank와 Baseline | [순위 평가](../outputs/arxiv_v2_20261106_review_20261107/final/pagerank_metrics.csv) | [비교 그림](../outputs/arxiv_v2_20261106_review_20261107/figures/metrics/pagerank_vs_indegree.png) |
| Apriori와 보충 | [추천 평가](../outputs/arxiv_v2_20261106_review_20261107/final/apriori_metrics.csv) | [비교 그림](../outputs/arxiv_v2_20261106_review_20261107/figures/metrics/apriori_vs_popularity.png) |
| 선택·설명분산 | [검증 군집](../outputs/arxiv_v2_20261106_review_20261107/validation/clustering_metrics.csv), [차원 진단](../outputs/arxiv_v2_20261106_review_20261107/validation/dimension_diagnostics.csv) | [설명분산·선택](../outputs/arxiv_v2_20261106_review_20261107/figures/metrics/validation_and_variance.png) |
| 군집 품질·무작위 비교 | [군집 평가](../outputs/arxiv_v2_20261106_review_20261107/final/clustering_metrics.csv) | [군집 품질](../outputs/arxiv_v2_20261106_review_20261107/figures/metrics/final_clustering.png), [무작위 기준](../outputs/arxiv_v2_20261106_review_20261107/figures/metrics/clustering_random_baseline.png) |
| 인용 패턴 | [인용 패턴 평가](../outputs/arxiv_v2_20261106_review_20261107/final/cluster_citation_patterns.csv) | [비교 그림](../outputs/arxiv_v2_20261106_review_20261107/figures/metrics/citation_pattern_baselines.png) |
| 관계 예시 | [노드·간선·좌표](../outputs/arxiv_v2_20261106_review_20261107/visualization/) | [인용망 그림 8개](../outputs/arxiv_v2_20261106_review_20261107/figures/networks/) |

원본 MAG 임베딩이 과거 시점에 이용 가능했는지는 보장되지 않습니다. train-only fit만으로 완전한 미래 정보 차단을 보장할 수 없습니다. seed 간 변동은 초기화/마스킹 변동이며 독립 환경에서 전체 실험의 재현성을 보장하지 않습니다. 현재 실행 가능한 검증 범위는 [검증 요약](../VALIDATION.md)에 있습니다.

[방법과 강의 대응](../docs/METHODS.md), [번호형 출처](../docs/REFERENCES.md), [AI 사용 공개](../docs/AI_USE.md)를 함께 참조하세요. 보고서는 8~10쪽·10pt·줄 간격 1.5·IEEE식 번호 인용, 발표는 5분/30MB의 수업 가이드 조건을 따릅니다.
