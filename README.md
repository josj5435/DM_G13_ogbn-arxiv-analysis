# 시간 제한 인용망에서 논문의 영향력·내용 군집·참고문헌 복원

SCE331 Data Mining 프로젝트. 이 프로젝트는 출판연도를 기준으로 재구성한 과거 인용망과 OGB에서 제공하는 논문 특징을 활용하여, 다음 해의 인용 행동을 얼마나 설명·예측할 수 있는지 분석합니다. arXiv 컴퓨터과학 논문은 빠르게 늘어나(이 데이터에서 2018년 한 해만 29,799편) 연구자가 먼저 읽을 논문, 같은 주제의 논문, 빠뜨린 참고문헌을 스스로 찾기 어렵습니다. 이 질문을 논문의 영향력, 내용 집단, 개별 참고문헌의 세 관점으로 나눠 ogbn-arxiv [1]에서 같은 시간 분할로 검증합니다.

| 연구 질문 | 주요 알고리즘 | 비교 기준 | 활용 맥락 |
| --- | --- | --- | --- |
| 1. 과거 인용망의 전역 중요도가 단순 인용 수보다 다음 해 인용을 잘 예측하는가 | PageRank | In-degree | 먼저 읽을 논문의 순위 |
| 2. 제목·초록 특징으로 만든 내용 군집이 실제 인용 관계와 겹치는가 | k-means | 크기 보존 무작위 배정 | 주제별 문헌 정리 |
| 3. 함께 인용되는 패턴으로 숨긴 참고문헌을 복원할 수 있는가 | Apriori 연관 규칙 | 인기순 추천 | 참고문헌 추천 |

인용망 PageRank는 중요한 논문을 찾는 데 쓰여 왔지만 [20], 오래된 논문에 유리한 시간 편향도 보고돼 [21] 1년 뒤 인용 예측에 맞는지는 따로 확인해야 합니다. 질문 2에서 비교하는 참고문헌 공유는 bibliographic coupling [22], 질문 3의 규칙이 쓰는 함께 인용되는 관계는 co-citation [23]이라는 고전적인 논문 관계 측정에 해당합니다. PCA는 군집 특징의 차원 축소가 정보 손실·품질·비용에 주는 영향을 비교하는 보조 실험입니다. 대괄호 번호는 [References](docs/REFERENCES.md)의 출처입니다.

## 팀

| 팀 번호 | 이름 | 학번 | 역할·기여 |
| --- | --- | --- | --- |
| G13 | 윤지원 | 202220747 | 데이터 수집, AI 활용 코드 구현 관리, 결과 검증 |
| G13 | 조성진 | 202220701 | 보고서 작성 |

GitHub URL: (입력 요망)

비공개 저장소는 수업 가이드의 `leesael` 또는 `sael@ajou.ac.kr` 계정에 collaborator 접근 권한을 설정합니다.

## 구조

```text
src/                  데이터 획득·전처리·3개 알고리즘·평가·그림·재현 실행
configs/              검증·최종 실험 설정
tests/                알고리즘·지표·시간 분리·추천·시각화 테스트
requirements-lock.txt 실제 사용 환경의 고정 패키지 버전
data/README.md        데이터 출처·형식·라이선스·다운로드
outputs/
  arxiv_v2_20261106_review_20261107/
    frozen.json       검증으로 선택한 동결 설정
    validation/       검증 표와 12개 군집 모델
    final/            최종 표·예측·30개 군집 모델·PCA·고정 쌍
    figures/          평가/EDA/인용망 그림 15개, 각각 PNG와 SVG
    visualization/    관계 그림의 노드·간선·좌표·추출 설정
    diagnostics/      순위 사례·참고문헌 길이별 성능·fallback 분석
  splits/             연도별 5개 질의 마스크
  snapshots/          과거 인용 그래프
  tables/             전처리·차수 통계
  manifests/          공식 데이터 출처·원본 해시
docs/                 방법·결과 해석·강의 대응·출처·AI 사용 공개
handoff/              보고서·발표 담당자용 결과 요약
manifest.json         코드·환경·핵심 실험 파일의 무결성 목록
VALIDATION.md         검증 방법·결과·범위
```

원본 데이터는 아래 명령으로 내려받습니다. 가상환경은 설치 과정에서 생성하며, 보고서 PDF와 발표 영상은 별도 제출 자료입니다.

## 환경 설치와 제출 결과 확인

Python **3.12**를 사용합니다. 실제 실험은 Windows 11, Python 3.12.10, CPU, 32GB RAM 환경이었으며 스레드는 4개로 제한했습니다. GPU·PyTorch는 필요하지 않습니다. NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3, scikit-learn 1.8.0, NetworkX 3.4.2, Matplotlib 3.11.2, PyArrow 19.0.1 등의 정확한 버전은 lock 파일에 있습니다.

프로젝트 루트에서 Windows PowerShell:

```powershell
py -3.12 -m venv .venv
$projectPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $projectPython -m pip install -r requirements-lock.txt
& $projectPython -m pytest -q -p no:cacheprovider
& $projectPython -m src.check_submission
```

Linux/macOS에서는 `python3.12 -m venv .venv` 후 `.venv/bin/python`을 사용합니다. 타 운영체제의 전체 재학습 결과는 미검증입니다. `check_submission`은 원본 다운로드 없이 파일 해시·동결 설정·마스크·점수·CSV/JSON/Parquet 일치·저장 모델·그림을 검사합니다. README·일반 문서는 해시 고정 대상에서 제외하며 필수 문서의 존재와 내부 링크를 확인합니다. 팀 정보·GitHub URL을 수정해도 핵심 실험 파일이 같으면 검사를 통과합니다. 단위 테스트와 검증 범위는 [검증 요약](VALIDATION.md)에 있습니다.

## 데이터 획득부터 전체 재현까지

[OGB 공식 ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv)를 사용합니다. 원본은 논문 169,343개, 방향 인용 1,166,243개, 128차원 제목·초록 특징, 출판연도, 40개 분야 라벨입니다. **이 프로젝트는 OGB의 공식 분류 리더보드 실험과 다릅니다.** 직접 공식 ZIP을 내려받아 읽으며 상세 출처·형식은 [데이터 안내](data/README.md)에 있습니다.

전체 실험은 기존 산출물과 충돌하지 않도록 **별도의 작업 디렉터리**에서 실행합니다. 같은 Python 환경을 사용하는 아래 한 명령이 공식 다운로드 → 전처리 → 테스트 → 검증 학습 → 설정 동결 → 최종 학습/평가 → 진단 → 비용 비교 → 그림 → 요약을 순서대로 실행합니다. 작업 디렉터리는 새 폴더 또는 빈 폴더여야 합니다.

```powershell
& $projectPython -m src.reproduce --workspace ../ogbn_arxiv_reproduction
```

다운로드와 학습을 바로 시작하지 않고 파일과 명령만 준비하려면 `--prepare-only`를 추가합니다. 학습 시간·RSS는 장비와 부하에 따라 달라집니다. 전체 실행은 검증 군집 12개, 최종 군집 30개를 포함합니다.

개별 실행이 필요하면 준비된 새 폴더로 이동한 뒤 다음 명령을 순서대로 사용합니다. `python`은 위에서 설치한 Python 환경을 가리켜야 합니다.

```text
python -m src.acquire --root data
python -m src.preprocess
python -m pytest -q -p no:cacheprovider
python -m src.run --stage validation --config configs/validation.yaml
python -m src.run --stage final --config configs/final.yaml
python -m src.diagnostics --run-id arxiv_v2_20261106
python -m src.benchmark --run-id arxiv_v2_20261106
python -m src.make_figures --run-id arxiv_v2_20261106 --mode both
python -m src.summarize --run-id arxiv_v2_20261106
```

수동 실행에서도 OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS를 각각 4로 설정하세요. `src.reproduce`는 이 값을 자동 설정합니다. 새 결과는 재현 폴더의 `outputs/arxiv_v2_20261106/`, 요약은 `handoff/RESULTS_SUMMARY_KO.md`에 생성됩니다. 완료된 실행에 재학습을 덮어쓰지 않으므로 반복하려면 또 다른 새 폴더를 사용합니다.

원본 데이터를 준비한 재현 폴더에서 계산 검사를 수행하려면 `python -m src.verify --run-id arxiv_v2_20261106 --report outputs/checks/model_replay.json`을 실행합니다. 검사 범위는 PageRank·PCA·선택 k-means seed=0·Apriori 규칙과 5개 마스크·고정 쌍·관계 그림입니다. 기존 report 경로가 있으면 새 경로를 지정합니다.

## 알고리즘과 평가

| 구성 | 역할·구현 | 공정한 비교 |
| --- | --- | --- |
| PageRank | 방향 인용망의 희소 power iteration, 균등 teleport·dangling 재분배 | 같은 후보의 In-degree, 다음 1년 내부 인용에 대한 선형 gain NDCG@100·Spearman |
| k-means | L2 정규화 내용 특징에 scikit-learn Lloyd/k-means++ | 같은 k의 원본 128D·PCA 차원, 공통 128D 표본 Silhouette·ARI·무작위 기준 |
| Apriori | 빈발 항목을 먼저 구하고 pair 후보만 세는 2-pass Apriori, confidence 규칙 | 동일 후보·질의·마스크의 인기순(B2), 규칙(A2), 규칙+보충(A3), Recall/NDCG/Precision@10 |
| PCA | 학습 논문으로만 full SVD fit, 16/32/64D 비교 | 설명분산·재구성·거리 분포·군집 품질·실행 비용 |

전처리는 잘못된 끝점·자기 인용·중복·연도 역전을 검사하며, 실제로 연도 역전 20,903개를 제외해 1,145,340개 간선을 사용했습니다. 고립 노드와 빈 학습 basket은 유지합니다. 특징에 NaN이나 무한대가 포함되거나 특징 벡터의 L2 노름이 0인 논문은 군집화 대상에서 제외합니다.

설정 선택은 **≤2016 학습 → 2017 검증**, 최종은 **≤2017 재학습 → 2018 평가**입니다. 최종 라벨·점수로 튜닝하지 않습니다. 후보는 α={0.70,0.85,0.95}, k={20,40,60}, d={16,32,64,128}, Apriori count={100,200,500} 및 사전 자원 조건을 통과한 {20,50}, confidence={0.10,0.20,0.30}입니다. 선택 규칙·모든 seed·지표 정의는 [방법 문서](docs/METHODS.md)와 `configs/validation.yaml`에 있습니다. 선택된 값은 **이 후보 범위의 결과이며 전역 최적이 아닙니다**.

## 주요 결과

| 방법·지표 | 공식 결과 |
| --- | ---: |
| PageRank NDCG@100 | 0.597558 |
| In-degree NDCG@100 | 0.924337 |
| Apriori Recall@10 | 0.189744 |
| Apriori+Fallback Recall@10 | 0.192853 |
| 인기순 Recall@10 | 0.083361 |
| 선택 k-means | k=20, d=128 |
| 평균 Silhouette / ARI | 0.024570 / 0.133399 |

최종 표는 `outputs/arxiv_v2_20261106_review_20261107/final/`, 그림은 같은 Run의 `figures/`에 있습니다. 해석과 그림별 원본 표는 [담당자용 결과 요약](handoff/RESULTS_SUMMARY_KO.md)을 확인하세요.

결과 해석의 핵심은 다음과 같으며, 근거 수치와 출처 파일은 [결과 해석](docs/ANALYSIS.md)에 있습니다.

- PageRank: α를 낮춰 점수 전파를 줄일수록 2017 검증 NDCG@100이 0.316→0.428→0.542로 올랐습니다. 순위 차이가 가장 큰 사례는 참고문헌이 1~4개뿐인 영향력 있는 논문에 한 번 인용된 오래된 논문이었습니다. 다음 해 인용은 누가 인용했는가보다 이미 얼마나 인용됐는가를 따랐습니다.
- k-means: 군집 분리는 약하지만(Silhouette 0.025, 크기 보존 무작위 −0.027) 2018년 논문의 인용 중 23.5%가 같은 군집 안에서 일어나 무작위 기준 9.1%의 약 2.6배입니다.
- Apriori: 인기순보다 Recall@10이 2.3배 높지만, 숨긴 참고문헌의 약 60%는 어떤 규칙의 결과에도 없어 규칙만으로는 복원할 수 없습니다. 참고문헌이 적은 논문일수록 성능이 낮습니다.

## 주요 한계

- PageRank는 이 데이터의 미래 1년 인용 평가에서 In-degree보다 낮았습니다. 다른 데이터까지 일반화하지 않습니다.
- 참고문헌 복원은 적격 논문 15,072/29,799편(50.6%)에 한정됩니다. 순수 Apriori 빈 추천 17.9%도 평가 평균에 포함합니다.
- RuleCoverage 82.1%는 규칙 후보 유무이며 적중률이 아닙니다. B2는 N/A입니다. A3 fallback은 규칙이 있는 질의에도 적용될 수 있습니다.
- Silhouette·ARI가 낮아 군집 품질이 제한적입니다. k-means는 내용 군집화이며 그래프 커뮤니티 탐지가 아닙니다.
- PCA64 설명분산은 약 83.2%로 사전 90% 조건에 미달했습니다. PCA 개선이 입증되지 않아 원본 128D를 선택했습니다. 시간 평균의 큰 관측값으로 속도 배수를 주장하지 않습니다.
- MAG 사전학습 임베딩의 과거 시점 가용성은 보장되지 않습니다. train-only fit을 완전한 미래 정보 차단으로 표현하지 않습니다.
- 선택된 α=0.70, k=20, 최소 count=20, confidence=0.1이 모두 후보 범위의 끝값이고 검증 성능이 한 방향으로 변했습니다. 범위 밖에 더 나은 값이 있을 수 있으나 최종 결과를 본 뒤 범위를 넓히지 않았습니다.

## 출처·AI 사용 공개

OGB/MAG와 논문 저자, 수업 자료, 알고리즘·소프트웨어의 출처는 [References 및 외부 코드](docs/REFERENCES.md)에 정리했습니다. Codex·ChatGPT·Claude의 활용 범위와 검증 방법은 [AI 사용 공개](docs/AI_USE.md)에 명시했습니다. AI 출력 자체를 학술 근거로 사용하지 않습니다.

세 주요 알고리즘의 강의 슬라이드 근거와 실험 설정은 [방법 문서](docs/METHODS.md)에 있습니다.
