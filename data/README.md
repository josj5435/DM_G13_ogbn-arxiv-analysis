# 데이터 획득·출처·라이선스

- 제공자: Open Graph Benchmark(OGB), Microsoft Academic Graph(MAG), 원 논문 저자·arXiv의 분야 라벨.
- 데이터: [ogbn-arxiv 공식 설명](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv), 배포 version 1. 원본 파일의 크기·SHA-256은 [데이터 명세](../outputs/manifests/data_manifest.json)에 기록했습니다.
- 공식 파일: [arxiv.zip](https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip), 선택 메타데이터 [titleabs.tsv.gz](https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz).
- OGB가 명시한 데이터 라이선스: ODC-BY. OGB/MAG와 원 논문의 출처를 유지합니다. 논문 제목·초록을 프로젝트 코드 라이선스로 재허가한 것으로 간주하지 않습니다. 본문 초록은 제출본에 포함하지 않습니다.

프로젝트 루트의 `python -m src.acquire --root data`가 공식 ZIP을 다운로드·안전 경로 검사 후 해제하고, gzip CSV를 NumPy 배열로 파싱합니다. PyTorch·OGB Python 패키지는 사용하지 않습니다. 선택 텍스트 다운로드 실패는 상태를 기록하며 핵심 그래프 계산과 구분합니다. `python -m src.preprocess`가 정리된 간선·특징 유효 마스크·과거 snapshot·5개 평가 마스크·통계를 생성합니다. README의 새 재현 폴더에서 실행하세요.

| 필드 | 원본 형식·규모 | 용도 |
| --- | --- | --- |
| edge.csv.gz | 1,166,243 × 2, 정수 source/target | 방향 인용 관계 |
| node-feat.csv.gz | 169,343 × 128, float32 | 제목·초록 단어 임베딩 평균 |
| node_year.csv.gz | 169,343개 출판연도 | 시간 분할 |
| node-label.csv.gz | 169,343개, 40개 분야 | 최종 군집 외부 평가 |
| nodeidx2paperid.csv.gz | 원본 node ID→MAG ID | 제목 연결 |

node ID는 arXiv ID가 아닙니다. 인용 방향은 source→target을 유지하고 역간선을 추가하지 않습니다. 정리된 데이터의 노드 ID를 다시 매기지 않습니다. 내부 참고문헌만 관측할 수 있으며 누락·외부 인용을 정답으로 복원할 수 없습니다. 제공 특징의 skip-gram 단어 임베딩은 MAG 코퍼스로 학습됐으므로 과거 시점 정보만 사용했다는 보장은 없습니다.

원본 ZIP, 해제 CSV, original_arrays 및 processed 배열은 용량 때문에 제출본·Git에서 제외합니다. 학습 모델, 평가 마스크, snapshot, 최종 예측·고정 쌍은 `outputs/`에 포함합니다. 원본을 다시 확보하면 나머지 입력을 생성할 수 있습니다. 외부 파일 해시가 기록과 달라지면 데이터 버전을 조사한 뒤 별도 실행으로 취급해야 합니다.
