from pathlib import Path
import csv, json, math, re, statistics, html, os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/arxiv_v2_20261106_review_20261107'
OUT=ROOT/'output/pdf'; OUT.mkdir(parents=True,exist_ok=True)
QA=ROOT/'tmp/report_qa'; QA.mkdir(parents=True,exist_ok=True)
for name,file in [('KR','malgun.ttf'),('KRB','malgunbd.ttf'),('EN','times.ttf'),('ENB','timesbd.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
W,H=A4
WIDTH=W-36*mm
styles={
 'body':ParagraphStyle('body',fontName='KR',fontSize=10,leading=15,spaceAfter=5,wordWrap='CJK',textColor=colors.black),
 'head':ParagraphStyle('head',fontName='KRB',fontSize=10,leading=15,spaceBefore=6,spaceAfter=6,wordWrap='CJK',keepWithNext=True),
 'title':ParagraphStyle('title',fontName='KRB',fontSize=12,leading=18,spaceAfter=8,wordWrap='CJK',alignment=TA_CENTER),
 'caption':ParagraphStyle('caption',fontName='KR',fontSize=10,leading=15,spaceAfter=5,wordWrap='CJK'),
 'cell':ParagraphStyle('cell',fontName='KR',fontSize=10,leading=15,wordWrap='CJK'),
 'ref':ParagraphStyle('ref',fontName='KR',fontSize=10,leading=15,spaceAfter=4,wordWrap='CJK',leftIndent=17,firstLineIndent=-17),
}
def formatted(s,bold=False):
    escaped=html.escape(str(s))
    en='ENB' if bold else 'EN'
    out=re.sub(r'[\x20-\x7e\u00c0-\u024f\u1d40\u1d62\u1d65\u2078\u2079\u207b\u2212]+',lambda m:f'<font name="{en}">{m.group(0)}</font>',escaped)
    return out.replace('\n','<br/>')
def para(s,kind='body'):
    return Paragraph(formatted(s,kind in ('head','title')),styles[kind])
def readcsv(rel):
    with (RUN/rel).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def mean(a,k):return statistics.mean(float(r[k]) for r in a)
def sd(a,k):return statistics.stdev(float(r[k]) for r in a)
def ms(a,k,d=6):return f'{mean(a,k):.{d}f} ± {sd(a,k):.{d}f}'
pages=[]; current=[]
def p(s):current.append(('p',s))
def h(s):current.append(('h',s))
def table(caption,header,rows,widths):current.append(('t',caption,header,rows,widths))
def fig(rel,caption,width):current.append(('f',rel,caption,width))
def end():
    global current
    pages.append(current);current=[]

# Page 1
current.append(('title','시간 제한 인용망에서 논문의 영향력과 내용 군집 및 참고문헌 복원'))
p('SCE331 Data Mining  |  G13  |  아주대학교 소프트웨어학과')
table('', ['이름','학번','역할과 기여'], [
 ['윤지원','202220747','데이터 수집, AI 활용 코드 구현 관리, 결과 검증'],
 ['조성진','202220701','보고서 작성']], [70,85,WIDTH-155])
p('GitHub URL  [입력 필요]    제출 버전 전체 commit hash  [입력 필요]')
h('Abstract')
p('본 연구는 ogbn-arxiv의 논문 169,343편과 방향 인용 1,166,243건을 이용해 과거 인용망의 중요도, 내용 군집, 공동 인용 패턴이 다음 해의 인용 행동과 어떤 관련을 갖는지 분석하였다. PageRank, k-means, Apriori를 각각 미래 인용 순위, 내용 기반 군집화, 숨긴 참고문헌 복원에 적용하였다. 2016년까지의 데이터로 학습하고 2017년으로 설정을 선택한 뒤, 2017년까지 재학습하여 2018년을 최종 평가하였다. PCA는 차원 축소의 보조 실험으로 사용하였다.')
p('최종 PageRank의 NDCG@100은 0.597558로 In-degree의 0.924337보다 낮았다. 선택한 k=20, 원본 128차원 k-means의 평균 Silhouette은 0.024570으로 분리가 약했지만, 동일 군집 내 직접 인용 비율은 23.46%로 무작위 기준 9.10%보다 높았다. Apriori의 Recall@10은 0.189744로 인기순 0.083361의 약 2.28배였다. 다만 참고문헌 복원 평가는 2018년 논문의 50.6%에만 적용되며, 순수 Apriori는 적격 질의의 17.89%에서 빈 추천을 반환하였다. 세 결과는 과제별 효용과 한계가 달라 하나의 정확도로 통합할 수 없다.')
h('1 Introduction')
p('문헌을 탐색하는 연구자는 먼저 읽을 논문, 같은 주제의 논문 집단, 빠뜨린 참고문헌을 찾아야 한다. 본 연구는 공개 인용망과 제목·초록 특징으로 이 세 요구를 분석한다 [1]. 연구 질문은 과거 PageRank가 단순 인용 수보다 다음 해 인용 순위를 잘 설명하는지, 내용 군집이 실제 인용 관계와 겹치는지, 공동 인용 규칙이 일부 가려진 참고문헌을 복원하는지이다.')
p('PageRank는 연결을 통한 중요도 전파를 사용하며 [2], 과학 인용망에도 적용되었다 [3]. 그러나 논문의 연령에 따른 편향이 보고되어 [4] 미래 1년 인용 예측에서의 성능을 별도로 확인해야 한다. 참고문헌 공유는 bibliographic coupling [5], 여러 논문이 함께 인용되는 관계는 co-citation [6]에 해당한다. 본 연구는 이 관계들을 구분하고 시간 분할과 단순 기준 방법으로 각 알고리즘의 기여를 평가한다. 연구 범위는 데이터셋 내부 인용과 제공 특징으로 제한한다.')
end()

# Page 2
h('2 Data acquisition and preprocessing')
h('2.1 데이터 출처와 획득')
p('데이터 제공자는 Open Graph Benchmark와 Microsoft Academic Graph이며, ogbn-arxiv 배포 버전 1을 사용하였다 [1]. 데이터는 arXiv 컴퓨터과학 논문으로 구성된다. src/acquire.py가 OGB 공식 arxiv.zip을 직접 다운로드하고 압축 경로의 안전성을 검사한 뒤 gzip CSV를 NumPy 배열로 읽는다. OGB Python 패키지나 PyTorch 로더는 사용하지 않았다. 선택 메타데이터 titleabs.tsv.gz는 node ID에서 MAG ID를 거쳐 제목을 연결하는 데 사용하였다. 본 연구에서 임베딩을 새로 학습하지는 않았다.')
p('원본 URL은 https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip 이며, 선택 텍스트 URL은 https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz 이다. 원본 ZIP은 83,058,288 bytes이고 SHA-256은 다음과 같다. 버전 식별은 접근일 대신 배포 버전과 원본 파일별 해시를 사용한다.')
p('49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276')
table('표 1 데이터 형식과 변수', ['필드','규모와 형식','분석 용도'],[
 ['edge','1,166,243 × 2 정수','인용하는 논문에서 인용받는 논문으로의 방향 간선'],
 ['node features','169,343 × 128 float32','제목·초록 단어 임베딩의 평균으로 제공된 특징'],
 ['publication year','169,343 정수','과거 snapshot 구성과 평가 연도 분리'],
 ['field label','40개 분야','최종 군집 외부 평가에만 사용'],
 ['node ID와 MAG ID','논문별 대응 관계','원본 ID 유지와 제목 연결']], [95,135,WIDTH-230])
h('2.2 정리 정책과 기술 통계')
p('src/preprocess.py는 끝점 범위, 자기 인용, 중복, 연도 결측·역전을 검사하였다. 유효 연도는 1800~2026년의 정수로 정하고 동일 연도 인용은 유지하였다. 미래에 출판된 논문을 인용하는 연도 역전 간선 20,903건을 제거하여 1,145,340건을 남겼다. 실제 잘못된 끝점, 자기 인용, 중복 간선, 무효 연도·특징·라벨은 각각 0건이었다. 특징은 모든 성분이 유한하고 L2 norm이 양수일 때 군집화에 사용하였다.')
table('표 2 과거 인용망의 구조', ['출판연도 cutoff','노드','간선','고립 노드','Dangling 노드'],[
 ['2016','69,499','226,954','3,909','13,750'],
 ['2017','90,941','361,979','4,001','15,916']], [112,85,100,95,WIDTH-392])
p('2017 snapshot의 평균 In-degree와 Out-degree는 3.9804, 중앙값은 각각 1과 2, 최댓값은 3,015와 221이었다. 고립 노드와 빈 참고문헌 basket을 제거하지 않아 순위와 support의 모집단을 유지하였다. 원본 node ID는 배열 행 번호이며 arXiv ID가 아니다. 직접 인용의 역간선을 추가하지 않았고 ID를 다시 매기지 않았다.')
p('라이선스는 OGB가 명시한 ODC-BY이다 [1]. 외부 논문 인용과 수집에서 누락된 관계는 관측되지 않으며, 연도 단위 정보로 동일 연도 내 선후 관계를 복원할 수 없다. MAG 코퍼스로 학습된 제공 임베딩은 역사적 학습 시점이 보장되지 않는다. 따라서 시간 분할은 그래프·모델 적합의 미래 정보 사용을 줄이지만 완전한 과거 시점 재현을 보장하지 않는다.')
end()

# Page 3
h('3 Methods')
p('전체 파이프라인은 공식 데이터 획득, 정리, 과거 snapshot·평가 마스크 생성, 2017년 검증, 설정 동결, 2018년 최종 평가, 진단·그림 생성 순서이다. 공통 데이터에서 PageRank는 구조적 중요도, k-means는 내용 유사성, Apriori는 공동 인용을 다룬다. 각 방법은 목적에 맞는 지표로 평가하며 서로 다른 과제의 점수를 합산하지 않는다.')
table('표 3 주요 알고리즘과 수업 범위의 대응', ['주요 알고리즘','강의자료와 페이지','프로젝트 구현'],[
 ['PageRank','DM_11 Link Analysis 11, 13, 18, 27, 29, 31쪽','src/pagerank.py'],
 ['k-means','DM_05 Clustering 34~40쪽','src/clusters.py'],
 ['Apriori','DM_03 Association Rules 34~38쪽, 40~42쪽 보완','src/apriori.py']], [90,WIDTH-225,135])
p('강의 대응은 저장소 docs/METHODS.md에 기록된 페이지에 따른다 [7]. 주요 알고리즘은 각각 5장 이상의 수업 내용에 대응한다. PCA, 표본추출, 초기화·hyperparameter 변경은 별도의 주요 알고리즘으로 세지 않았다.')
h('3.1 PageRank')
p('과거 노드 집합 V와 인용 간선 E에서 논문 u가 v를 인용하면 u→v로 둔다. 전이행렬 P는 u의 중요도를 Out-degree에 따라 참고문헌에 나누어 전달한다. 초기 확률은 1/N이며, Out-degree가 0인 dangling 노드의 확률 질량은 모든 노드에 균등 배분한다 [2]. 업데이트는 다음과 같다.')
p('p_next = α(Pᵀp + dangling_mass × 1/N) + (1 − α) × 1/N')
p('SciPy 희소행렬로 직접 power iteration을 구현하였다. 연속 반복의 L1 차이가 10⁻⁸ 미만이면 종료하고 최대 1,000회에서 미수렴하면 실패 처리한다. 최종 고정점 residual, 비음수·유한성, 점수 합 1을 검사한다. 후보 α는 0.70, 0.85, 0.95이다. 비교 방법은 같은 과거 간선의 In-degree로, 인용한 논문의 중요도를 구분하지 않는 기준이다.')
h('3.2 k-means와 보조 PCA')
p('각 논문의 128차원 특징을 float64로 변환하고 논문별 L2 정규화한다. k-means는 군집 내 제곱거리 합을 최소화하도록 가장 가까운 중심으로 배정하고 평균 중심을 갱신한다. 원형에 가까운 집단과 유클리드 거리를 전제로 하며 초기화와 k에 민감하다. scikit-learn 1.8.0 KMeans의 k-means++ 초기화와 Lloyd 반복을 사용하였다. n_init=10, max_iter=300, tol=10⁻⁴이다 [8].')
p('보조 PCA는 학습 논문에만 full SVD로 적합하고 평가 논문에는 transform만 적용한다. whiten=False이며 PCA 후 추가 L2 정규화는 하지 않는다. d=16, 32, 64와 PCA를 적용하지 않는 원본 d=128을 비교한다. 평가는 공통 원본 정규화 128차원 공간에서 수행하여 차원별 Silhouette을 비교한다. 내용 특징으로 만든 군집을 이후 실제 인용과 결합하며, 이를 그래프 community detection으로 해석하지 않는다.')
end()

# Page 4
h('3.3 Apriori 연관 규칙과 참고문헌 추천')
p('과거 논문 한 편을 transaction으로, 그 논문이 인용한 내부 과거 논문의 중복 없는 집합을 basket으로 둔다. 모든 학습 논문을 분모 M에 포함하므로 빈 basket도 support 계산에 남는다. 두 번의 집계로 빈발 단일 항목 L1과 빈발 쌍 L2를 찾는 pair-only Apriori를 직접 구현하였다. 빈발 집합의 부분집합도 빈발하다는 성질로 L1에 없는 항목을 쌍 집계에서 제외한다 [9]. 길이 3 이상 itemset이나 복합 antecedent는 학습하지 않았다.')
p('쌍 {a,b}에서 양방향 규칙 a⇒b와 b⇒a를 만들며, support=count(a,b)/M, confidence(a⇒b)=count(a,b)/count(a), lift=confidence/(count(b)/M)이다. 최소 count와 confidence를 통과한 규칙만 사용한다. 질의의 관측 참고문헌에 a가 있으면 b를 후보로 내고, 여러 규칙이 같은 b를 제안하면 최대 confidence를 점수로 사용한다. 동률은 학습 빈도 내림차순, node ID 오름차순으로 해소한다.')
p('B2는 전체 후보의 학습 인용 빈도 순서로 추천한다. A2는 규칙 후보만 최대 10개 추천한다. A3는 A2의 순서를 유지하고 빈자리를 B2로 보충한다. 세 방법 모두 관측 참고문헌과 질의 자신을 제외하며, 과거 V 안에서만 추천한다. A3의 보충은 규칙이 없는 질의뿐 아니라 규칙 추천 수가 10개보다 적은 질의에도 적용된다. lift는 규칙 설명용이며 추천 순위 점수로 쓰지 않는다.')
h('4 Experimental design and evaluation')
h('4.1 시간 분할과 질의 구성')
table('표 4 학습과 평가 모집단', ['단계','학습','평가 연도 논문','복원 적격 질의'],[
 ['Validation','출판연도 ≤2016','2017년 21,442편','8,712편'],
 ['Final','출판연도 ≤2017','2018년 29,799편','15,072편']], [85,135,140,WIDTH-360])
p('이는 OGB의 공식 분류 리더보드 split과 다르다. 2017년으로 설정을 선택해 frozen.json과 해시로 동결한 뒤 2018년에는 그 설정만 사용하였다. 최종 분야 라벨은 ARI 진단에만 쓰고 모델 선택에는 쓰지 않았다. PageRank와 인기순·Apriori 학습에는 미래 source의 간선을 넣지 않았다.')
p('복원 질의는 평가 연도 논문 중 내부 과거 참고문헌이 5개 이상인 논문이다. 참고문헌 B에서 max(1, floor(0.2|B|))개를 숨긴 집합 H로 정하고 나머지를 관측 집합 O로 제공한다. 동일한 질의·후보·마스크를 세 방법에 공유한다. 마스크 seed는 0~4이며 O와 H는 겹치지 않는다. 2018년 14,727편은 이 조건을 충족하지 않아 제외되므로 복원 결과는 전체 신규 논문의 성능이 아니다.')
h('4.2 지표의 정의와 공정한 비교')
p('영향력 순위의 정답은 평가 연도 논문에서 과거 후보 v로 들어온 인용 수 cᵥ이다. relevance rᵥ=log₂(1+cᵥ)로 정하고 DCG@100=Σᵢ rᵢ/log₂(i+1)의 선형 gain을 사용한다. IDCG는 동일한 전체 후보의 relevance를 정렬해 계산한다 [10]. 점수 동률은 node ID 오름차순이며 Spearman 상관은 점수와 미래 인용 수를 비교한다. 미래 인용은 관심도의 대리 지표로 해석한다.')
end()

# Page 5
h('4.2 지표의 정의와 공정한 비교 계속')
p('군집의 Silhouette은 점별 군집 내부 평균거리 a와 가장 가까운 다른 군집까지의 평균거리 b를 이용한 (b−a)/max(a,b)의 평균이다 [8]. 매 설정에서 같은 평가 논문 2,000편(seed=101)을 원본 정규화 128차원 공간에서 평가한다. ARI는 40개 분야 라벨과의 일치도를 우연 수준에 대해 보정한 값이다 [8]. 군집 크기를 보존한 무작위 배정 5회를 기준으로 쓰며, 모델 seed 사이의 ARI를 안정성 진단으로 제시한다.')
p('군집별 참고문헌 공유는 평가 논문의 참고문헌 수 교차층에서 군집을 보지 않고 고정한 100,000쌍(seed=202)에 대해 Jaccard=|A∩B|/|A∪B|를 계산한다. 평균은 쌍의 포함확률 역수를 가중치로 사용하는 ΣwJ/Σw이다. 이는 항목별 가중 Jaccard와 다르다. 같은 군집과 다른 군집의 차이를 크기 보존 무작위 배정과 비교한다. 직접 인용은 참고문헌 길이층 안에서 source 군집을 섞은 기준과 비교한다.')
p('복원 Recall@10은 적중 수/전체 숨긴 참고문헌 수이며 비빈발 정답도 분모에 남긴다. Precision@10은 짧은 추천도 분모를 10으로 둔다. NDCG@10은 이진 적중과 로그 할인으로 계산한다. 빈 추천은 0점으로 포함한다. RuleCoverage는 규칙 후보가 있는 적격 질의의 비율로 적중률이 아니며 B2에는 적용하지 않는다. Catalog coverage는 전체 후보 중 한 번이라도 추천된 항목 비율이다.')
h('4.3 검증 기반 설정 선택')
table('표 5 후보와 동결된 설정', ['방법','검증 후보와 선택 기준','선택'],[
 ['PageRank','α={0.70,0.85,0.95}; NDCG@100 최대','α=0.70'],
 ['k-means와 PCA','k={20,40,60}, d={16,32,64,128}; 아래 규칙','k=20, d=128'],
 ['Apriori','count={20,50,100,200,500}, confidence={0.1,0.2,0.3}; 5마스크 평균 Recall@10 최대','count=20\nconfidence=0.1']], [95,WIDTH-205,110])
p('PageRank 동률은 0.85와의 거리, 작은 α 순서로 정한다. PCA 후보는 학습 누적 설명분산 90% 이상이어야 하며, 최고 검증 Silhouette에서 0.01 이내인 적격 후보 중 작은 d, 작은 k를 선택한다. Apriori 동률은 NDCG, 큰 count, 큰 confidence 순서이다. count 20·50의 확장은 평가 점수를 보기 전 학습 집계가 1,800초·8GiB RSS 예산에 들어오는지로 결정하였다. 최종 군집은 선택 k의 모든 d와 선택 d의 모든 k를 합친 6설정에 seed 0~4를 적용한 30모델이다.')
h('4.4 계산 환경과 재현성')
p('저장된 실험 환경은 Windows 11, Python 3.12.10, Intel64 Family 6 Model 170 CPU, 물리 16코어·논리 22코어, RAM 약 31.5GiB이며 연산 thread를 4개로 제한하였다. NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3, scikit-learn 1.8.0, NetworkX 3.4.2, Matplotlib 3.11.2, PyArrow 19.0.1을 사용했다 [8], [11], [12]. wall time과 전체 프로세스의 표본 최대 RSS를 기록했다. RSS는 알고리즘만의 추가 메모리가 아니다.')
p('requirements-lock.txt와 README의 명령으로 환경을 만들고 python -m src.reproduce --workspace ../ogbn_arxiv_reproduction 을 실행하면 획득부터 표·그림까지 재현하도록 구성했다. 저장소 VALIDATION.md에는 35개 테스트와 제출 검사 통과가 기록되어 있다. 검증은 작은 예제 손계산, 독립 PageRank, Apriori 완전열거, 지표·누수·예측 파일 검사에 기반한다. 다른 운영체제에서 전체 재학습의 일치는 확인되지 않았다.')
end()

# Page 6
h('5 Results and discussion')
h('5.1 PageRank와 미래 인용 순위')
p('검토된 실험 결과 묶음은 arxiv_v2_20261106_review_20261107이다. 최종 후보는 2017년까지의 논문 90,941편이다. 표 6은 같은 후보와 정답으로 계산한 결과이며, PageRank는 검증에서 선택한 α=0.70을 사용하였다.')
pr=readcsv('final/pagerank_metrics.csv')
table('표 6 2018년 미래 인용 순위 평가', ['방법','NDCG@100','Spearman','계산 시간 초'],[
 ['In-degree',f'{float(pr[0]["ndcg100"]):.6f}',f'{float(pr[0]["spearman"]):.6f}',f'{float(pr[0]["seconds"]):.6f}'],
 ['PageRank',f'{float(pr[1]["ndcg100"]):.6f}',f'{float(pr[1]["spearman"]):.6f}',f'{float(pr[1]["seconds"]):.6f}']], [125,120,120,WIDTH-365])
fig('figures/metrics/pagerank_vs_indegree.png','그림 1 동일한 과거 후보에서의 NDCG@100 비교. 원자료는 final/pagerank_metrics.csv이다.',330)
p('PageRank는 NDCG@100과 전체 순위 상관에서 모두 In-degree보다 낮았다. 40회 반복으로 수렴했으며 고정점 L1 residual은 5.18×10⁻⁹, 점수 합은 1이었다. 따라서 낮은 예측 성능을 미수렴으로 설명할 수 없다. 두 방법의 상위 100편 중 공통 논문은 40편이다. PageRank 상위 집합의 다음 해 인용 수 중앙값은 17, In-degree는 204.5였다. 다음 해 인용이 0인 논문도 각각 9편과 0편으로 차이를 보였다.')
table('표 7 검증 연도에서 α에 따른 민감도', ['α','0.70','0.85','0.95'],[
 ['NDCG@100','0.542','0.428','0.316'],
 ['Spearman','0.259','0.256','0.255'],
 ['반복 수','40','86','259']], [125,120,120,WIDTH-365])
p('α가 작을수록 상위 순위 성능은 좋아졌지만 Spearman 변화는 작아 상위권 배치가 주요 차이였음을 시사한다. 순위 차이가 큰 사례 중 2006년 논문 node 103323은 과거 인용이 한 번인데 PageRank 147위였다. 이를 인용한 node 154444는 PageRank 34위이면서 참고문헌이 2개뿐이었다. 높은 중요도를 적은 참고문헌에 나누는 구조가 저인용 논문을 상위로 올린 사례이다.')
p('이 관측은 점수 전파와 단기 인용 정답의 불일치를 보여주지만 전체 성능 차이의 원인을 확정하지는 않는다. 과학적 중요성 탐색 [3]과 다음 해 인용 예측은 다른 목표이다. 시간 편향을 보정한 지표 [4]와의 비교는 후속 과제로 남기며, PageRank가 일반적으로 열등하다고 결론내리지 않는다. 저장된 diagnostics/rank_difference_cases.csv와 top100_overlap.json이 사례 근거이다.')
end()

# Page 7
h('5.2 내용 군집과 인용 패턴')
km=readcsv('final/clustering_metrics.csv'); stability=readcsv('final/cluster_stability.csv')
rr=[]
for d in [16,32,64,128]:
    a=[r for r in km if r['method']=='kmeans' and r['k']=='20' and r['d']==str(d)]
    b=[r for r in stability if r['k']=='20' and r['d']==str(d)]
    rr.append([str(d)+(' 선택' if d==128 else ''),ms(a,'silhouette'),f'{mean(a,"ari"):.6f}',f'{mean(b,"ari"):.4f}'])
random=[r for r in km if r['method']=='B4_random' and r['k']=='20' and r['d']=='128']
rr.append(['무작위 배정',f'{mean(random,"silhouette"):.6f}',f'{mean(random,"ari"):.6f}','해당 없음'])
table('표 8 k=20의 최종 군집 품질과 안정성', ['차원 d','Silhouette 평균 ± SD','ARI 평균','seed 간 ARI'],rr,[90,180,100,WIDTH-370])
p('실제 모델의 평균은 초기화 seed 5개의 결과이며, 무작위 기준은 각 모델의 군집 크기를 유지한 5회 배정으로 총 25개 결과이다. 선택 모델의 Silhouette은 0.024570으로 군집 간 분리가 약하다. 그러나 무작위 배정은 −0.027126이고 ARI도 0.000380에 가까워, 내용 특징의 군집 신호가 전혀 없는 결과와는 다르다. k=40·60의 Silhouette은 각각 약 0.0187·0.0131로 낮아졌다.')
fig('figures/metrics/citation_pattern_baselines.png','그림 2 참고문헌 공유 차이와 동일 군집 직접 인용의 무작위 기준 비교. 왼쪽은 포함확률 역수가중 평균 Jaccard의 차이이다.',WIDTH)
p('동일 군집 쌍의 평균 Jaccard는 0.007340, 다른 군집은 0.003004로 차이는 0.004336이었다. 크기 보존 무작위 기준의 차이는 0.000074였다. 참고문헌을 하나 이상 공유한 비율은 각각 12.36%와 5.20%였다. 절대 Jaccard가 작다는 점도 함께 해석해야 한다. 2018년에서 과거로의 직접 인용 215,544건 중 동일 군집 인용은 23.46%로, 길이층 내 source 군집을 섞은 9.10%의 약 2.58배였다. 군집에 넣지 않은 인용 관계와 내용 집단의 관련을 보여주지만 인과나 통계적 유의성을 입증한 것은 아니다.')
p('PCA64의 검증 학습 설명분산은 83.18%로 사전 기준 90%를 넘지 못해 최종 선택은 PCA 없는 128차원이었다. 최종 학습에서 PCA16·32·64의 설명분산은 49.84%·65.43%·83.15%였다. PCA 모델의 seed 간 ARI는 약 0.80~0.82로 원본 0.72보다 높았지만, 품질 개선은 확인되지 않았다. 128차원이라는 사실만으로 차원의 저주가 성능 저하를 일으켰다고 단정할 수 없다.')
p('k=20의 pipeline 시간 중앙값은 d=16·32·64·128에서 4.94·7.63·12.96·20.75초였다. 원본 128차원에는 1,297.38초의 큰 관측이 있어 평균 275.77초, 표본 SD 571.12초였다. 이 값을 제외하지 않았으며, PCA 비용을 포함한 관측 시간만으로 통제된 속도 향상을 주장하지 않는다.')
end()

# Page 8
h('5.3 참고문헌 복원의 성능과 실패 범위')
p('표 9는 동일한 적격 질의 15,072편에 대한 마스크 seed 5개의 평균이다. ±는 마스크 간 표본 표준편차이며 독립 데이터셋의 신뢰구간이 아니다. A2는 순수 Apriori, A3는 인기순 보충을 결합한 방법이다.')
ap=readcsv('final/apriori_metrics.csv');rr=[]
for method,label in [('B2','인기순 B2'),('A2','Apriori A2'),('A3','보충 A3')]:
    a=[r for r in ap if r['method']==method]
    rr.append([label,ms(a,'recall10'),f'{mean(a,"ndcg10"):.6f}',f'{mean(a,"precision10"):.6f}','해당 없음' if method=='B2' else f'{100*mean(a,"rule_coverage"):.2f}%',f'{100*mean(a,"empty"):.2f}%'])
table('표 9 2018년 적격 질의의 복원 결과', ['방법','Recall@10 평균 ± SD','NDCG@10','P@10','RuleCoverage','빈 추천'],rr,[75,150,72,62,78,WIDTH-437])
fig('figures/metrics/apriori_vs_popularity.png','그림 3 복원 Recall과 규칙 적용 비율. Error bar는 5개 마스크의 표준편차이며 RuleCoverage는 적중률과 구별한다.',WIDTH)
p('A2의 Recall@10은 인기순의 약 2.28배, 절대 차이는 0.106383이었다. A3는 A2보다 0.003109 높았고 빈 목록을 없앴다. 그러나 규칙 존재 비율 82.11%가 정답 복원 비율을 뜻하지는 않는다. 순수 Apriori의 빈 추천 17.89%도 전체 평균에 0점으로 포함했다. 실제 추천된 catalog 비율은 A2·A3 약 1.05%, B2 약 0.024%였다.')
table('표 10 참고문헌 수별 A2 진단 마스크 seed 0', ['내부 과거 참고문헌','질의 수','Recall@10','규칙 적용','빈 추천'],[
 ['5~9개','6,948','0.1570','68.45%','31.55%'],
 ['10~19개','5,808','0.2185','92.17%','7.83%'],
 ['20개 이상','2,316','0.2225','98.62%','1.38%']], [145,80,95,90,WIDTH-410])
p('최종 학습은 빈 basket 15,916개를 포함한 M=90,941개에서 L1 2,991개, 관측 후보 쌍 189,735개, L2 5,972개, 규칙 8,331개를 얻었다. L1의 이론적 쌍 수는 4,471,545개로, 빈발 항목에 대한 가지치기와 관측된 쌍 집계로 계산 범위를 제한하였다. 규칙의 consequent는 1,110편으로 전체 후보의 약 1.22%뿐이다. 저장된 진단에서는 숨긴 참고문헌의 약 60.1%가 어떤 규칙의 consequent에도 없었다.')
p('참고문헌이 짧으면 적용 규칙이 줄어 빈 목록이 많아진다. A3의 fallback은 평균 약 31.3%의 질의에서 쓰였으며 규칙이 있는 질의에서도 부족한 자리를 채웠다. 인기순 보충의 개선은 작아 빈도 기반 규칙의 범위 제한을 해결하지 못했다. 이 성능을 참고문헌 5개 미만인 논문, 외부 문헌, 완전한 신규 분야로 일반화할 수 없다.')
end()

# Page 9
h('5.4 정적 관계 시각화와 계산 비용')
p('선택 seed=42로 군집 1·13·14와 논문 14544·15942·33989·117825·145246을 추출하였다. 군집은 ID별 균등 추출하고 논문은 고립 노드를 제외한 86,940편에서 추출하였다. 표시 한도는 80노드·250방향간선, 축소 seed=43, layout seed=44이다. 노드 색은 내용 군집, 면적은 전체 snapshot PageRank의 99분위수 제한 값이다. 군집 13은 전체 5,829편이지만 anchor의 2-hop에서 3노드·2간선만 표시되었다. 이를 군집 전체의 구조로 해석하지 않았다. 실제 노드·간선·좌표·metadata와 8개 PNG·SVG를 보존하였다.')
table('표 11 동일 질의에서의 추천 자원 비교', ['방법','학습 초','추론과 지표 평균 초','합계 평균 초'],[
 ['B2','0.005957','0.318167','0.324124'],
 ['A2','0.708216','0.543128','1.251344'],
 ['A3','0.714173','0.559470','1.273643']], [100,110,160,WIDTH-370])
p('시간은 별도 benchmark에서 같은 후보·질의·평가 범위로 측정하였다. 학습은 한 번 측정해 마스크들에 공유했으므로 반복 학습의 평균이 아니다. 방법 간 RSS는 약 224~242MiB의 전체 프로세스 표본 최대치였다. 이 관측에서는 규칙 추천이 인기순보다 비용이 크지만, 품질과 추천 범위도 함께 달라 단일 시간만으로 방법을 선택할 수 없다.')
h('6 Conclusion and limitations')
p('세 연구 질문의 답은 서로 달랐다. 과거 PageRank는 다음 해 인용 순위에서 In-degree보다 낮았다. 내용 군집의 분리는 약했지만 동일 군집 내 참고문헌 공유와 직접 인용이 무작위 기준보다 높았다. Apriori는 적격 질의에서 인기순보다 높은 복원 Recall을 얻었으나 규칙의 대상 범위가 좁았다. 따라서 단기 순위에는 단순 인용 수가 강한 기준이고, 내용 군집은 문헌 정리, 공동 인용 규칙은 부분적인 참고문헌 추천에 활용할 수 있다는 제한된 근거를 얻었다.')
p('한계는 내부 인용만 관측한다는 점, 제공 임베딩의 과거 시점 순수성이 보장되지 않는다는 점, 연도 단위 평가와 적격 질의 선택 편향, 단일 데이터셋, 초기화·마스크 반복을 통계적 신뢰구간으로 볼 수 없다는 점이다. 선택한 α=0.70, k=20, count=20, confidence=0.1은 모두 후보 범위의 경계이므로 전역 최적이 아니다. 후속 연구는 새로운 검증 구간에서 탐색 범위를 확장하고, 시간 보정 PageRank와 희소·짧은 참고문헌 질의의 보완 방법을 평가하는 것이다. 최종 평가 점수로 현재 설정을 다시 고르지는 않았다.')
h('AI-use disclosure')
p('저장소 AI_USE.md에 따르면 2026년 11월 6~9일 OpenAI Codex의 GPT-6 Astra를 구현·획득·실험·테스트·그림 작성에, Anthropic Claude Opus 5.5를 교차 검토에 사용하였다. OpenAI ChatGPT GPT-6는 방법론과 결과 검토에 사용하였다. 영향 범위는 src, tests, configs, README, docs, data 안내, handoff, outputs이다. 본 보고서는 2026년 11월 14일 OpenAI Codex의 도움으로 전체 본문·표·편집을 작성하였다. 이번 작성에 사용한 정확한 모델 버전은 확인하지 않아 기재하지 않는다.')
p('코드 단계는 작은 예제·독립 구현·누수 방지·저장 예측 검사를 사용했고, 보고서 단계는 원 CSV의 재집계, 코드·동결 설정·진단·원출처 대조로 수치와 해석을 확인하였다. AI 출력은 학술 근거로 인용하지 않았다. 윤지원은 데이터 수집·AI 활용 구현 관리·결과 검증, 조성진은 보고서 작성을 담당하였다. 제출 내용의 최종 확인과 책임은 팀에 있다.')
end()

# Page 10
h('References')
refs=[
 '[1] Open Graph Benchmark, “Node Property Prediction: ogbn-arxiv,” dataset ver. 1, OGB and MAG, ODC-BY. [Online]. Available: https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv (accessed Nov. 14, 2026).',
 '[2] L. Page, S. Brin, R. Motwani, and T. Winograd, “The PageRank Citation Ranking: Bringing Order to the Web,” Stanford InfoLab, Tech. Rep. 1999-66, 1999. Available: http://ilpubs.stanford.edu:8090/422/.',
 '[3] P. Chen, H. Xie, S. Maslov, and S. Redner, “Finding scientific gems with Google’s PageRank algorithm,” Journal of Informetrics, vol. 1, no. 1, pp. 8-15, 2007, doi: 10.1016/j.joi.2006.06.001.',
 '[4] M. S. Mariani, M. Medo, and Y.-C. Zhang, “Identification of milestone papers through time-balanced network centrality,” Journal of Informetrics, vol. 10, no. 4, pp. 1207-1223, 2016, doi: 10.1016/j.joi.2016.10.005.',
 '[5] M. M. Kessler, “Bibliographic coupling between scientific papers,” American Documentation, vol. 14, no. 1, pp. 10-25, 1963, doi: 10.1002/asi.5090140103.',
 '[6] H. Small, “Co-citation in the scientific literature: A new measure of the relationship between two documents,” JASIS, vol. 24, no. 4, pp. 265-269, 1973, doi: 10.1002/asi.4630240406.',
 '[7] S. Lee, “Link Analysis,” pp. 11, 13, 18, 27, 29, 31; “Clustering,” pp. 34-40; “Association Rules,” pp. 34-42; “Dimension Reduction,” pp. 52-59, 70, SCE331 Data Mining lecture materials, Ajou University.',
 '[8] scikit-learn developers, “KMeans,” “PCA,” “silhouette_score,” and “adjusted_rand_score,” ver. 1.8.0 documentation. [Online]. Available: https://scikit-learn.org/1.8/modules/classes.html (accessed Nov. 14, 2026).',
 '[9] R. Agrawal and R. Srikant, “Fast Algorithms for Mining Association Rules,” in Proc. 20th Int. Conf. Very Large Data Bases, 1994, pp. 487-499. Available: https://www.vldb.org/conf/1994/P487.PDF.',
 '[10] K. Järvelin and J. Kekäläinen, “Cumulated gain-based evaluation of IR techniques,” ACM TOIS, vol. 20, no. 4, pp. 422-446, 2002, doi: 10.1145/582415.582418.',
 '[11] NetworkX developers, “pagerank and drawing,” ver. 3.4.2 documentation. [Online]. Available: https://networkx.org/documentation/networkx-3.4.2/ (accessed Nov. 14, 2026).',
 '[12] NumPy, SciPy, pandas, Apache Arrow, and Matplotlib developers, official software documentation. Available: https://numpy.org/doc/ ; https://docs.scipy.org/doc/scipy/ ; https://pandas.pydata.org/docs/ ; https://arrow.apache.org/docs/python/ ; https://matplotlib.org/stable/ . Versions: requirements-lock.txt.'
]
for s in refs:current.append(('r',s))
h('외부 코드와 소프트웨어의 활용')
p('PageRank 반복, pair-only Apriori, 시간 분할·질의·평가 코드는 프로젝트용으로 AI 도움을 받아 작성하였다. 다른 저장소 코드를 복사·vendor한 파일은 없다는 저장소 출처 기록에 따른다. k-means·PCA의 수치 계산은 scikit-learn [8], 작은 그래프의 독립 비교·시각화는 NetworkX [11], 배열·희소행렬·표·그림은 [12]를 사용하였다. 직접 구현과 library 호출을 구분하였다. BSD 계열, Apache-2.0, Matplotlib 등의 고지는 각 설치 배포물의 라이선스를 따른다.')
h('GitHub')
p('Repository URL  [입력 필요]')
p('제출 버전의 전체 commit hash  [입력 필요]')
p('실험 결과 경로  outputs/arxiv_v2_20261106_review_20261107/\n비공개 제출 시 가이드라인에 지정된 leesael 또는 sael@ajou.ac.kr의 접근 권한을 설정한다.')
end()

def make_items(page):
    items=[]
    for e in page:
        if e[0] in ('p','h','title','r'):
            kind={'p':'body','h':'head','title':'title','r':'ref'}[e[0]]
            items.append(para(e[1],kind))
        elif e[0]=='t':
            _,caption,head,rows,widths=e
            data=[[para(v,'cell') for v in row] for row in [head]+rows]
            t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([
             ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#efefef')),
             ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),
             ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),
             ('BOTTOMPADDING',(0,0),(-1,-1),4),('LINEBELOW',(0,0),(-1,0),.5,colors.black),
             ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#bbbbbb'))]))
            bundle=[]
            if caption:bundle.append(para(caption,'caption'))
            bundle.extend([t,Spacer(1,6)])
            items.extend(bundle)
        elif e[0]=='f':
            _,rel,caption,width=e
            im=Image(str(RUN/rel)); ratio=im.imageHeight/im.imageWidth
            im.drawWidth=width;im.drawHeight=width*ratio;im.hAlign='CENTER'
            items.extend([im,Spacer(1,3),para(caption,'caption')])
    return items

def footer(canvas,doc):
    canvas.setFont('EN',10);canvas.drawCentredString(W/2,10*mm,str(doc.page))

# Keep page boundaries deterministic; fail rather than silently shrinking text.
measure=[];story=[]
for i,page in enumerate(pages,1):
    elements=make_items(page); total=0
    for el in elements:
        _,eh=el.wrap(WIDTH,10000)
        total+=eh+el.getSpaceBefore()+el.getSpaceAfter()
    measure.append({'planned_page':i,'height_pt':round(total,2),'available_pt':round(H-36*mm,2)})
    if i>1:story.append(PageBreak())
    story.extend(elements)
print('PAGE MEASURES',json.dumps(measure))
doc=SimpleDocTemplate(str(OUT/'G13_Report.pdf'),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=18*mm,bottomMargin=18*mm,title='시간 제한 인용망에서 논문의 영향력과 내용 군집 및 참고문헌 복원',author='G13 윤지원 조성진',pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)

# Editable manuscript shares exactly the same text, numbers, and table content.
md=[]
for i,page in enumerate(pages,1):
    for e in page:
        if e[0] in ('p','r'):md.extend([e[1],''])
        elif e[0]=='title':md.extend(['# '+e[1],''])
        elif e[0]=='h':md.extend(['## '+e[1],''])
        elif e[0]=='t':
            _,caption,head,rows,widths=e
            if caption:md.extend([caption,''])
            md.append('| '+' | '.join(head)+' |')
            md.append('| '+' | '.join(['---']*len(head))+' |')
            md.extend('| '+' | '.join(str(v).replace('\n','<br>') for v in row)+' |' for row in rows);md.append('')
        elif e[0]=='f':
            rel=os.path.relpath(RUN/e[1],OUT).replace('\\','/')
            md.extend([f'![{e[2]}]({rel})','',e[2],''])
(OUT/'G13_Report.md').write_text('\n'.join(md),encoding='utf-8')
pdf=PdfReader(str(OUT/'G13_Report.pdf'))
texts=[page.extract_text() for page in pdf.pages]
qa={'pages':len(pdf.pages),'page_measurements':measure,'page_starts':[t[:90] for t in texts], 'chars_by_page':[len(t) for t in texts], 'body_font_pt':10,'body_leading_pt':15,'korean_font':'Malgun Gothic','english_font':'Times New Roman','missing_submission_fields':['GitHub URL','full commit hash']}
(QA/'qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
(QA/'extracted.txt').write_text('\n\n'.join(texts),encoding='utf-8')
print(json.dumps(qa,ensure_ascii=False,indent=2))
assert len(pdf.pages)==10,f'Expected 10 pages, got {len(pdf.pages)}'
