# References·외부 코드

본문의 번호는 아래 출처와 대응합니다. 최종 보고서 담당자는 실제 인용 위치에 맞춰 IEEE식 번호를 일관되게 사용해야 합니다.

[1] Open Graph Benchmark, “Node Property Prediction: ogbn-arxiv,” dataset version 1. [Online]. Available: https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv. 제공자 OGB/MAG; 원 논문·분야 라벨의 저자 및 arXiv 출처를 함께 표시. 라이선스 ODC-BY.

[2] S. Lee, “Link Analysis,” SCE331 Data Mining lecture, Ajou University, `DM_11-Link Analysis.pdf`, pp.11,13,18,27,29,31. 슬라이드에 Mining of Massive Datasets 출처가 명시돼 있음.

[3] S. Lee, “Clustering,” SCE331 Data Mining lecture, Ajou University, `DM_05-Clustering.pdf`, pp.34–40.

[4] S. Lee, “Association Rules,” SCE331 Data Mining lecture, Ajou University, `DM_03-Association Rules.pdf`, pp.34–42.

[5] S. Lee, “Dimension Reduction,” SCE331 Data Mining lecture, Ajou University, `DM_06-Dimension Reduction.pdf`, pp.52–59,70.

[6] R. Agrawal and R. Srikant, “Fast Algorithms for Mining Association Rules,” in *Proc. 20th Int. Conf. Very Large Data Bases*, 1994, pp.487–499. [Online]. Available: https://www.vldb.org/conf/1994/P487.PDF.

[7] scikit-learn developers, “KMeans” and “PCA,” version 1.8.0 documentation. [Online]. Available: https://scikit-learn.org/1.8/modules/generated/sklearn.cluster.KMeans.html and https://scikit-learn.org/1.8/modules/generated/sklearn.decomposition.PCA.html. 사용: Lloyd/k-means++, full PCA, Silhouette, ARI 및 테스트의 NDCG 기준 구현.

[8] NetworkX developers, “pagerank,” version 3.4.2 documentation. [Online]. Available: https://networkx.org/documentation/networkx-3.4.2/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html. 사용: 작은 그래프의 독립 PageRank 비교, 그림의 그래프 처리·배치. 수렴 비교 테스트는 NetworkX의 `err < N * tol`에 맞춰 tol을 N으로 나눠 설정.

[9] NumPy, SciPy, pandas, Apache Arrow and Matplotlib developers, official documentation. [Online]. Available: https://numpy.org/doc/ , https://docs.scipy.org/doc/scipy/ , https://pandas.pydata.org/docs/ , https://arrow.apache.org/docs/python/ , https://matplotlib.org/stable/ . 사용: 배열·희소 행렬·상관·표·Parquet·그림. 실제 버전은 requirements-lock.txt.

[10] SCE331 Data Mining, “프로젝트 안내 및 평가 rubric,” `SCE331_Project_Guidelines_Kor.pdf`, pp.1–5; “Major Algorithms by Topic,” `SCE331_DM_Algorithms_by_Topic.pdf`; “프로젝트 주제 아이디어,” `SCE331_Project_Topic_Ideas_Kor.pdf`. 강의자료. 평가 기준·알고리즘 범위 확인에 사용했으며 강의 자료는 저장소에 재배포하지 않음.

[11] W. Hu, M. Fey, M. Zitnik, Y. Dong, H. Ren, B. Liu, M. Catasta, and J. Leskovec, “Open Graph Benchmark: Datasets for Machine Learning on Graphs,” in *Advances in Neural Information Processing Systems*, vol. 33, 2020. [공식 논문](https://proceedings.neurips.cc/paper/2020/hash/fb60d411a5c5b72b2e7d3527cfc84fd0-Abstract.html). OGB 데이터셋과 벤치마크 설계의 출처.

[12] K. Wang, Z. Shen, C. Huang, C.-H. Wu, Y. Dong, and A. Kanakia, “Microsoft Academic Graph: When experts are not enough,” *Quantitative Science Studies*, vol. 1, no. 1, pp. 396–413, 2020, doi: 10.1162/qss_a_00021. [DOI](https://doi.org/10.1162/qss_a_00021), [저자 기관의 논문 페이지](https://www.microsoft.com/en-us/research/publication/microsoft-academic-graph-when-experts-are-not-enough/). MAG의 데이터 구성·출처에 사용.

[13] L. Page, S. Brin, R. Motwani, and T. Winograd, “The PageRank Citation Ranking: Bringing Order to the Web,” Stanford InfoLab, Technical Report 1999-66, 1999. [공식 보고서](http://ilpubs.stanford.edu:8090/422/). 인용 기반 중요도와 random-surfer 해석의 원 논문.

[14] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean, “Distributed Representations of Words and Phrases and their Compositionality,” in *Advances in Neural Information Processing Systems*, vol. 26, pp. 3111–3119, 2013. [공식 논문](https://proceedings.neurips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html). OGB 공식 설명에서 인용하는 skip-gram 단어 임베딩의 원 논문이며, 본 프로젝트가 임베딩을 새로 학습한 것은 아님.

[15] K. Järvelin and J. Kekäläinen, “Cumulated gain-based evaluation of IR techniques,” *ACM Transactions on Information Systems*, vol. 20, no. 4, pp. 422–446, 2002, doi: 10.1145/582415.582418. [DOI](https://doi.org/10.1145/582415.582418). NDCG의 원 논문. 본 프로젝트는 relevance r=log2(1+c)에 선형 gain을 사용.

[16] P. J. Rousseeuw, “Silhouettes: A graphical aid to the interpretation and validation of cluster analysis,” *Journal of Computational and Applied Mathematics*, vol. 20, pp. 53–65, 1987, doi: 10.1016/0377-0427(87)90125-7. [DOI](https://doi.org/10.1016/0377-0427%2887%2990125-7). Silhouette의 원 논문.

[17] L. Hubert and P. Arabie, “Comparing partitions,” *Journal of Classification*, vol. 2, no. 1, pp. 193–218, 1985, doi: 10.1007/BF01908075. [DOI](https://doi.org/10.1007/BF01908075). Adjusted Rand Index의 원 논문.

[18] D. Arthur and S. Vassilvitskii, “k-means++: The advantages of careful seeding,” in *Proc. 18th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA)*, 2007, pp. 1027–1035. k-means++ 초기화의 원 논문. scikit-learn KMeans의 기본 초기화.

[19] S. P. Lloyd, “Least squares quantization in PCM,” *IEEE Transactions on Information Theory*, vol. 28, no. 2, pp. 129–137, 1982, doi: 10.1109/TIT.1982.1056489. [DOI](https://doi.org/10.1109/TIT.1982.1056489). k-means 반복(Lloyd 알고리즘)의 원 논문.

[20] P. Chen, H. Xie, S. Maslov, and S. Redner, “Finding scientific gems with Google’s PageRank algorithm,” *Journal of Informetrics*, vol. 1, no. 1, pp. 8–15, 2007, doi: 10.1016/j.joi.2006.06.001. [DOI](https://doi.org/10.1016/j.joi.2006.06.001). 인용망에 PageRank를 적용한 선행연구.

[21] M. S. Mariani, M. Medo, and Y.-C. Zhang, “Identification of milestone papers through time-balanced network centrality,” *Journal of Informetrics*, vol. 10, no. 4, pp. 1207–1223, 2016, doi: 10.1016/j.joi.2016.10.005. [DOI](https://doi.org/10.1016/j.joi.2016.10.005). 인용망 PageRank의 출판 시점 편향과 시간 보정 지표.

[22] M. M. Kessler, “Bibliographic coupling between scientific papers,” *American Documentation*, vol. 14, no. 1, pp. 10–25, 1963, doi: 10.1002/asi.5090140103. [DOI](https://doi.org/10.1002/asi.5090140103). 참고문헌 공유(bibliographic coupling)의 원 논문.

[23] H. Small, “Co-citation in the scientific literature: A new measure of the relationship between two documents,” *Journal of the American Society for Information Science*, vol. 24, no. 4, pp. 265–269, 1973, doi: 10.1002/asi.4630240406. [DOI](https://doi.org/10.1002/asi.4630240406). 공동 인용(co-citation)의 원 논문.

## 직접 작성한 코드와 라이브러리 사용의 구분

다른 저장소의 코드를 복사하거나 vendor한 파일은 없습니다. PageRank 반복, pair-only Apriori, 선형 gain NDCG, 시간 snapshot·질의 마스크·층화 pair pool·실행 연결·검증 코드는 이 프로젝트용으로 AI 도움을 받아 작성했습니다. k-means와 PCA의 수치 계산은 scikit-learn을 호출하므로 직접 수치 구현했다고 설명하지 않습니다. 외부 패키지의 알고리즘을 사용한 부분과 프로젝트가 정의한 평가·데이터 준비를 구별합니다.

NumPy/SciPy/pandas/scikit-learn/NetworkX는 BSD 계열, PyArrow는 Apache-2.0, Matplotlib은 Matplotlib 라이선스를 사용합니다. 정확한 고지문은 설치 배포물의 라이선스 파일을 따릅니다. 프로젝트 코드에 새 공개 라이선스를 임의로 부여하지 않았습니다. 공개 여부·프로젝트 라이선스는 팀이 정합니다. 데이터·논문 텍스트의 라이선스와 코드 라이선스를 혼동하지 않습니다.
