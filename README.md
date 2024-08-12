# TSP Solving

이 프로젝트는 다양한 방법론을 이용하여 TSP(Traveling Salesman Problem) 문제를 해결한 결과를 비교합니다. 

사용된 데이터셋은 `ulysses16`, `att48`, 그리고 `att532`입니다. 각 방법론의 성능을 평가하여 결과를 분석하였습니다.

## 데이터셋

- **ulysses16**: 16개의 노드
- **att48**: 48개의 노드
- **att532**: 532개의 노드

데이터셋은 [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSP/data/)에서 다운로드할 수 있습니다.

## 테스트 시스템 사양
- **CPU**: AMD Ryzen 5 5600X
- **RAM**: 32GB
- **GPU**: NVIDIA RTX 3060 Ti

## 방법론

다양한 접근 방식이 사용:

1. **Greedy Algorithm**: 가장 가까운 노드를 선택하여 순회를 구성.
2. **N-Opt**: 현재 경로의 일부를 변경하여 더 짧은 경로를 탐색.
3. **Genetic Algorithm**: 유전자 알고리즘을 사용하여 최적의 경로를 탐색.
4. **Simulated Annealing**: 탐색을 통해 최적의 경로를 찾아가는 확률적 기법.
5. **Or-Tools Routing**: Google의 OR-Tools를 사용하여 TSP를 라우팅 기반으로 해결.
6. **Or-Tools CP-SAT**: Google의 OR-Tools를 사용한 제약 프로그래밍 기반 SAT 솔버.
7. **Or-Tools SCIP**: SCIP를 사용한 정수 프로그래밍 기법.
8. **Clustering TSP**: 클러스터링 기법을 사용하여 TSP를 해결. [링크](https://github.com/p-idx/TSP-project)
9. **Attention Policy RL**: 강화 학습 기반의 접근 방식. [링크](https://rl4.co/examples/3-creating-new-env-model/)

## 결과

다음은 각 방법론에 의해 얻어진 결과입니다. 

결과는 유클리디안 거리 기준으로 측정되었으며, 각 데이터셋에 대한 경로의 총 거리를 나타냅니다. 최소값은 **볼드체**로 표시되었습니다.

각 시간 초과는 4시간이상으로 걸렸을때의 기준입니다.

| 방법론                | ulysses16 | att48    | att532   |
|----------------------|------------|----------|----------|
| Greedy               | 104.73     | 40,526.42 | 112,099.45 |
| N-Opt(Max:3)                | 75.32      | 40,128.00 | 98,008.00  |
| Genetic Algorithm    | 73.90      | 33,899.00 | 시간 초과 |
| Simulated Annealing  | 시간 초과 | 시간 초과  | 시간 초과  |
| Or-Tools Routing         | 72.00      | 34,184.00 | **90,168.00**  |
| Or-Tools CP-SAT      |  **71.00**  | **33,522.00** | 시간 초과 |
| Or-Tools SCIP        | **71.00**  | 33,700.00 | 시간 초과 |
| Clustering TSP       | N/A        | 33,831.73 | 92,630.86 |
| Attention Policy RL  | 74.30      | 35,287.09 | 시간 초과 |
