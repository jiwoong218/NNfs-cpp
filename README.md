# NNfs-cpp

Layer 중심 설계
- 성능과 확장성 고려

$$Y = XW + b$$
하나의 Layer는 이전 Layer의 데이터(X)를 받아 가중치(W)를 곱하고 편향(b)를 더해 다음 Layer로 전달한다.

가중치(W): 이전 층의 노드와 현재 층의 노드의 연결을 나타내는 *2차원 행렬*
편향(b): 현재 층의 노드 수만큼 존재 *1차원 벡터*

std::vector
reserve(size): 메모리 공간만 미리 예약, 데이터 들어갈 공간은 만들지 않음
resize(size, init_value): 데이터 공간 만들고 초기화 수행


