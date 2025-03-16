# TensorflowSui v.2.0.1

TensorflowSui는 Sui 블록체인 위에서 머신러닝 모델을 온체인으로 배포하고 추론할 수 있게 해주는 프로젝트입니다. 이 프로젝트는 Tensorflow를 Web3 Sui 버전으로 구현하여 완전한 온체인 추론(fully onchain inference)을 가능하게 합니다.

## 주요 기능

- 온체인 신경망 모델 배포
- 완전한 온체인 추론(fully onchain inference)
- 사용자 정의 모델 지원 (v.2.0.1 신규 기능)

## 모듈 구조

TensorflowSui는 다음과 같은 주요 모듈로 구성되어 있습니다:

1. **tensor**: 텐서 데이터 구조와 연산을 정의합니다.
2. **graph**: 신경망 그래프 구조와 레이어를 정의합니다.
3. **model**: 모델 생성 및 초기화 기능을 제공합니다.

## v.2.0.1 신규 기능: 사용자 정의 모델

v.2.0.1에서는 사용자가 직접 모델 데이터를 입력하여 커스텀 모델을 생성할 수 있는 기능이 추가되었습니다. 이전 버전에서는 MNIST 모델이 하드코딩되어 있어 확장성이 제한적이었지만, 이제는 다양한 모델을 온체인에 배포할 수 있습니다.

### 사용자 정의 모델 생성 방법

`initialize_custom_model` 함수를 사용하여 사용자 정의 모델을 생성할 수 있습니다:

```move
public entry fun initialize_custom_model(
    ctx: &mut TxContext,
    layer_names: vector<vector<u8>>,
    layer_in_dims: vector<u64>,
    layer_out_dims: vector<u64>,
    weights_magnitudes: vector<vector<u64>>,
    weights_signs: vector<vector<u64>>,
    biases_magnitudes: vector<vector<u64>>,
    biases_signs: vector<vector<u64>>,
    scale: u64
)
```

#### 매개변수 설명

- `ctx`: Sui 트랜잭션 컨텍스트
- `layer_names`: 각 레이어의 이름 (예: `["dense1", "dense2", "output"]`)
- `layer_in_dims`: 각 레이어의 입력 차원 (예: `[49, 16, 8]`)
- `layer_out_dims`: 각 레이어의 출력 차원 (예: `[16, 8, 10]`)
- `weights_magnitudes`: 각 레이어의 가중치 크기 벡터
- `weights_signs`: 각 레이어의 가중치 부호 벡터 (0: 음수, 1: 양수)
- `biases_magnitudes`: 각 레이어의 편향 크기 벡터
- `biases_signs`: 각 레이어의 편향 부호 벡터 (0: 음수, 1: 양수)
- `scale`: 고정 소수점 스케일 (예: 2는 10^2 = 100을 의미)

### 사용 예시

다음은 3개의 레이어를 가진 간단한 모델을 생성하는 예시입니다:

```move
// 레이어 이름 정의
let layer_names = vector[b"dense1", b"dense2", b"output"];

// 레이어 차원 정의
let layer_in_dims = vector[49, 16, 8];
let layer_out_dims = vector[16, 8, 10];

// 가중치와 편향 데이터 정의 (간략화된 예시)
let weights_magnitudes = vector[w1_mag, w2_mag, w3_mag];
let weights_signs = vector[w1_sign, w2_sign, w3_sign];
let biases_magnitudes = vector[b1_mag, b2_mag, b3_mag];
let biases_signs = vector[b1_sign, b2_sign, b3_sign];

// 모델 초기화
initialize_custom_model(
    ctx,
    layer_names,
    layer_in_dims,
    layer_out_dims,
    weights_magnitudes,
    weights_signs,
    biases_magnitudes,
    biases_signs,
    2 // 스케일 (10^2 = 100)
);
```

### 외부 모델 변환

TensorFlow나 PyTorch 같은 외부 프레임워크에서 학습된 모델을 TensorflowSui 형식으로 변환하려면 다음 단계를 따르세요:

1. 모델 가중치와 편향을 추출합니다.
2. 가중치와 편향을 크기(magnitude)와 부호(sign) 벡터로 변환합니다.
3. 변환된 데이터를 `initialize_custom_model` 함수에 전달합니다.

## 레거시 지원

v.2.0.1은 이전 버전과의 호환성을 위해 기존 MNIST 모델 생성 기능도 계속 지원합니다:

```move
public entry fun initialize(ctx: &mut TxContext)
```

## 라이센스

Apache-2.0



