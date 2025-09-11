# 🤖 Transformer from Scratch (Toy Project)

"Attention Is All You Need" (Vaswani et al., 2017) 논문을 기반으로 Transformer 모델을 밑바닥부터(from scratch) 구현한 프로젝트입니다. 단순한 논문 복제를 넘어, 현대적인 언어 모델(LLM)에 사용되는 핵심 개선 사항인 **RoPE**와 **Pre-Layer Normalization**을 직접 구현하고 적용하여 모델의 작동 원리를 깊이 있게 탐구하는 것을 목표로 했습니다.

## 🎯 프로젝트 목표

-   Transformer의 Encoder-Decoder 구조 및 Self-Attention 메커니즘을 완벽히 이해하고 구현합니다.
-   최신 위치 임베딩 방식인 **RoPE (Rotary Positional Embedding)** 를 구현하여 절대 위치가 아닌 상대 위치 정보를 모델에 주입합니다.
-   **Pre-Layer Normalization** 구조를 적용하여 깊은 모델의 학습 안정성을 확보하는 원리를 체득합니다.
-   PyTorch를 사용하여 재현 가능하고 확장성 있는 모델 코드를 작성하는 능력을 기릅니다.

---

## ✨ 핵심 구현 내용 (Key Features)

이 프로젝트는 다음과 같은 Transformer의 핵심 요소 및 개선 사항을 직접 구현했습니다.

-   **Encoder-Decoder Architecture**: 입력 시퀀스의 정보를 압축하는 인코더와, 이를 바탕으로 새로운 시퀀스를 생성하는 디코더의 전체 구조
-   **Multi-Head Self-Attention**: Q, K, V를 사용해 시퀀스 내 단어 간의 복잡한 관계를 병렬적으로 학습하는 메커니즘
-   **Rotary Positional Embedding (RoPE)**: 임베딩 벡터를 '회전'시켜 절대 위치가 아닌 토큰 간의 '상대 위치' 정보를 주입하는 최신 위치 인코딩 방식
-   **Pre-Layer Normalization**: Sub-layer에 입력을 넣기 전에 정규화를 수행하여 그래디언트 흐름을 안정시키고 깊은 모델의 학습을 원활하게 하는 구조
-   **Masking**:
    -   **Padding Mask**: 길이를 맞추기 위한 `<pad>` 토큰이 어텐션 계산에 영향을 주지 않도록 마스킹
    -   **Look-ahead Mask**: 디코더가 예측 시 미래의 정답 토큰을 미리 볼 수 없도록(cheating 방지) 마스킹

---

## 🔧 실행 방법 (How to Run)

1.  **저장소 복제 및 이동**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name/transformer-toy
    ```

2.  **필요 라이브러리 설치**
    ```bash
    pip install torch
    ```
    *(이 프로젝트는 `torch` 외의 특별한 라이브러리를 요구하지 않습니다.)*

3.  **학습 스크립트 실행**
    ```bash
    python train.py
    ```

---

## 📊 학습 결과 예시 (Expected Results)

`train.py`를 실행하면 숫자 시퀀스를 뒤집는 Task에 대해 모델이 학습되며, 다음과 같은 결과를 확인할 수 있습니다.

**1. 학습 과정 (Training Process)**
```text
Training started...
Epoch: 01, Train Loss: 2.1534
Epoch: 02, Train Loss: 1.7421
...
Epoch: 49, Train Loss: 0.0158
Epoch: 50, Train Loss: 0.0142

Training finished!
