# Transformer Toy Project

## 목표
- Self-Attention과 Encoder-Decoder 구조를 직접 구현
- 작은 병렬 코퍼스(영->불, toy dataset)로 번역 실험

## 구조
- `models/` : 모델 정의 (attention.py, transformer.py)
- `train.py` : 학습 코드
- `inference.py` : 추론 코드
- `results/` : 학습 로그, 그래프, 번역 결과 

## 실행 방법
```bash
python train.py --epochs 10 --batch_size 32
python inference.py --input "I like cats"