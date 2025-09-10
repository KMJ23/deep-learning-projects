import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import math

# models 폴더의 transformer.py에서 Transformer 클래스를 가져옵니다.
from models.transformer import Transformer

# ----------------------------------------------------------------
# 1. 하이퍼파라미터 및 설정
# ----------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 특별 토큰 정의
PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2
# 단어 집합 크기 (0~9 숫자 + 특별 토큰 3개)
VOCAB_SIZE = 10 + 3

# 모델 하이퍼파라미터
D_MODEL = 128    # 임베딩 차원
NHEAD = 4        # 어텐션 헤드 개수
D_FF = 512       # 피드포워드 신경망 차원
NUM_ENCODER_LAYERS = 3 
NUM_DECODER_LAYERS = 3
DROPOUT = 0.1

# 학습 하이퍼파라미터
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.0001

# ----------------------------------------------------------------
# 2. 데이터셋 생성
# ----------------------------------------------------------------
class SeqReverseDataset(Dataset):
    def __init__(self, num_samples=10000, min_len=5, max_len=12):
        self.samples = []
        for _ in range(num_samples):
            length = random.randint(min_len, max_len)
            # 3부터 12까지의 숫자 (단어 집합 인덱스에 맞춤)
            seq = [random.randint(3, VOCAB_SIZE - 1) for _ in range(length)]

            src = [SOS_IDX] + seq + [EOS_IDX]
            tgt = [SOS_IDX] + seq[::-1] + [EOS_IDX]
            self.samples.append((src, tgt))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx][0]), torch.tensor(self.samples[idx][1])

def collate_fn(batch):
    """배치 내의 시퀀스 길이를 맞추기 위한 padding 함수"""
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded



# ----------------------------------------------------------------
# 3. 학습 및 평가 로직
# ----------------------------------------------------------------
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        # Teacher Forcing을 위해 디코더 입력을 준비합니다. (마지막 토큰 제외)
        decoder_input = tgt[:, :-1]

        # Loss 계산을 위한 타겟을 준비합니다. (첫 SOS 토큰 제외)
        target = tgt[:, 1:]

        optimizer.zero_grad()

        # 모델 순전파
        output = model(src, decoder_input)

        # Loss 계산
        # output: [Batch, Seq_len, Vocab_size] -> [Batch * Seq_len, Vocab_size]
        # target: [Batch, Seq_len] -> [Batch * Seq_len]
        loss = criterion(output.reshape(-1, VOCAB_SIZE), target.reshape(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, src_sentence):
    """학습된 모델로 하나의 문장을 번역(추론)하는 함수"""
    model.eval()
    src = torch.tensor([SOS_IDX] + src_sentence + [EOS_IDX]).unsqueeze(0).to(DEVICE)

    # 디코더의 첫 입력은 SOS 토큰
    tgt_tokens = [SOS_IDX]

    for i in range(len(src_sentence) + 5): # 최대 길이 제한
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(src, tgt_tensor)

        # 다음 토큰 예측 (가장 확률이 높은 단어 선택 - Greedy Search)
        pred_token = output.argmax(2)[:, -1].item()
        tgt_tokens.append(pred_token)

        # EOS 토큰이 나오면 종료
        if pred_token == EOS_IDX:
            break
    
    return tgt_tokens

# ----------------------------------------------------------------
# 4. 메인 실행 블록
# ----------------------------------------------------------------
if __name__ == "__main__":
    # 데이터셋 및 데이터로더 생성
    train_dataset = SeqReverseDataset(num_samples=10000)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 모델, Loss 함수, 옵티마이저 초기화
    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT
    ).to(DEVICE)

    # Padding 토큰은 Loss 계산에서 제외
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Training started...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion)
        print(f"Epoch {epoch:02d}, Loss: {train_loss:.4f}")

    print("\nTraining finished!")

    # 테스트 예시
    test_seq = [5, 2, 8, 9, 3]
    result = evaluate(model, test_seq)
    print(f"Input:     {[SOS_IDX] + test_seq + [EOS_IDX]}")
    print(f"Expected:  {[SOS_IDX] + test_seq[::-1] + [EOS_IDX]}")
    print(f"Result:    {result}")

