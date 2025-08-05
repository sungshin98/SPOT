import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import pandas as pd
import csv
import os
import glob

from core.spot.model.SPOT_res18 import SPOT # ✅ CNN+TimeSformer 모델 import
from dataset import TimeSformerHighlightDataset  # ✅ 새로운 DataLoader
from results import metadata, log_folder


video_dir = r"J:\downloaded_videos"
label_dir = log_folder
metadata_csv = metadata.metadata_csv
log_data = []

frame_count = 24
batch_size = 4
epochs = 20
learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ Dataset 구성
dataset = TimeSformerHighlightDataset(
    video_dir=video_dir,
    label_dir=label_dir,
    metadata_path=metadata_csv,
    frame_count=frame_count,
    image_size=(224, 224)
)

# ✅ 데이터셋 분할
indices = np.arange(len(dataset))
train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

# ✅ video_id 저장
video_ids = [dataset.video_ids[i] for i in test_idx]
pd.DataFrame({'video_id': video_ids}).to_csv("../test/test_video_list.csv", index=False)

# ✅ DataLoader 설정
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

print(f"Train set size: {len(train_idx)}")
print(f"Validation set size: {len(val_idx)}")
print(f"Test set size: {len(test_idx)}")
CNN = "resnet50"
# ✅ 모델 초기화 (frame 수만큼 출력)
model = SPOT(num_frames=24, num_classes=24, cnn_backbone = CNN).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ---------------------------
# 체크포인트 불러오기
# ---------------------------
checkpoint_dir = f"trained_model_{CNN}"
os.makedirs(checkpoint_dir, exist_ok=True)

# 저장된 체크포인트 중 가장 마지막 파일 찾기
checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "hybrid_checkpoint_*.pth"))
start_epoch = 0
if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    print(f"🔄 Found checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    # optimizer도 저장되어 있다면 로드
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}")
else:
    print("No checkpoint found. Starting from scratch.")

# ---------------------------
# 학습 루프 (start_epoch부터 시작)
# ---------------------------
for epoch in range(start_epoch, epochs):
    model.train()
    train_loss = 0.0
    for frames, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train"):
        frames, targets = frames.to(device), targets.to(device)
        outputs = model(frames)  # [B, T]
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # 검증 루프
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for frames, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Val"):
            frames, targets = frames.to(device), targets.to(device)
            outputs = model(frames)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    log_data.append({'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})

    # 체크포인트 저장 (optimizer 상태도 저장)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, f"{checkpoint_dir}/hybrid_checkpoint_{epoch+1}.pth")

# 로그 저장
with open("hybrid_training_log.csv", mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['epoch', 'train_loss', 'val_loss'])
    writer.writeheader()
    writer.writerows(log_data)

print("📄 Training log saved to hybrid_training_log.csv")
