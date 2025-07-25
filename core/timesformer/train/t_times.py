import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from Timesformer.model.timesformer import TimeSformer
from Timesformer.dataset import TimeSformerHighlightDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# ------------------------------
# 하이퍼파라미터
epochs = 15
batch_size = 4
lr = 1e-4
frame_count = 24
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# ------------------------------
# Dataset 로드
dataset = TimeSformerHighlightDataset(
    video_dir=r"I:\downloaded_videos",
    label_dir=r"../../dataset/most_csv",
    metadata_path=r"../../data/dataset/metadata.csv",
    frame_count=frame_count
)

# ------------------------------
# Split (7:1.5:1.5)
indices = np.arange(len(dataset))
train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

# video_id 저장을 위한 작업
video_ids = [dataset.video_ids[idx] for idx in test_indices]
test_list_df = pd.DataFrame({'video_id': video_ids})
test_list_df.to_csv("test_video_list.csv", index=False)
print(f"✅ Test video list saved to test_video_list.csv")

# ------------------------------
# Subset과 DataLoader 생성
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader는 여기선 안 쓰고, 나중에 test용으로 따로 사용

print(f"Train set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# ------------------------------
# 모델 정의
model = TimeSformer(
    image_size=224,
    patch_size=16,
    num_frames=24,
    num_classes=24
)
model.head = nn.Linear(model.head.in_features, frame_count)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

checkpoint_path = "timesformer_checkpoint.pth"
start_epoch = 0

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed from checkpoint at epoch {start_epoch}")

# ------------------------------
# 학습 루프 (Train + Validation Loss)
for epoch in range(start_epoch, epochs):
    model.train()
    running_loss = 0.0

    for videos, targets in tqdm(train_loader):
        videos, targets = videos.to(device), targets.to(device)

        outputs = model(videos)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Validation loss 계산
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for videos, targets in val_loader:
            videos, targets = videos.to(device), targets.to(device)
            outputs = model(videos)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

    # Checkpoint 저장
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, checkpoint_path)

# 마지막 모델 저장
torch.save(model.state_dict(), "timesformer_highlight_final.pth")
