from core.spot.model.SPOT_ablation import SPOT_Ablation
from dataset import TimeSformerHighlightDataset
from results import metadata, log_folder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import csv

# ✅ 고정 설정
video_dir = r"J:\downloaded_videos"
label_dir = log_folder
metadata_csv = metadata.metadata_csv
frame_count = 24
batch_size = 4
epochs = 10
learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ 사용할 CNN (빠른 버전)
cnn_backbone = "mobilenet_v3_small"

# ✅ 실험 대상 (full 제외)
ablation_settings = {
    "no_cafgl":    {"use_cnn": True,  "use_patch": True,  "use_cafgl": False, "use_transformer": True},
    "no_cnn":      {"use_cnn": False, "use_patch": True,  "use_cafgl": False, "use_transformer": True},
    "no_patch":    {"use_cnn": True,  "use_patch": False, "use_cafgl": False, "use_transformer": True},
    "no_temporal": {"use_cnn": True,  "use_patch": True,  "use_cafgl": True,  "use_transformer": False},
}

# ✅ Dataset 구성 (1회만 고정)
dataset = TimeSformerHighlightDataset(
    video_dir=video_dir,
    label_dir=label_dir,
    metadata_path=metadata_csv,
    frame_count=frame_count,
    image_size=(224, 224)
)

indices = np.arange(len(dataset))
train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_idx)}")

# ✅ Ablation 실험 루프 (1회만)
for setting_name, config in ablation_settings.items():
    print(f"\n=== 🔍 Starting Ablation: {setting_name} ===")

    log_data = []

    # 모델 초기화
    model = SPOT_Ablation(
        num_frames=frame_count,
        num_classes=frame_count,
        cnn_backbone=cnn_backbone,
        **config
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # 저장 경로
    checkpoint_dir = f"trained_models_ablation/{setting_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for frames, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train"):
            frames, targets = frames.to(device), targets.to(device)
            outputs = model(frames)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Val"):
                frames, targets = frames.to(device), targets.to(device)
                outputs = model(frames)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"[{setting_name} | Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        log_data.append({'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})

        # 체크포인트 저장
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(checkpoint_dir, f"checkpoint_{epoch+1}.pth"))

    # 로그 저장
    log_path = os.path.join(checkpoint_dir, "log.csv")
    with open(log_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['epoch', 'train_loss', 'val_loss'])
        writer.writeheader()
        writer.writerows(log_data)

    print(f"📄 Log saved to {log_path}")
