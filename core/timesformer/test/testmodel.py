import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from core.timesformer.model.timesformer import TimeSformer
from dataset.dataloader import TimeSformerHighlightDataset
import os
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from results import metadata, log_folder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
metadata_csv = metadata.metadata_csv
frame_count = 24
batch_size = 4

dataset = TimeSformerHighlightDataset(
    video_dir=r"I:\downloaded_videos",
    label_dir=r"../../dataset/most_csv",
    metadata_csv = metadata_csv,
    frame_count=frame_count
)

metadata = pd.read_csv(r"../../data/dataset/metadata.csv")
video_ids_all = metadata['video_id'].tolist()

# 2. 테스트 영상 ID 로드
test_list = pd.read_csv("test_video_list.csv")  # 컬럼명은 'video_id'여야 함
test_ids = test_list['video_id'].tolist()

# 3. test_indices 생성
test_indices = [i for i, vid in enumerate(video_ids_all) if vid in test_ids]

# 4. train_indices는 나머지
all_indices = list(range(len(video_ids_all)))
train_indices = list(set(all_indices) - set(test_indices))

# 5. Subset으로 분할
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Test set size: {len(test_dataset)}")

model = TimeSformer(
    image_size=224,
    patch_size=16,
    num_frames=24,
    num_classes=24
)
model.head = nn.Linear(model.head.in_features, frame_count)
model = model.to(device)

model_path = "../../data/timesformer_highlight_final.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print(f"✅ Loaded model from {model_path}")
else:
    raise FileNotFoundError("❌ Model checkpoint not found!")

model.eval()

all_outputs = []
all_targets = []

with torch.no_grad():
    for videos, targets in tqdm(test_loader):
        videos, targets = videos.to(device), targets.to(device)
        outputs = model(videos)

        all_outputs.append(outputs.cpu())
        all_targets.append(targets.cpu())

if len(all_outputs) == 0:
    raise ValueError("❌ No data in test_loader. Check if your dataset has videos with both mp4 and CSV labels.")

all_outputs = torch.cat(all_outputs, dim=0).detach().numpy()
all_targets = torch.cat(all_targets, dim=0).detach().numpy()

mse = mean_squared_error(all_targets, all_outputs)
rmse = np.sqrt(mse)
mae = mean_absolute_error(all_targets, all_outputs)
r2 = r2_score(all_targets, all_outputs)

print(f"\n[Test Accuracy Metrics]")
print(f"MSE   : {mse:.4f}")
print(f"RMSE  : {rmse:.4f}")
print(f"MAE   : {mae:.4f}")
print(f"R²    : {r2:.4f}")
