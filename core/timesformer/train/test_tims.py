import torch
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

from Timesformer.model.timesformer import TimeSformer
from Timesformer.dataset import TimeSformerHighlightDataset

# ------------------------------
# 설정
video_dir = r"F:\downloaded_videos"
label_dir = r"../../dataset/most_csv"
metadata_path = r"../../data/dataset/metadata.csv"
frame_count = 24
batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "timesformer_highlight_final.pth"

# ------------------------------
# Dataset 불러오기
dataset = TimeSformerHighlightDataset(
    video_dir=video_dir,
    label_dir=label_dir,
    metadata_path=metadata_path,
    frame_count=frame_count
)

# test_video_list.csv에서 test set 인덱스 추출
test_list_df = pd.read_csv("../test/test_video_list.csv")
test_video_ids = set([vid.replace(".mp4", "") for vid in test_list_df["video_id"]])

test_indices = [i for i, vid in enumerate(dataset.video_ids) if vid in test_video_ids]

test_dataset = Subset(dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ------------------------------
# 모델 로드
model = TimeSformer(
    image_size=224,
    patch_size=16,
    num_frames=frame_count,
    num_classes=frame_count
)
model.head = torch.nn.Linear(model.head.in_features, frame_count)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ------------------------------
# 예측 및 결과 저장
results = []

with torch.no_grad():
    for i, (videos, targets) in enumerate(tqdm(test_loader, desc="Testing TimeSformer")):
        videos = videos.to(device)
        targets = targets.to(device)

        outputs = model(videos)

        pred = outputs.squeeze().cpu().numpy().tolist()
        target = targets.squeeze().cpu().numpy().tolist()
        video_id = dataset.video_ids[test_indices[i]]

        results.append({
            "video_id": video_id,
            "predictions": pred,
            "targets": target
        })

# ------------------------------
# 결과 저장
with open("../test/timesformer_test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("✅ TimeSformer 테스트 결과 저장 완료: timesformer_test_results.json")
