import torch
from torch.utils.data import DataLoader, Subset
import pandas as pd
from tqdm import tqdm
import json

from core.spot import SPOT
from dataset import TimeSformerHighlightDataset
from results import metadata, log_folder
import torchvision.transforms as transforms

# 경로 설정
video_dir = r"I:\downloaded_videos"
label_dir = log_folder
metadata_csv = metadata.metadata_csv
model_path = "../train/hybrid_checkpoint_13.pth"  # 학습 시 저장된 경로 사용

frame_count = 24
batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 로드
dataset = TimeSformerHighlightDataset(
    video_dir=video_dir,
    label_dir=label_dir,
    metadata_path=metadata_csv,
    frame_count=frame_count,
    image_size=(224, 224)
)

# test_video_list.csv 기준으로 test 인덱스 추출
test_list_df = pd.read_csv("test_video_list.csv")
test_video_ids = set(test_list_df["video_id"].tolist())
test_indices = [i for i, vid in enumerate(dataset.video_ids) if vid in test_video_ids]

test_dataset = Subset(dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델 로드
model = SPOT.SPOT(num_frames=frame_count, num_classes=frame_count).to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 예측 수행
results = []
with torch.no_grad():
    for i, (frames, targets) in enumerate(tqdm(test_loader, desc="Testing")):
        frames = frames.to(device)
        targets = targets.to(device)

        outputs = model(frames)  # [1, T]
        outputs = outputs.squeeze().cpu().numpy()
        targets = targets.squeeze().cpu().numpy()
        video_id = dataset.video_ids[test_indices[i]]

        results.append({
            "video_id": video_id,
            "predictions": outputs.tolist(),
            "targets": targets.tolist()
        })

# 결과 저장
with open("test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("✅ Test predictions saved to test_results.json")
