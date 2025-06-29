import os
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import torchvision.io as io

from torch.utils.data import Dataset

class TimeSformerHighlightDataset(Dataset):
    def __init__(self, video_dir, label_dir, metadata_path, frame_count=24, image_size=(224, 224)):
        self.video_dir = video_dir
        self.label_dir = label_dir
        self.frame_count = frame_count
        self.image_size = image_size

        # 전체 video 목록 중 정답 있는 것만 필터링
        self.video_ids = []
        for file in os.listdir(video_dir):
            if file.endswith(".mp4"):
                video_id = file.replace(".mp4", "")
                label_path = os.path.join(label_dir, f"{video_id}_macroMarkersListEntity.csv")
                if os.path.exists(label_path):
                    self.video_ids.append(video_id)

        # 클래스 레이블 (선택사항)
        self.metadata = pd.read_csv(metadata_path).set_index("video_id")

        # 프레임 전처리
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ConvertImageDtype(torch.float32),  # 0~1
        ])

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]

        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        label_path = os.path.join(self.label_dir, f"{video_id}_macroMarkersListEntity.csv")

        # 1. 비디오 프레임 추출
        video, _, info = io.read_video(video_path, pts_unit='sec')
        total_frames = video.shape[0]

        # 프레임 샘플링 (equal spacing)
        if total_frames < self.frame_count:
            indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
            pad = self.frame_count - total_frames
            indices = np.pad(indices, (0, pad), mode='edge')
        else:
            indices = np.linspace(0, total_frames - 1, self.frame_count, dtype=int)

        frames = video[indices]  # [T, H, W, C]
        frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
        frames = torch.stack([self.transform(f) for f in frames])  # float32, normalized

        # 2. 정답 score 불러오기
        label_df = pd.read_csv(label_path)
        scores = label_df["intensityScoreNormalized"].values

        # 정답도 프레임 수에 맞춰 샘플링
        if len(scores) < self.frame_count:
            pad = self.frame_count - len(scores)
            scores = np.pad(scores, (0, pad), mode='edge')
        else:
            indices = np.linspace(0, len(scores) - 1, self.frame_count, dtype=int)
            scores = scores[indices]

        targets = torch.tensor(scores, dtype=torch.float32)  # [T]

        # (선택) 클래스 정보
        if video_id in self.metadata.index:
            class_labels = eval(self.metadata.loc[video_id, "labels"])
            # return frames, targets, class_labels
        else:
            class_labels = []

        return frames, targets  #, class_labels
