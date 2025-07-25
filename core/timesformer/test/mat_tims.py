import matplotlib.pyplot as plt
import json
import numpy as np

# 🔧 Windows 시스템 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 📂 예측 결과 파일 경로
with open("timesformer_test_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

# 🎬 첫 5개 영상만 시각화
num_videos = min(5, len(results))
fig, axes = plt.subplots(num_videos, 1, figsize=(10, 3 * num_videos), constrained_layout=True)

if num_videos == 1:
    axes = [axes]

for i in range(num_videos):
    video_id = results[i]["video_id"]
    preds = results[i]["predictions"]
    targets = results[i]["targets"]
    x = np.arange(len(preds))

    ax = axes[i]
    ax.plot(x, preds, label="예측값", marker='o')
    ax.plot(x, targets, label="정답값", marker='x')
    ax.set_title(f"영상: {video_id}", fontsize=12)
    ax.set_xlabel("프레임 인덱스")
    ax.set_ylabel("하이라이트 점수")
    ax.legend()
    ax.grid(True)

plt.suptitle("🎬 테스트셋 예측 vs 정답 (샘플)", fontsize=16)
plt.tight_layout()
plt.show()
