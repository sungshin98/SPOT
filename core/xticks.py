import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드
df = pd.read_excel("video_complexity_distribution.xlsx")

# 한글 폰트 설정 (윈도우에서 Malgun Gothic 또는 NanumGothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 복잡도 구간 나누기
bins = np.histogram_bin_edges(df["complexity_score"], bins=10)
df["complexity_bin"] = pd.cut(df["complexity_score"], bins=bins, include_lowest=True)

# 각 bin별 평균 MSE 계산
bin_mse_summary = df.groupby("complexity_bin")[["spot_mse", "timesformer_mse"]].mean().reset_index()

# 시각화를 위한 라벨 처리
bin_labels = [f"{interval.left:.3f} ~ {interval.right:.3f}" for interval in bin_mse_summary["complexity_bin"]]
x = np.arange(len(bin_labels))

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(x, bin_mse_summary["spot_mse"], marker='o', label="SPOT 평균 MSE", color='blue')
plt.plot(x, bin_mse_summary["timesformer_mse"], marker='o', label="Timesformer 평균 MSE", color='green')

plt.xticks(x, bin_labels, rotation=45, ha='right')
plt.xlabel("복잡도 구간")
plt.ylabel("평균 MSE")
plt.title("복잡도 구간별 모델 성능 비교")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
