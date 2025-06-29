import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_excel("video_complexity_distribution.xlsx")

# 한글 폰트 설정 (원하는 경우 Windows 폰트 경로로 설정)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 복잡도 bin 설정
bins = np.histogram_bin_edges(df["complexity_score"], bins=10)
df["complexity_bin"] = pd.cut(df["complexity_score"], bins=bins, include_lowest=True)

# 각 bin에서 음수/양수 개수 세기
bin_summary = df.groupby("complexity_bin").apply(
    lambda g: pd.Series({
        "SPOT 우위 (음수)": (g["mse_diff"] < 0).sum(),
        "Timesformer 우위 (양수)": (g["mse_diff"] > 0).sum()
    })
)

# 시각화
bin_labels = [str(interval) for interval in bin_summary.index]
x = np.arange(len(bin_labels))
width = 0.6

plt.figure(figsize=(10, 6))
plt.bar(x, bin_summary["SPOT 우위 (음수)"], width, label="SPOT 우위 (음수)", color="skyblue")
plt.bar(x, bin_summary["Timesformer 우위 (양수)"], width,
        bottom=bin_summary["SPOT 우위 (음수)"], label="Timesformer 우위 (양수)", color="salmon")

plt.xticks(x, bin_labels, rotation=45, ha='right')
plt.xlabel("복잡도 구간")
plt.ylabel("영상 수")
plt.title("복잡도 구간별 음수/양수(MSE 차이) 분포")
plt.legend()
plt.tight_layout()
plt.grid(axis='y')
plt.show()
