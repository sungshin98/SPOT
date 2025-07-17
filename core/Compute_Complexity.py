import pandas as pd
import cv2
import numpy as np
import os

# ---------- 함수 1: 영상 이름 읽기 ----------
def read_video_names(csv_path, column_name="video_id"):
    """
    CSV 파일에서 영상 이름 리스트 가져오기
    """
    df = pd.read_csv(csv_path)
    video_names = df[column_name].tolist()
    return video_names

# ---------- 함수 2: 영상 복잡도 계산 ----------
def calculate_video_complexity(video_path):
    """
    영상의 복잡도(프레임 밝기 표준편차) 계산
    """
    if not os.path.exists(video_path):
        print(f"❌ 파일 없음: {video_path}")
        return np.nan

    cap = cv2.VideoCapture(video_path)
    frame_std_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_std = np.std(gray)
        frame_std_list.append(frame_std)

    cap.release()
    complexity = np.std(frame_std_list) if frame_std_list else np.nan
    return complexity

# ---------- 함수 3: 복잡도 계산 후 CSV 저장 ----------
def save_complexities_to_csv(video_names, video_dir, output_csv):
    """
    모든 영상 복잡도 계산 후 CSV로 저장
    """
    results = []

    for name in video_names:
        video_file = os.path.join(video_dir, f"{name}.mp4")
        complexity = calculate_video_complexity(video_file)
        results.append({"video_name": name, "complexity": complexity})
        print(f"📦 {name}: 복잡도={complexity}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    print(f"✅ 복잡도 결과 저장 완료: {output_csv}")

# ---------- 메인 ----------
if __name__ == "__main__":
    # 상대경로 기준
    spot_csv = os.path.join("spot", "test", "test_video_list.csv")
    timesformer_csv = os.path.join("timesformer", "test", "test_video_list.csv")
    video_dir = 'I:\downloaded_videos'
    spot_output_csv = os.path.join("spot", "test", "video_complexities.csv")
    timesformer_output_csv = os.path.join("timesformer", "test", "video_complexities.csv")

    # SPOT 처리
    spot_videos = read_video_names(spot_csv, column_name="video_id")
    save_complexities_to_csv(
        spot_videos,
        video_dir,
        spot_output_csv
    )

    # Timesformer 처리
    timesformer_videos = read_video_names(timesformer_csv, column_name="video_id")
    save_complexities_to_csv(
        timesformer_videos,
        video_dir,
        timesformer_output_csv
    )
