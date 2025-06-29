import os
import time
import json
import shutil
import subprocess
import pandas as pd

# 기본 경로 설정
METADATA_FILE = "../../data/dataset/metadata.csv"
PRIMARY_DIR = "I:/downloaded_videos"
SECONDARY_DIR = "D:/downloaded_videos"
FAILED_LOG = "download_failed.txt"

# 디렉토리 생성
os.makedirs(PRIMARY_DIR, exist_ok=True)
os.makedirs(SECONDARY_DIR, exist_ok=True)

# 💾 메타데이터 로드
df_meta = pd.read_csv(METADATA_FILE)
df_meta = df_meta.dropna(subset=["youtube_id"])  # 결측 제거
df_meta = df_meta.set_index("video_id")

# 유효한 video_id 목록
video_ids = df_meta.index.tolist()

# 디스크 용량 체크
def has_enough_space(path, required_bytes=300 * 1024 * 1024):
    try:
        total, used, free = shutil.disk_usage(path)
        return free > required_bytes
    except:
        return False

# 유효한 영상 체크
def is_valid_video_file(path, min_size_kb=100):
    return os.path.exists(path) and os.path.getsize(path) > min_size_kb * 1024

# 실패 기록
def log_failure(vid, reason):
    with open(FAILED_LOG, "a", encoding="utf-8") as f:
        f.write(f"{vid}: {reason}\n")

# 다운로드 함수
def download_video(youtube_id, out_path, vid):
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    print(f"[INFO] Downloading {url} → {out_path}")
    try:
        subprocess.run([
            "yt-dlp", "-f", "mp4", "-o", out_path, url
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        reason = f"yt-dlp error: {e}"
        log_failure(vid, reason)
        print(f"[ERROR] {vid} failed: {reason}")
        return False

# 기존에 이미 존재하는 video_id 목록
def get_existing_video_ids():
    all_ids = set()
    for folder in [PRIMARY_DIR, SECONDARY_DIR]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.endswith(".mp4"):
                    if is_valid_video_file(os.path.join(folder, file)):
                        all_ids.add(os.path.splitext(file)[0])
    return all_ids

existing_ids = get_existing_video_ids()

# 다운로드 실행
for vid in video_ids:
    if vid in existing_ids:
        print(f"[SKIP] {vid} already exists.")
        continue

    print(f"\n[▶] Processing {vid}...")

    try:
        youtube_id = df_meta.loc[vid]["youtube_id"]
    except KeyError:
        log_failure(vid, "video_id not in metadata")
        print(f"[WARNING] {vid} not found in metadata")
        continue

    out_filename = f"{vid}.mp4"
    out_path = os.path.join(PRIMARY_DIR, out_filename)
    fallback_path = os.path.join(SECONDARY_DIR, out_filename)

    time.sleep(2)  # 너무 빠르게 요청하지 않도록 delay

    if has_enough_space(PRIMARY_DIR):
        success = download_video(youtube_id, out_path, vid)
        if not success and has_enough_space(SECONDARY_DIR):
            download_video(youtube_id, fallback_path, vid)
    elif has_enough_space(SECONDARY_DIR):
        download_video(youtube_id, fallback_path, vid)
    else:
        log_failure(vid, "Insufficient disk space")
        print(f"[ERROR] Not enough disk space for {vid}")
