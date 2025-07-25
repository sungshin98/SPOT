import json
import subprocess
import pandas as pd
import os
import time
import shutil

# 경로 설정
SPLIT_FILE = "C:\\Users\king2\PycharmProjects\SPOT\MR_HiSum\dataset\mr_hisum_split.json"
METADATA_FILE = "metadata.csv"
PRIMARY_DIR = "I:/downloaded_videos"
SECONDARY_DIR = "D:/downloaded_videos"
FAILED_LOG = "download_failed.txt"

os.makedirs(PRIMARY_DIR, exist_ok=True)
os.makedirs(SECONDARY_DIR, exist_ok=True)

# split json 로드
with open(SPLIT_FILE, "r") as f:
    splits = json.load(f)
train_ids = splits["train_keys"]

# metadata 로드
df_meta = pd.read_csv(METADATA_FILE, sep='\t' if '\t' in open(METADATA_FILE).readline() else ',', engine='python')
df_meta = df_meta.set_index('video_id')

# 디스크 사용 가능 여부 확인 함수
def has_enough_space(path, required_bytes=300 * 1024 * 1024):
    try:
        total, used, free = shutil.disk_usage(path)
        return free > required_bytes
    except:
        return False

# 유효한 파일인지 확인 (용량 기준)
def is_valid_video_file(path, min_size_kb=100):
    return os.path.exists(path) and os.path.getsize(path) > min_size_kb * 1024

# 실패 기록 함수
def log_failure(vid, reason):
    with open(FAILED_LOG, "a", encoding="utf-8") as f:
        f.write(f"{vid}: {reason}\n")

# 다운로드 함수
def download_video(youtube_id, out_path, vid):
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    print(f"[INFO] Downloading {url} to {out_path}")
    try:
        subprocess.run([
            "yt-dlp", "-f", "mp4", "-o", out_path, url
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        reason = f"yt-dlp error - {str(e)}"
        log_failure(vid, reason)
        print(f"[ERROR] Failed to download {url}: {reason}")
        return False

# 이미 존재하는 video_id 확인
def get_existing_video_ids():
    all_ids = set()
    for folder in [PRIMARY_DIR, SECONDARY_DIR]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.endswith(".mp4"):
                    path = os.path.join(folder, file)
                    if is_valid_video_file(path):  # 유효성 검사
                        all_ids.add(os.path.splitext(file)[0])
    return all_ids

existing_ids = get_existing_video_ids()

# 영상 하나씩 처리
for vid in train_ids:
    if vid in existing_ids:
        print(f"[SKIP] {vid} already downloaded in I:/ or D:/")
        continue

    print(f"\n[INFO] Processing {vid}...")

    # YouTube ID 확인
    if vid not in df_meta.index:
        reason = "video_id not found in metadata"
        log_failure(vid, reason)
        print(f"[WARNING] {reason} for {vid}")
        continue
    youtube_id = df_meta.loc[vid]['youtube_id']

    # 저장 경로
    out_filename = f"{vid}.mp4"
    out_path = os.path.join(PRIMARY_DIR, out_filename)
    fallback_path = os.path.join(SECONDARY_DIR, out_filename)

    # 다운로드
    time.sleep(5)
    if has_enough_space(PRIMARY_DIR):
        success = download_video(youtube_id, out_path, vid)
        if not success and has_enough_space(SECONDARY_DIR):
            download_video(youtube_id, fallback_path, vid)
    elif has_enough_space(SECONDARY_DIR):
        download_video(youtube_id, fallback_path, vid)
    else:
        reason = "Not enough disk space"
        log_failure(vid, reason)
        print(f"[ERROR] {reason}. Skipping {vid}.")
