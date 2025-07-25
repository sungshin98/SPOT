import os
import pandas as pd
import requests
import json
import subprocess
import time

# 경로 설정
video_dir = r"I:\downloaded_videos"
metadata_csv = "../dataset/metadata.csv"
output_dir = "/most_replayed"
error_dir = "/nothavemostreplayed"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(error_dir, exist_ok=True)

# 파일 이름 리스트 (확장자 제외)
video_files = [
    os.path.splitext(filename)[0]
    for filename in os.listdir(video_dir)
    if os.path.isfile(os.path.join(video_dir, filename))
]

# 메타데이터 불러오기 및 필터링
metadata = pd.read_csv(metadata_csv)
filtered = metadata[metadata['video_id'].isin(video_files)]
youtube_ids = filtered['youtube_id'].tolist()

# 실행 루프
for youtube_id in youtube_ids:
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    wrapped_json_path = f"{youtube_id}_page_wrapped.json"

    try:
        # HTML 요청
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        html = response.text

        # HTML 저장 (JSON 형식으로 래핑)
        with open(wrapped_json_path, "w", encoding="utf-8") as f:
            json.dump({"html": html}, f, indent=2, ensure_ascii=False)

        # 파싱 실행
        subprocess.run(["python", "parsing.py", wrapped_json_path], check=True)

        # 결과 파일 확인 및 이동
        result_json = f"{youtube_id}_macroMarkersListEntity.json"
        if os.path.exists(result_json):
            os.replace(result_json, os.path.join(output_dir, result_json))

        # 임시 파일 제거
        if os.path.exists(wrapped_json_path):
            os.remove(wrapped_json_path)
        if os.path.exists(result_json):  # 이미 복사되었으므로 삭제
            os.remove(result_json)

        print(f"✅ 처리 완료: {youtube_id}")


    except Exception as e:

        print(f"❌ 오류 발생 ({youtube_id}): {e}")

        # 에러 발생 시 결과 파일이 있으면 에러 디렉토리로 이동

        if os.path.exists(result_json):
            os.replace(result_json, os.path.join(error_dir, result_json))


    finally:

        # 임시 파일 제거

        if os.path.exists(wrapped_json_path):
            os.remove(wrapped_json_path)

        if os.path.exists(result_json):  # 이미 이동되었을 수도 있으니 재확인 후 삭제

            try:

                os.remove(result_json)

            except Exception:

                pass

        # 5초 대기

    time.sleep(5)