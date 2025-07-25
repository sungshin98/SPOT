import os
from moviepy.editor import VideoFileClip

video_folder = r"I:\downloaded_videos"
audio_folder = os.path.abspath("/extract_audio")  # 절대 경로로 변환
os.makedirs(audio_folder, exist_ok=True)

# 실패한 파일 기록용 리스트
failed_files = []

def extract_audio_from_video(video_path, audio_path):
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        print(f"✅ 추출 완료: {audio_path}")
    except Exception as e:
        print(f"❌ {os.path.basename(video_path)} 실패: {e}")
        failed_files.append(os.path.basename(video_path))

# 영상 순회하며 오디오 추출
for file in os.listdir(video_folder):
    if file.lower().endswith(('.mp4', '.mov', '.mkv')):
        video_path = os.path.abspath(os.path.join(video_folder, file))
        base_name = os.path.splitext(file)[0]
        audio_path = os.path.abspath(os.path.join(audio_folder, base_name + '.mp3'))

        print(f"🎬 입력: {video_path}")
        print(f"🎧 출력: {audio_path}")
        extract_audio_from_video(video_path, audio_path)

# 실패한 파일 목록 저장
if failed_files:
    failed_txt_path = os.path.join(audio_folder, "failed_audio_list.txt")
    with open(failed_txt_path, "w", encoding="utf-8") as f:
        for name in failed_files:
            f.write(name + "\n")
    print(f"\n📄 실패한 파일 목록 저장 완료: {failed_txt_path}")
else:
    print("\n🎉 모든 파일에서 오디오 추출 성공!")
