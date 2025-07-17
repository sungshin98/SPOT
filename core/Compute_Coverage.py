import pandas as pd
import numpy as np
import os
import json

# ---------- GT 마스크 생성 함수 ----------
def load_gt_highlight(json_file, frame_rate=25):
    """
    markersDecoration 기반으로 GT 마스크 생성
    """
    with open(json_file, "r", encoding="utf-8") as f:  # <<< 여기 추가
        data = json.load(f)
    decorations = data['markersList']['markersDecoration']['timedMarkerDecorations']

    total_duration = max(int(d['visibleTimeRangeEndMillis']) for d in decorations)
    total_frames = int((total_duration / 1000) * frame_rate)

    gt_mask = np.zeros(total_frames, dtype=int)
    for deco in decorations:
        start = int(int(deco['visibleTimeRangeStartMillis']) / 1000 * frame_rate)
        end = int(int(deco['visibleTimeRangeEndMillis']) / 1000 * frame_rate)
        gt_mask[start:end] = 1
    return gt_mask

# ---------- Coverage 계산 ----------
def calculate_coverage(predictions, gt_mask):
    """
    predictions 길이에 맞게 GT 마스크 리샘플링 후 Coverage 계산
    """
    pred_len = len(predictions)
    gt_len = len(gt_mask)

    if pred_len != gt_len:
        # 리샘플링: GT 마스크를 predictions 길이에 맞춤
        gt_mask_resized = np.round(np.interp(
            np.linspace(0, 1, pred_len),
            np.linspace(0, 1, gt_len),
            gt_mask
        )).astype(int)
    else:
        gt_mask_resized = gt_mask

    pred_mask = (np.array(predictions) >= np.mean(predictions)).astype(int)
    intersection = np.logical_and(pred_mask, gt_mask_resized).sum()
    gt_total = gt_mask_resized.sum()
    coverage = intersection / gt_total if gt_total != 0 else np.nan
    return coverage
# ---------- 복잡도 그룹화 ----------
def assign_complexity_group(x, low_th, high_th):
    if x < low_th:
        return "Low"
    elif x < high_th:
        return "Medium"
    else:
        return "High"

# ---------- 메인 함수 ----------
def compute_coverage_by_complexity(spot_results_json, timesformer_results_json, complexities_csv, markers_dir, output_csv, frame_rate=25):
    # 복잡도 데이터 로드
    complexities_df = pd.read_csv(complexities_csv)

    # 복잡도 그룹 경계 계산 (33%, 66% 백분위수)
    low_th = complexities_df["complexity"].quantile(0.33)
    high_th = complexities_df["complexity"].quantile(0.66)

    # SPOT / Timesformer 결과 로드
    with open(spot_results_json, "r") as f:
        spot_results = json.load(f)
    with open(timesformer_results_json, "r") as f:
        timesformer_results = json.load(f)

    # 결과 저장용
    rows = []

    for _, row in complexities_df.iterrows():
        video_name = row["video_name"]
        complexity = row["complexity"]
        group = assign_complexity_group(complexity, low_th, high_th)

        # GT 마스크 생성
        gt_json_path = os.path.join(markers_dir, f"{video_name}_macroMarkersListEntity.json")
        if not os.path.exists(gt_json_path):
            print(f"⚠️ GT JSON 없음: {gt_json_path}")
            continue
        gt_mask = load_gt_highlight(gt_json_path, frame_rate)

        # SPOT / Timesformer 예측 가져오기
        spot_pred = next((v["predictions"] for v in spot_results if v["video_id"] == video_name), None)
        timesformer_pred = next((v["predictions"] for v in timesformer_results if v["video_id"] == video_name), None)

        if spot_pred is None or timesformer_pred is None:
            print(f"⚠️ 예측 데이터 없음: {video_name}")
            continue

        # Coverage 계산
        spot_cov = calculate_coverage(spot_pred, gt_mask)
        timesformer_cov = calculate_coverage(timesformer_pred, gt_mask)

        rows.append({
            "video_name": video_name,
            "complexity": complexity,
            "complexity_group": group,
            "SPOT_Coverage": spot_cov,
            "Timesformer_Coverage": timesformer_cov
        })

    # 결과 DataFrame
    result_df = pd.DataFrame(rows)

    # 복잡도별 평균 Coverage
    group_stats = result_df.groupby("complexity_group")[["SPOT_Coverage", "Timesformer_Coverage"]].mean().reset_index()

    print("\n📊 복잡도별 평균 Coverage:")
    print(group_stats)

    return result_df, group_stats


if __name__ == "__main__":
    result_df, group_stats = compute_coverage_by_complexity(
        spot_results_json="spot/test/test_results.json",
        timesformer_results_json="timesformer/test/test_results.json",
        complexities_csv="./video_complexities.csv",
        markers_dir=r"F:\dataset\most_replayed",
        output_csv="./coverage_by_complexity.csv",
        frame_rate=25
    )

    # CSV 저장
    result_df.to_csv("./coverage_by_complexity.csv", index=False)
    print(f"✅ 결과 저장 완료: ./coverage_by_complexity.csv")
