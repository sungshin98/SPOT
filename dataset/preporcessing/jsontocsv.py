import os
import json
import pandas as pd

most_rp_f = '../dataset/most_replayed/'
most_rp_list = os.listdir(most_rp_f)
most_csv = '../dataset/most_csv'
os.makedirs(most_csv, exist_ok=True)

for most_rp_j in most_rp_list:
    json_path = os.path.join(most_rp_f, most_rp_j)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # markers 추출 및 DataFrame 변환
    markers = data["markersList"]["markers"]
    df = pd.DataFrame(markers)

    # 파일 이름에서 확장자 제거하고 .csv 붙이기
    base_name = os.path.splitext(most_rp_j)[0]
    csv_output_path = os.path.join(most_csv, base_name + ".csv")

    df.to_csv(csv_output_path, index=False)

print('done')
