# parse_macro_markers.py

import sys
import os
import json
import json5

def extract_macroMarkers(json_html_path):
    # 입력 파일 로딩
    with open(json_html_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    html_text = data.get("html", "")

    # macroMarkersListEntity의 위치 찾기
    start_index = html_text.find('"macroMarkersListEntity"')
    if start_index == -1:
        raise ValueError("macroMarkersListEntity not found in file")

    brace_index = html_text.find('{', start_index)

    # JSON 블록 괄호 매칭
    depth = 0
    end_index = brace_index
    for i in range(brace_index, len(html_text)):
        if html_text[i] == '{':
            depth += 1
        elif html_text[i] == '}':
            depth -= 1
            if depth == 0:
                end_index = i
                break

    # 블록 파싱
    macro_block_str = html_text[brace_index:end_index + 1]
    macro_entity = json5.loads(macro_block_str)

    # 저장 경로 설정
    video_id = macro_entity.get("externalVideoId", "output")
    output_filename = f"{video_id}_macroMarkersListEntity.json"
    output_path = os.path.join(os.path.dirname(json_html_path), output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(macro_entity, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("❗ Usage: python parse_macro_markers.py <path_to_json_wrapped_html>")
        sys.exit(1)

    input_path = sys.argv[1]
    try:
        extract_macroMarkers(input_path)
    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)
