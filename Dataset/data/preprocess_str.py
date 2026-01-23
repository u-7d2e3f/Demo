import json
import re
import os

def srt_time_to_seconds(srt_time):
    hh, mm, ss_ms = srt_time.split(':')
    ss, ms = ss_ms.split(',')
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0

def convert_input_to_seconds(time_val):
    if isinstance(time_val, (int, float)): return float(time_val)
    parts = str(time_val).strip().split(':')
    if len(parts) == 3: return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2: return int(parts[0]) * 60 + float(parts[1])
    return float(time_val)

def batch_extract_srt_to_json(srt_path, json_path, scene_configs):
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    blocks = re.split(r'\n\n|\r\n\r\n', content.strip())
    parsed_blocks = []
    
    for block in blocks:
        lines = block.split('\n')
        if len(lines) < 2: continue
        time_line = next((l for l in lines if " --> " in l), None)
        if not time_line: continue
        time_match = re.search(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', time_line)
        if not time_match: continue
        text_lines = [l for l in lines if " --> " not in l and not l.isdigit()]
        parsed_blocks.append({
            "start": srt_time_to_seconds(time_match.group(1)),
            "end": srt_time_to_seconds(time_match.group(2)),
            "text": " ".join(text_lines).strip()
        })

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for scene_id, g_start, g_end in scene_configs:
        start_offset = convert_input_to_seconds(g_start)
        end_limit = convert_input_to_seconds(g_end)
        new_utterances = []
        u_idx = 1
        
        for block in parsed_blocks:
            if block["start"] >= start_offset and block["end"] <= end_limit:
                new_utterances.append({
                    "utterance_index": u_idx,
                    "char_id": "Unknown",
                    "audio_path": f"",
                    "video_path": f"",
                    "start_time": round(block["start"] - start_offset, 2),
                    "end_time": round(block["end"] - start_offset, 2),
                    "text": block["text"],
                    "emotion_label": "",
                    "vad": {"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
                    "director_arc": "",
                    "if_face": True
                })
                u_idx += 1
        
        scene_obj = next((s for s in data["scenes"] if s["scene_id"] == scene_id), None)
        if scene_obj:
            scene_obj["utterances"] = new_utterances
            print(f"{scene_id}: {len(new_utterances)} utterances updated.")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("\n Batch JSON update completed.")

if __name__ == "__main__":
    fixed_segments = [
        (1,  "",    ""), (2,  "",    "")
    ]
    
    scene_configs = [(f"XXX{idx:03d}", start, end) for idx, start, end in fixed_segments]
    
    batch_extract_srt_to_json(
        srt_path="", 
        json_path="", 
        scene_configs=scene_configs
    )