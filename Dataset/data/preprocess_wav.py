import json
import os
import subprocess
from tqdm import tqdm

def extract_scenes_batch_and_sync_json(json_path, video_src_dir, wav_root, clip_root, target_scene_ids):
    if not os.path.exists(json_path):
        print(f" Error: JSON not found: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f" Starting batch extraction for {len(target_scene_ids)} scenes (Real-time Saving Enabled)...")

    total_tasks = []
    for scene in data.get("scenes", []):
        sid = scene["scene_id"]
        if sid not in target_scene_ids:
            continue
        
        source_video = os.path.join(video_src_dir, f"{sid}.mp4")
        if not os.path.exists(source_video):
            print(f" Skip {sid}: {source_video} not found")
            continue

        os.makedirs(os.path.join(wav_root, sid), exist_ok=True)
        os.makedirs(os.path.join(clip_root, sid), exist_ok=True)

        for utt in scene.get("utterances", []):
            total_tasks.append((source_video, sid, utt))

    if not total_tasks:
        print("No valid tasks found.")
        return

    success_count = 0
    for source_video, sid, utt in tqdm(total_tasks, desc="Extracting Clips"):
        idx = f"{utt['utterance_index']:03d}"
        char = utt.get("char_id", "Unknown").replace(" ", "_")
        
        base_name = f"{sid}@{char}_00_{sid}_{idx}"
        a_p = os.path.join(wav_root, sid, f"{base_name}.wav")
        v_p = os.path.join(clip_root, sid, f"{base_name}.mp4")

        start = str(utt["start_time"])
        dur = str(round(float(utt["end_time"]) - float(utt["start_time"]), 3))

        try:
            subprocess.run(["ffmpeg", "-y", "-i", source_video, "-ss", start, "-t", dur, "-vn", "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", a_p], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            subprocess.run(["ffmpeg", "-y", "-hwaccel", "cuda", "-i", source_video, "-ss", start, "-t", dur, "-an", "-vf", "scale=-2:720", "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", v_p], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            utt["audio_path"] = a_p
            utt["video_path"] = v_p
            success_count += 1

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            continue

    print(f"\n All done! Successfully saved {success_count} entries to JSON.")

if __name__ == "__main__":
    my_target_ids = [
        "", ""
    ]
    
    extract_scenes_batch_and_sync_json(
        json_path="",
        video_src_dir="",
        wav_root="",
        clip_root="",
        target_scene_ids=my_target_ids
    )