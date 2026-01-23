import json
import os
from faster_whisper import WhisperModel

def update_scenes_with_vad_targeted(json_path, scenes_dir, target_scene_ids=None):
    if not os.path.exists(json_path):
        print(f"Error: JSON path {json_path} not found.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if target_scene_ids is None:
        video_files = sorted([f for f in os.listdir(scenes_dir) if f.endswith(".mp4")])
        target_scene_ids = [os.path.splitext(f)[0] for f in video_files]
    
    print(f"Targeting {len(target_scene_ids)} scenes: {target_scene_ids}")

    local_model_path = "faster-whisper-large-v3"
    model = WhisperModel(local_model_path, device="cuda", compute_type="float16")

    for scene_id in target_scene_ids:
        video_path = os.path.join(scenes_dir, f"{scene_id}.mp4")
        
        if not os.path.exists(video_path):
            print(f" Warning: Video file for {scene_id} not found at {video_path}, skipping.")
            continue

        scene_obj = next((s for s in data.get("scenes", []) if s.get("scene_id") == scene_id), None)
        if not scene_obj:
            print(f" Warning: Scene ID '{scene_id}' not found in JSON, skipping.")
            continue

        print(f"\n--- Precisely Transcribing: {scene_id} ---")
        segments, info = model.transcribe(
            video_path, 
            beam_size=5,
            language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        new_utterances = []
        for i, segment in enumerate(segments):
            start = round(segment.start, 2)
            end = round(segment.end, 2)
            text = segment.text.strip()
            
            new_utterances.append({
                "utterance_index": i + 1,
                "char_id": "Unknown",
                "audio_path": "",
                "video_path": "",
                "start_time": start,
                "end_time": end,
                "text": text,
                "emotion_label": "",
                "vad": {"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
                "director_arc": "",
                "if_face": True
            })
            print(f"  [{start}s - {end}s] {text}")

        scene_obj["utterances"] = new_utterances
        print(f" {scene_id} update successful: {len(new_utterances)} lines.")

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f" {scene_id} saved to disk.")
    
    print("\n Targeted batch transcription complete.")

if __name__ == "__main__":
    target_ids = [
        "", ""
    ]
    
    batch_update_scenes_with_vad_targeted(
        json_path = "", 
        scenes_dir = "",
        target_scene_ids = target_ids
    )