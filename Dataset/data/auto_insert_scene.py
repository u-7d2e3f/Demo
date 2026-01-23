import json
import os
import re

def insert_next_scenes_batch(json_path, num_to_insert=1):
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} not found")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    prefix = data.get("movie_id", "Scene")
    existing_scenes = data.get("scenes", [])
    
    max_num = 0
    pattern = re.compile(rf"^{prefix}(\d+)$")
    
    for scene in existing_scenes:
        sid = scene['scene_id']
        match = pattern.match(sid)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num

    print(f"Auto-detected prefix: '{prefix}' (Current Max: {max_num:03d})")
    
    for i in range(1, num_to_insert + 1):
        next_num = max_num + i
        new_scene_id = f"{prefix}{next_num:03d}"
        video_path = f"movies/{prefix}/Scenes/{new_scene_id}.mp4"

        new_scene = {
            "scene_id": new_scene_id,
            "scene_description": "",
            "video_path": video_path,
            "characters": [
                {
                    "char_id": "",
                    "persona": "",
                    "global_timbre_ref": ""
                }
            ],
            "utterances": [] 
        }

        data["scenes"].append(new_scene)
        print(f" -> Prepared template for: '{new_scene_id}'")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Successfully inserted {num_to_insert} new scenes.")

if __name__ == "__main__":
  
    insert_next_scenes_batch(
        json_path="", 
        num_to_insert=10
    )