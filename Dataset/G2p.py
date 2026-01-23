import os
import json
from phonemizer import phonemize

def get_phonemes(text):
    return phonemize(
        text,
        language='en-us',
        backend='espeak',
        strip=True,
        preserve_punctuation=True,
        with_stress=True
    )

def generate_split_metadata(index_path, output_dir):
    if not os.path.exists(index_path):
        print(f"Error: Index file {index_path} not found")
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(index_path, 'r', encoding='utf-8') as f:
        index_data = json.load(f)

    split_results = {
        "train": [],
        "val": [],
        "test": []
    }

    for movie in index_data.get("movies", []):
        raw_path = movie.get("dataset_path")
        scene_json_path = os.path.join("data", raw_path)
        
        splits_map = {
            "train": movie.get("train", []),
            "val": movie.get("val", []),
            "test": movie.get("test", [])
        }

        if not os.path.exists(scene_json_path):
            print(f"Warning: Scene JSON {scene_json_path} not found")
            continue

        with open(scene_json_path, 'r', encoding='utf-8') as f:
            scene_data = json.load(f)

        for scene in scene_data.get("scenes", []):
            scene_id = scene["scene_id"]
            target_splits = [s_name for s_name, s_list in splits_map.items() if scene_id in s_list]
            
            if not target_splits:
                continue

            for utt in scene.get("utterances", []):
                if not utt.get("if_face", True):
                    continue

                char_id = utt["char_id"]
                raw_text = utt["text"]
                audio_path = utt["audio_path"]
                
                speaker_id = f"{scene_id}@{char_id}"
                filename = os.path.basename(audio_path)
                relative_path = f"{filename}"
                
                try:
                    ipa_phonemes = get_phonemes(raw_text)
                    line = f"{relative_path}|{ipa_phonemes}|{speaker_id}"
                    
                    for split in target_splits:
                        split_results[split].append(line)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    for split_name, lines in split_results.items():
        output_path = os.path.join(output_dir, f"{split_name}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')
        print(f"Success: {len(lines)} lines written to {output_path}")

if __name__ == "__main__":
    INDEX_FILE = "data/dataset.json"
    OUTPUT_DIR = "preprocessed_data/features"
    generate_split_metadata(INDEX_FILE, OUTPUT_DIR)