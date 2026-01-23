import json
import os
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForAudioClassification
from tqdm import tqdm

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

current_dir = os.path.dirname(os.path.abspath(__file__))
repo = os.path.join(current_dir, "MERaLiON-SER-v1") 
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading MERaLiON-SER model from: {repo} ...")

try:
    processor = AutoProcessor.from_pretrained(repo, local_files_only=True)
    model = AutoModelForAudioClassification.from_pretrained(
        repo, trust_remote_code=True, local_files_only=True, device_map=None
    ).to(device)
    model.eval()
except Exception as e:
    print(f"Offline loading failed: {e}")
    exit()

emo_map = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgusted", "Surprised"]

def analyze_audio_vad(audio_path):
    try:
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)

        inputs = processor(
            wav.squeeze().numpy(), 
            sampling_rate=16000, 
            return_tensors="pt", 
            return_attention_mask=True
        )
        
        with torch.inference_mode():
            input_dict = {k: v.to(device) for k, v in inputs.items() if k in ("input_features", "attention_mask")}
            out = model(**input_dict)
        
        logits, dims = out["logits"], out["dims"]
        emo_idx = torch.argmax(logits, dim=1).item()
        vad_values = dims.squeeze().tolist() 

        return {
            "label": emo_map[emo_idx],
            "valence": round(vad_values[0], 4),
            "arousal": round(vad_values[1], 4),
            "dominance": round(vad_values[2], 4)
        }
    except Exception as e:
        print(f"Processing failed for {audio_path}: {e}")
        return None

def annotate_dataset_batch(json_path, target_scene_ids):
    if not os.path.exists(json_path):
        print(f"JSON not found: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Targeting VAD annotation for {len(target_scene_ids)} scenes...")

    all_utts = []
    for scene in data.get("scenes", []):
        sid = scene.get("scene_id")
        if sid in target_scene_ids:
            for utt in scene.get("utterances", []):
                if "audio_path" in utt and os.path.exists(utt["audio_path"]):
                    all_utts.append(utt)

    if not all_utts:
        print("No valid audio files found for the target scenes.")
        return

    print(f" Total utterances to analyze: {len(all_utts)}")

    for utt in tqdm(all_utts, desc="Analyzing Emotions"):
        res = analyze_audio_vad(utt["audio_path"])
        if res:
            utt["emotion_label"] = res["label"]
            utt["vad"]["valence"] = res["valence"]
            utt["vad"]["arousal"] = res["arousal"]
            utt["vad"]["dominance"] = res["dominance"]

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    print(f" All annotations for targeted scenes completed and saved.")

if __name__ == "__main__":
    my_target_ids = [
        "", ""
    ]
    
    annotate_dataset_batch(
        json_path="", 
        target_scene_ids=my_target_ids
    )