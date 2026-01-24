import os
import requests
import subprocess
import shutil
from audio_separator.separator import Separator

LOG_FILE = "processed_scenes.txt"

def dummy_load_model_data(self, model_path):
    return {
        "model_name": "UVR-MDX-NET-Voc_FT",
        "model_filename": "UVR-MDX-NET-Voc_FT.onnx",
        "model_type": "mdx",
        "mdx_dim_f_set": 3072,
        "mdx_dim_t_set": 8,
        "mdx_n_fft_scale_set": 6144,
        "mdx_hop_set": 1024,
        "mdx_window_size_set": 512,
        "primary_stem": "Vocals",
        "compensate": 1.035,
        "mdx_architecture": "MDXNET"
    }

Separator.load_model_data_using_hash = dummy_load_model_data
requests.get = lambda *args, **kwargs: None
os.environ["HF_HUB_OFFLINE"] = "1"

def load_processed_list():
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def mark_as_processed(video_path):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(video_path + "\n")

def process_and_overwrite_video(video_path):
    vocal_tmp_dir = "temp_vocals"
    os.makedirs(vocal_tmp_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_final_video = video_path + ".tmp.mp4"
    temp_input_wav = os.path.join(vocal_tmp_dir, f"{base_name}_in.wav")

    subprocess.run([
        'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
        '-ar', '44100', '-ac', '2', temp_input_wav, '-y'
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "audio-separator-models")
    separator = Separator(output_dir=vocal_tmp_dir, output_format="WAV", model_file_dir=model_dir)
    separator.load_model(model_filename='UVR-MDX-NET-Voc_FT.onnx')
    
    output_files = separator.separate(temp_input_wav)
    
    vocal_wav = ""
    all_temp_outputs = [os.path.join(vocal_tmp_dir, f) for f in output_files]
    
    for f in output_files:
        if "Vocals" in f:
            vocal_wav = os.path.join(vocal_tmp_dir, f)
            break

    if vocal_wav and os.path.exists(vocal_wav):
        subprocess.run([
            'ffmpeg', '-i', video_path, '-i', vocal_wav,
            '-map', '0:v:0', '-map', '1:a:0', 
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', 
            temp_final_video, '-y'
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        os.replace(temp_final_video, video_path)
        mark_as_processed(video_path)

    for file_to_del in all_temp_outputs:
        if os.path.exists(file_to_del):
            os.remove(file_to_del)
            
    if os.path.exists(temp_input_wav): 
        os.remove(temp_input_wav)

if __name__ == "__main__":

    scenes_dir = ""
    vocal_tmp_dir = "temp_vocals"
    
    processed_files = load_processed_list()
    
    if os.path.exists(scenes_dir):
        video_files = sorted([f for f in os.listdir(scenes_dir) if f.endswith(".mp4")])
        
        for filename in video_files:
            video_path = os.path.join(scenes_dir, filename)
            
            if video_path in processed_files:
                print(f"Skipping (Already processed): {filename}")
                continue
            
            print(f"Processing: {filename}...")
            try:
                process_and_overwrite_video(video_path)
            except Exception as e:
                print(f"Error: {e}")
        
        if os.path.exists(vocal_tmp_dir):
            try:
                if not os.listdir(vocal_tmp_dir):
                    os.rmdir(vocal_tmp_dir)
                    print(f"\n Successfully cleaned up empty directory: {vocal_tmp_dir}")
                else:
                    print(f"\n Warning: {vocal_tmp_dir} is not empty, skipping cleanup.")
            except Exception as e:
                print(f"\n Cleanup failed: {e}")
    else:
        print(f"Directory not found: {scenes_dir}")