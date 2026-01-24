import subprocess
import os

def convert_to_seconds(time_val):
    if isinstance(time_val, (int, float)):
        return float(time_val)
    parts = str(time_val).strip().split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(time_val)

def fast_cut_video_gpu(input_file, output_file, start_input, end_input):
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    start = convert_to_seconds(start_input)
    end = convert_to_seconds(end_input)
    duration = end - start
    
    command = [
        'ffmpeg', '-y', '-hwaccel', 'cuda',
        '-ss', str(start),
        '-i', input_file,
        '-t', str(duration),
        '-map', '0:v:0', '-map', '0:a:0',
        '-map_metadata', '-1',
        '-avoid_negative_ts', 'make_zero',
        '-sn',
        '-vf', 'scale=-2:720,fps=25',
        '-c:v', 'h264_nvenc',
        '-pix_fmt', 'yuv420p',
        '-preset', 'p4',
        '-tune', 'hq',
        '-c:a', 'aac', '-ar', '22050', '-ac', '1',
        output_file
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"Success: {output_file} [{start_input} -> {end_input}]")
    except subprocess.CalledProcessError as e:
        print(f"Error: {output_file} | {e.stderr.decode()}")

if __name__ == "__main__":
    input_movie = ""
    output_folder = ""
    
    time_segments = [
        ("", "")
    ]
    
    prefix = ""
    start_num = 1
    
    for i, (start, end) in enumerate(time_segments):
        scene_id = f"{prefix}{start_num + i:03d}"
        output_path = os.path.join(output_folder, f"{scene_id}.mp4")
        fast_cut_video_gpu(input_movie, output_path, start, end)