import os
import json
import torch
import sys
import gc
import re

os.environ['VIDEO_MAX_PIXELS'] = '200704'
os.environ['FPS_MAX_FRAMES'] = '16'
os.environ['MAX_PIXELS'] = '100352'
os.environ['ENABLE_AUDIO_OUTPUT'] = '0'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def run_consultant_evaluation(
    model_path, 
    output_jsonl, 
    plugin_path, 
    scene_desc, 
    char_persona, 
    char_id, 
    text_content, 
    audio_path_current, 
    audio_path_previous, 
    video_path, 
    is_redo=False
):
    from swift.llm import PtEngine, InferArguments

    args = InferArguments(
        model=model_path,
        temperature=0, 
        max_new_tokens=2048,
        model_type='qwen2_5_omni',
        template='qwen2_5_omni',
        system='You are a Senior Dubbing Producer. You provide objective, physics-based feedback.',
        external_plugins=[plugin_path] 
    )

    print("Loading SQ-LLM Engine")
    engine = PtEngine(args.model, args=args, model_kwargs={'torch_dtype': torch.bfloat16, 'device_map': 'cuda:0'})

    print(f"Starting single audit | Character: {char_id}")

    audio_path_current = os.path.abspath(audio_path_current)
    audio_path_previous = os.path.abspath(audio_path_previous) if audio_path_previous else None
    video_path = os.path.abspath(video_path)

    if not os.path.exists(audio_path_current) or not os.path.exists(video_path):
        print(f"Error: Audio file {audio_path_current} or video file {video_path} not found")
        return

    comp_report = None  
    eval_report = None
    sqi_advice = None
    history_msgs = []

    if is_redo and audio_path_previous and os.path.exists(audio_path_previous):
        print(f"Executing comparison mode: v1({os.path.basename(audio_path_previous)}) vs v2")
        prompt_sqc = (
                f"### [Scene Context]\n{scene_desc}\n\n"
                f"### [Character Persona: {char_id}]\n{char_persona}\n\n"
                f"### [Audit Target]\n"
                f"- Character: {char_id}\n"
                f"- Script: \"{text_content}\"\n\n"
                "Compare the speech quality of Sample A and Sample B. Please provide a comprehensive assessment, closely integrating the provided Scene context, specific character backgrounds, and the corresponding dialogue video."
        )
        
        infer_data_sqc = {
            "messages": 
                [
                    {
                        "role": "user", "content": 
                        [   
                            {"type": "audio", "audio": audio_path_previous},
                            {"type": "audio", "audio": audio_path_current},
                            {"type": "video", "video": video_path},
                            {"type": "text", "text": prompt_sqc}
                        ]
                    }
                ]
        }
        try:
            resp_sqc = engine.infer([infer_data_sqc])[0]
            comp_report = resp_sqc.choices[0].message.content
            answer_match = re.search(r'<answer>(.*?)</answer>', comp_report, re.DOTALL)
            if answer_match:
                report = answer_match.group(1).strip()
            else:
                report = re.sub(r'<think>.*?</think>', '', comp_report, flags=re.DOTALL).strip()

            history_msgs = [{"role": "assistant", "content": report}]
            print("SQC conclusion extraction completed")
        except Exception as e:
            print(f"SQC failed: {e}")

    reference_instruction = ""
    if is_redo:
        reference_instruction = "based on the previous Comparison Report "

    prompt_sqa = (
        f"### [Scene Context]\n{scene_desc}\n\n"
        f"### [Character Persona: {char_id}]\n{char_persona}\n\n"
        f"### [Audit Target]\n"
        f"- Character: {char_id}\n"
        f"- Script: \"{text_content}\"\n\n"
        f"Please evaluate the overall quality of this speech,closely integrating the provided Scene context{reference_instruction}, specific character backgrounds, and the corresponding dialogue video."
        "In your <think> block, assign a 1-10 score for each: Overall Quality, Intelligibility, Distortion, Speech Rate, Dynamic Range, Emotional Impact, Expression, Subjective Experience.\n\n"
    )

    infer_data_sqa = {
        "messages": history_msgs + [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "audio", "audio": audio_path_current},
                    {"type": "text", "text": prompt_sqa}
                ]
            }
        ]
    }

    try:
        resp_sqa = engine.infer([infer_data_sqa])[0]
        eval_report = resp_sqa.choices[0].message.content
        answer_match = re.search(r'<answer>(.*?)</answer>', eval_report, re.DOTALL)
        if answer_match:
            e_report = answer_match.group(1).strip()
        else:
            e_report = re.sub(r'<think>.*?</think>', '', eval_report, flags=re.DOTALL).strip()
        
        history_msgs.append({"role": "assistant", "content": e_report})
        print("SQA completed")
    except Exception as e:
        print(f"SQA failed: {e}")
        return

    prompt_sqi = (
        "Based on the evaluation report above, please suggest specific aspects for improvement to enhance the overall quality of this speech."
    )
    infer_data_sqi = {
        "messages": history_msgs + [
            {
                "role": "user", 
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "audio", "audio": audio_path_current},
                    {"type": "text", "text": prompt_sqi} 
                ]
            }
        ]
    }
    try:
        resp_sqi = engine.infer([infer_data_sqi])[0]
        sqi_advice = resp_sqi.choices[0].message.content
        print("SQI completed")
    except Exception as e:
        sqi_advice = f"SQI Error: {e}"

    result = {
        "char_id": char_id,
        "text": text_content,
        "sqc_report": comp_report,
        "sqa_report": eval_report,
        "sqi_suggestions": sqi_advice,
        "audio_current": audio_path_current,
        "audio_previous": audio_path_previous
    }
    with open(output_jsonl, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

    torch.cuda.empty_cache()
    gc.collect()
    print(f"Audit results saved to {output_jsonl}")

