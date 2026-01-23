import os
import json
import time
import base64
from openai import OpenAI

class DirectorAgent:
    def __init__(self, api_key=None):
        self.client = OpenAI(
            api_key=api_key or os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1",
        )
        self.model = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        self.session_history = [] 

    def _get_system_prompt(self):
        return (
            "You are a professional voice acting director performing Stage 1 labeling.\n"
            "Your mission is to generate a 'Director Arc'—a high-resolution emotional and acoustic "
            "fingerprint synthesized from AUDIO, VIDEO, and the SCENE CONTEXT.\n\n"

            "CORE HIERARCHY OF ANALYSIS:\n"
            "1. ATMOSPHERIC ANCHOR: Use the SCENE CONTEXT to set the acoustic "
            "baseline. The environment dictates the natural volume, reverb, and 'mood'.\n"
            "2. PERSONA FILTER: Apply the character's established personality. Every emotion must be "
            "filtered through their unique character soul (e.g., inherent insecurity vs. buoyant energy).\n"
            "3. FACIAL MICRO-EXPRESSIONS: Prioritize the VIDEO frames for subtext. Analyze eyes, "
            "brow tension, and mouth movements to detect if the character is masking true feelings.\n"
            "4. ACOUSTIC REALITY & OCCLUSION HANDLING: Verify if the AUDIO matches visual evidence. "
            "If the face is invisible (e.g., off-screen or back-shot), increase the weight of SCENE CONTEXT "
            "and character PERSONA. Ensure the label remains consistent with the character's soul.\n\n"

            "STRICT RULES:\n"
            "1. ALWAYS output in ENGLISH.\n"
            "2. FORMAT: '[Acoustic Feature], [Energy/Intensity], and [Emotional State].'\n"
            "3. LEXICAL PRECISION: Avoid generic labels like 'Happy' or 'Sad'. Use evocative adjectives like "
            "'Saturated', 'Brittle', 'Gravelly', or 'Exuberant'.\n"
            "4. SUBTEXT SENSITIVITY: If a character claims to be calm while physical actions suggest "
            "distress, the label should reflect 'defensive denial' rather than 'neutrality'.\n"
            "5. DYNAMIC TRANSITION: Capture shifts in performance (e.g., 'Starting with denial, then crumbling').\n\n"

            "CO-THOUGHT PROCESS (Internal Reasoning):\n"
            "- MOTIVATION: Given the context, why is this character choosing this specific tone now?\n"
            "- PHYSICALITY & FACIAL SYNC: How does the character's physical state color their vocal output?\n"
            "- CHARACTER LOYALTY: Is this performance consistent with their established persona?\n\n"

            "EXAMPLES OF HIGH-DISTINCTION LABELS:\n"
            " - 'Thin, breath-constrained, and saturated with sheepish, stress-induced regret.'\n"
            " - 'Resonant, steady, and radiating a quiet, mystical authority.'\n"
            " - 'High-pitched, rapid-fire, and sparkling with a frantic enthusiasm.'\n"
        )

    def reset_scene(self):
        self.session_history = []

    def _file_to_base64(self, file_path):
        if not (file_path and os.path.exists(file_path)):
            return None
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate_arc_from_clip(self, script_text, timestamp_info, char_id, char_persona, utt_idx, if_face=True, scene_desc=None, video_path=None, audio_path=None):
        face_status = "VISIBLE" if if_face else "INVISIBLE"
        history_text = f"CHARACTER FACE STATUS: {face_status}\nSpeaker: {char_id} (Line #{utt_idx})\nText: {script_text}\nTime: {timestamp_info}"
        
        api_text = (
            f"SCENE CONTEXT: {scene_desc}\n"
            f"CHARACTER PERSONA: {char_persona}\n\n"
            f"{history_text}"
        )

        current_multi_modal_content = []
        v_b64 = self._file_to_base64(video_path)
        if v_b64:
            current_multi_modal_content.append({
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{v_b64}", "detail": "high", "max_frames": 16, "fps": 1}
            })
        a_b64 = self._file_to_base64(audio_path)
        if a_b64:
            ext = os.path.splitext(audio_path)[1].replace('.', '') or "wav"
            current_multi_modal_content.append({"type": "audio_url", "audio_url": {"url": f"data:audio/{ext};base64,{a_b64}"}})
        
        current_multi_modal_content.append({"type": "text", "text": api_text})

        messages = [{"role": "system", "content": self._get_system_prompt()}]
        messages.extend(self.session_history)
        messages.append({"role": "user", "content": current_multi_modal_content})

        try:
            completion = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.5)
            arc_output = completion.choices[0].message.content.strip()
            self.session_history.append({"role": "user", "content": history_text})
            self.session_history.append({"role": "assistant", "content": arc_output})
            if len(self.session_history) > 10: self.session_history = self.session_history[-10:]
            return arc_output
        except Exception as e:
            raise e

def run_stage1_labeling_inplace(json_path, target_scene_ids=None, prefix_path="", api_key=None):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    director = DirectorAgent(api_key=api_key)
    
    for scene in data['scenes']:
        scene_id = scene['scene_id']
        if target_scene_ids and scene_id not in target_scene_ids: continue
        
        char_persona_map = {c['char_id']: c.get('persona', "Unknown persona") for c in scene.get('characters', [])}
        scene_desc = scene.get('scene_description', "")
        director.reset_scene()
        print(f"\n>>> scene: {scene_id}")
        
        for i, utt in enumerate(scene['utterances']):
            if (utt.get('director_arc') and utt['director_arc'] not in ["FAILED"]) or not utt['text']:
                continue

            current_persona = char_persona_map.get(utt['char_id'], "A general character in this scene.")
            v_clip = os.path.join(prefix_path, utt['video_path'])
            a_clip = os.path.join(prefix_path, utt['audio_path'])
            
            print(f"  Line #{utt['utterance_index']} | {utt['char_id']} | if_face: {utt['if_face']} | Persona: {current_persona[:30]}...")

            try:
                arc = director.generate_arc_from_clip(
                    script_text=utt['text'],
                    timestamp_info=f"{utt['start_time']}s-{utt['end_time']}s",
                    char_id=utt['char_id'],
                    char_persona=current_persona,
                    utt_idx=utt['utterance_index'],
                    if_face=utt['if_face'],
                    scene_desc=scene_desc, 
                    video_path=v_clip,
                    audio_path=a_clip
                )
                utt['director_arc'] = arc
                print(f"  [Director Arc]: {arc}")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"  [Error]:{e}")
                utt['director_arc'] = "FAILED"
            time.sleep(0.5)

if __name__ == "__main__":
    
    MY_API_KEY = ""

    run_stage1_labeling_inplace(
        json_path='', 
        target_scene_ids=[""], 
        api_key=MY_API_KEY
    )