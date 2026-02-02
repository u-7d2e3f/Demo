import os
import json
import base64
from openai import OpenAI

class DirectorAgent:
    def __init__(self, api_key=None, model="Qwen/Qwen3-Omni-30B-A3B-Instruct"):
        self.client = OpenAI(
            api_key=api_key or os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1",
        )
        self.model = model
        self.session_history = [] 


    def reset_scene(self):
        self.session_history = []


    def _get_initial_prompt(self):
        return (
            "You are a professional voice acting director.\n"
            "Your mission is to generate a 'Director Arc'—a high-resolution fingerprint synthesized from VIDEO, SCENE CONTEXT, and character PERSONA.\n\n"

            "### 1. CORE HIERARCHY OF ANALYSIS:\n"
            "1. ATMOSPHERIC ANCHOR: Use the SCENE CONTEXT to set the acoustic baseline.\n"
            "2. PERSONA FILTER: Filter emotions through the character's unique soul.\n"
            "3. FACIAL MICRO-EXPRESSIONS: Analyze physical subtext from the VIDEO.\n"
            "4. OCCLUSION HANDLING: If invisible, increase the weight of PERSONA.\n\n"

            "### 2.CO-THOUGHT PROCESS (Internal_Reasoning):\n"
            "Follow these steps sequentially to build your reasoning in 'Internal_Reasoning':\n"
            "Step 1: INTENT & SUBTEXT - Analyze the Script and Persona to find the character's hidden objective. Why are they saying this NOW?\n"
            "Step 2: PHYSICALITY & ENVIRONMENT - Identify facial tension, breathing patterns, and spatial constraints (reverb, distance) from the Video and Scene.\n"
            "Step 3: ACOUSTIC MAPPING - Translate the Step 1 (psychology) and Step 2 (physics) into technical terms (e.g., tension -> 'Brittle', authority -> 'Resonant').\n"
            "Step 4: LOYALTY CHECK - Verify if the resulting Arc is loyal to the character's established soul and long-term goals.\n\n"

            "### 3. STRICT RULES:\n"
            "1. FORMAT: '[Acoustic Feature], [Energy/Intensity], and [Emotional State].'\n"
            "2. LEXICAL PRECISION: Use adjectives like 'Saturated', 'Brittle', 'Gravelly', 'Resonant', 'Thin'.\n"
            "3. EXAMPLES: \n"
            "   - 'Thin, breath-constrained, and saturated with sheepish, stress-induced regret.'\n"
            "   - 'Resonant, steady, and radiating a quiet, mystical authority.'\n\n"

            "### 4. OUTPUT FORMAT\n"
            "Return ONLY a JSON object:\n"
            "{\n"
            "  \"Internal_Reasoning\": \"Step 1: ... Step 2: ... Step 3: ... Step 4: ...\",\n"
            "  \"Director Arc\": \"The expert-level instruction label.\"\n"
            "}"
        )

    def _get_scoring_prompt(self):
        return (
            "You are a Senior Quality Auditor. Your mission is to provide a high-precision score "
            "for the SYNTHESIZED AUDIO based on available Expert Feedback (SQA, SQC, and SQI).\n\n"

            "### 1. SCORING CONSTRAINTS:\n"
            "1. RANGE: 0.0 to 5.0.\n"
            "2. PRECISION: Use decimals for granular evaluation (e.g., 3.2, 4.7).\n"
            "3. Your mission is to perform a comprehensive evaluation by synthesizing three expert sources: SQI, SQC, and SQA."

            "### 2. CO-THOUGHT PROCESS (Internal_Reasoning):\n"
            "Follow these steps sequentially in 'Internal_Reasoning':\n"
            "Step 1: FEEDBACK DECONSTRUCTION - Extract core requirements from 'sqi_suggestions' and 'sqa_report' for the current character.\n"
            "Step 2: AUDITORY EVIDENCE - Analyze the current audio. Does it resolve the specific tone, energy, or distortion issues mentioned?\n"
            "Step 3: RESOLUTION ASSESSMENT - If SQC is available, confirm if the current sample is superior to the previous one.\n"
            "Step 4: SCORE CALIBRATION - Determine the final decimal score (0.0-5.0) by mapping the audio's success against the expert's specific complaints.\n\n"

            "### 3. OUTPUT FORMAT\n"
            "Return ONLY a JSON object:\n"
            "{\n"
            "  \"Internal_Reasoning\": \"Step 1: ... Step 2: ... Step 3: ... Step 4: ...\",\n"
            "  \"Comprehensive_Score\": 0.0\n"
            "}"
        )

    def _get_revision_prompt(self):
        return (
            "You are a Technical Director specializing in corrective voice synthesis. Your mission is to "
            "FIX the failed performance by calculating the 'Acoustic Delta' between the Previous Direct "
            "and the Expert Feedback (SQA, SQC, SQI).\n\n"

            "### 1. REVISION LOGIC:\n"
            "1. FAILURE ANALYSIS: Identify if the failure was due to the 'Instruction being ignored' or 'insufficient'.\n"
            "2. COMPENSATION PRINCIPLE: If the previous audio was 'too flat' despite a 'Neutral' instruction, "
            "the new instruction must over-compensate (e.g., move to 'Highly Expressive').\n\n"

            "### 2. STRICT RULES (Instruction Format):\n"
            "1. FORMAT: '[Acoustic Feature], [Energy/Intensity], and [Emotional State].'\n"
            "2. LEXICAL PRECISION: Use adjectives like 'Saturated', 'Brittle', 'Gravelly', 'Resonant', 'Thin'.\n"
            "3. EXAMPLES: \n"
            "   - 'Thin, breath-constrained, and saturated with sheepish, stress-induced regret.'\n"
            "   - 'Resonant, steady, and radiating a quiet, mystical authority.'\n\n"

            "### 3. CO-THOUGHT PROCESS (Internal_Reasoning):\n"
            "Step 1: DISCREPANCY AUDIT - Why did the previous audio fail to achieve the target?\n"
            "Step 2: FEEDBACK INTEGRATION - Deconstruct SQI/SQA to find the specific acoustic gap.\n"
            "Step 3: DELTA CALCULATION - Calculate the required shift (More/Less energy, tension, etc.).\n"
            "Step 4: ARC RE-SYNTHESIS - Formulate the NEW 'Director Arc' incorporating the Delta.\n\n"

            "### 4. OUTPUT FORMAT\n"
            "Return ONLY a JSON object:\n"
            "{\n"
            "  \"Internal_Reasoning\": \"Step 1: ... Step 2: ... Step 3: ... Step 4: ...\",\n"
            "  \"Director Arc\": \"The refined expert-level instruction label.\"\n"
            "}"
        )

    def _file_to_base64(self, file_path):
        if not (file_path and os.path.exists(file_path)): return None
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def run_task(self, task_type, script_text, char_id, char_persona, utt_idx, timestamp_info,
                 if_face=True, scene_desc=None, video_path=None, audio_path=None, 
                 feedback=None, last_direct=None):
       
        prompts = {
            "INITIAL": self._get_initial_prompt(),
            "SCORING": self._get_scoring_prompt(),
            "REVISION": self._get_revision_prompt()
        }
        
  
        face_status = "VISIBLE" if if_face else "INVISIBLE"
        

        task_headers = {
            "INITIAL": "### TASK 1: INITIAL GUIDANCE  ###",
            "SCORING": f"### TASK 2: PERFORMANCE SCORING  ###\nExpert Feedback: {feedback}",
            "REVISION": f"### TASK 3: REVISION DIRECT (CORRECTIVE) ###\nPrevious Direct: {last_direct} \nExpert Feedback: {feedback}"
        }

     
        user_text = (
            f"SCENE: {scene_desc}\n"
            f"CHARACTER PERSONA: {char_persona}\n\n"
            f"{task_headers[task_type]}\n"
            f"Line #{utt_idx} | {char_id} | Face: {face_status} | Time: {timestamp_info}\n"
            f"Text: {script_text}"
        )

        content = [{"type": "text", "text": user_text}]
        

        v_b64 = self._file_to_base64(video_path)
        if v_b64:
            content.append({
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{v_b64}", "detail": "high", "max_frames": 2, "fps": 1}
            })
        
        a_b64 = self._file_to_base64(audio_path)
        if a_b64:
            ext = os.path.splitext(audio_path)[1].replace('.', '') or "wav"
            content.append({"type": "audio_url", "audio_url": {"url": f"data:audio/{ext};base64,{a_b64}"}})

        messages = [{"role": "system", "content": prompts[task_type]}]
        messages.extend(self.session_history)
        messages.append({"role": "user", "content": content})

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"}
            )
            res_json = json.loads(completion.choices[0].message.content)

            history_text = (
                f"CHARACTER FACE STATUS: {face_status}\n"
                f"Speaker: {char_id} (Line #{utt_idx})\n"
                f"Text: {script_text}\n"
                f"Time: {timestamp_info}"
            )
            
       
            self.session_history.append({"role": "user", "content": f"TASK_TYPE: {task_type}\n{history_text}"})
      
            self.session_history.append({"role": "assistant", "content": json.dumps(res_json)})

          
            if len(self.session_history) > 20: self.session_history = self.session_history[-20:]

            return res_json
        except Exception as e:
            return {"Error": str(e), "Direct": "FAILED"}