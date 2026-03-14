[![Project Page](https://img.shields.io/badge/Project-Webpage-2ecc71?style=flat-square&logo=googlechrome&logoColor=white)](https://dar.dub-demopage.workers.dev)

# Dataset

We provide a sample subset of the DAR-Animation Dataset. The complete dataset (~120 hours, 100 movies), along with the high-dimensional feature vectors required for training, will be fully open-sourced upon the formal publication of our work.

<details>
<summary><b>🎬 Click to view the complete list of 100 movies (120 Hours)</b></summary>
1. *The Boss Baby*
2. *The Boss Baby 2*
3. *Brave*
4. *Cloudy with a Chance of Meatballs*
5. *Cloudy with a Chance of Meatballs 2*
6. *Coco*
7. *The Croods*
8. *The Croods 2*
9. *How to Train Your Dragon*
10. *How to Train Your Dragon 2*
11. *How to Train Your Dragon 3*
12. *Frozen*
13. *Frozen II*
14. *The Incredibles*
15. *Incredibles 2*
16. *Inside Out*
17. *Inside Out 2*
18. *Meet the Robinsons*
19. *Moana*
20. *Wreck-It Ralph*
21. *Ralph Breaks the Internet*
22. *Tangled*
23. *Tinker Bell*
24. *Tinker Bell and the Lost Treasure*
25. *Tinker Bell and the Great Fairy Rescue*
26. *Toy Story*
27. *Toy Story 2*
28. *Toy Story 3*
29. *Toy Story 4*
30. *Up*
31. *Zootopia*
32. *Abominable*
33. *Sing*
34. *Sing 2*
35. *Despicable Me*
36. *Despicable Me 2*
37. *Despicable Me 3*
38. *Despicable Me 4*
39. *The Secret Life of Pets*
40. *The Secret Life of Pets 2*
41. *Antz*
42. *Bee Movie*
43. *Big Hero 6*
44. *Bolt*
45. *A Bug's Life*
46. *Captain Underpants: The First Epic Movie*
47. *Cars*
48. *Cars 2*
49. *Cars 3*
50. *Chicken Little*
51. *Elemental*
52. *Encanto*
53. *Finding Dory*
54. *Finding Nemo*
55. *Flushed Away*
56. *Home*
57. *Kung Fu Panda*
58. *Kung Fu Panda 2*
59. *Kung Fu Panda 3*
60. *Kung Fu Panda 4*
61. *Lightyear*
62. *Luca*
63. *Madagascar*
64. *Madagascar: Escape 2 Africa*
65. *Madagascar 3: Europe's Most Wanted*
66. *Penguins of Madagascar*
67. *Megamind*
68. *Monsters Inc*
69. *Monsters University*
70. *Monsters vs. Aliens*
71. *Mr. Peabody & Sherman*
72. *Onward*
73. *Over the Hedge*
74. *Puss in Boots*
75. *Puss in Boots: The Last Wish*
76. *Ratatouille*
77. *Raya and the Last Dragon*
78. *Rise of the Guardians*
79. *Shark Tale*
80. *Shrek*
81. *Shrek 2*
82. *Shrek the Third*
83. *Shrek Forever After*
84. *Soul*
85. *The Bad Guys*
86. *The Good Dinosaur*
87. *Trolls*
88. *Trolls World Tour*
89. *Turbo*
90. *Turning Red*
91. *Minions*
92. *Minions: The Rise of Gru*
93. *The Mitchells vs. the Machines*
94. *Wish*
95. *Strange World*
96. *Migration*
97. *Spies in Disguise*
98. *Ferdinand*
99. *Epic*
100. *Smallfoot*
</details>



```
cd Dataset/data/
dataset.json

cd movies/example
example_dataset.json
```

## 🛠️ Data Preprocessing

### 1) Auto-detect and Insert New Scenes 

Run the automated script to detect the number of existing scenes in `XXX_dataset.json` and automatically generate a template for the next scene.

- `python auto_insert_scene.py`

### 2) Scene Video Extraction 

Extract complete scene clips from the original movie based on the timestamps defined in the JSON.

- `python scene.py`

### 3) Audio Denoising and Background Music Removal 

To eliminate the interference of movie background music and sound effects on model training, we use the **UVR-MDX-NET-Voc_FT** model to process the raw materials.

- `python preprocess.py`

### 4) Text Annotation and Timestamp Acquisition 

We provide the following two flexible options for obtaining the dialogue and exact timestamps corresponding to the video:

1. For materials with officially proofread subtitles, SRT files can be parsed directly.
2. For materials without SRT files, or for custom data construction, we provide an alternative method:
   Use **Faster-Whisper-large-v3** for high-precision transcription and enable Voice Activity Detection to ensure timestamps exclude silent parts.

- `python preprocess_txt.py`

### 5) Manual Correction of Characters and Faces 

You need to manually open `XXX_dataset.json` and verify the following information against the video footage:

- **Character ID**: Change "Unknown" to the actual character.
- **Face Detection**: Confirm whether the character's face is in the frame during that line of dialogue.
- **Scene Description**: Background description of the scene in the clip, as well as a description of the characters in the scene.
- **Character Voice Timbre**: Manually insert the character's voice timbre for the scene.

### 6) Audio and Video Stabilization Slicing 

Run `preprocess_wav.py` to physically cut the audio and video for each annotated line of dialogue, generating the final training units.

- `python preprocess_wav.py`

### 7) Emotion VAD Feature Tagging 

Use the **MERaLiON-SER** model to assist in annotating the Valence/Arousal/Dominance values of the audio.

- `python preprocess_VAD.py`

### 8) Director Arc Extraction 

For the training set, use the **Qwen3-Omni** multimodal large model to comprehensively analyze the video footage, audio track, and contextual plot of the entire scene to generate standardized guidance instructions.

- `python preprocess_arc.py`

### ⚠️⚠️⚠️ Be sure to manually verify the correctness of the data annotations after completion.

### 📁 Directory Structure Reference

```
movies/
├── XXX/                  # Movie dataset directory
│   ├── XXX_dataset.json  # Core annotation file
│   ├── Scenes/           # Complete scene clips
│   ├── wavs/             # Final audio segments (24kHz)
│   └── video/            # Final video segments (25fps)
└── preprocess_*.py       # Automated processing scripts
```


## 🛠️ Offline Feature Extraction

To improve training efficiency and reduce VRAM usage, we adopt an **Offline Extraction** strategy. Before training begins, all multimodal materials will be mapped into high-dimensional feature vectors and saved in `.npy` format.

### 1) Multimodal Affective Encoding

Use large language models and encoders to extract deep description features of the dialogue semantics and video frames.

- **Models**: **Emotion-RoBERTa-Large** & **VideoLLaMA3**
- **Extracted Content**:
  - **Textual Sentiment** & **Director Guidance**: Use Emotion-RoBERTa-Large to extract semantic-level emotional representations of the dialogue text and director's instructions.
  - **Environment Atmosphere**: Use VideoLLaMA3 to analyze the environmental atmosphere in the video, generate a description, and map it into an emotion vector.
  - **Facial Affect**: Use VideoLLaMA3 to deeply analyze the character's facial micro-expressions, generate a description, and map it into an emotion vector.
-  `python Text_emos.py`

### 2) Speaker Timbre and Acoustic Emotion 

Extract the character's inherent timbre features and the emotional expressiveness vectors in the speech.

- **Models**: **CAMPPlus** & **Wav2Vec2-Bert** & **UnifiedVoice**
- **Extracted Content**:
  - **Speaker Timbre**: Extract speaker timbre embeddings based on the CAMPPlus module inside IndexTTS2.
  - **Acoustic Sentiment**: Use Wav2Vec2-Bert 2.0 to extract the deep semantic hidden states of the audio, and then input them into the UnifiedVoice module of IndexTTS2 to extract acoustic emotion vectors.
-  `python Timbre.py`

### 3) Frame-Level Visual Feature Extraction

Extract frame-level lip movements and facial dimensional emotions through pixel-level tracking technology.

- **Models**: **SAM 2.1** & **S3FD** &  **EmoNet** & **ResNet-18** 
-  **Key Technology**: Use **SAM 2.1** for full-video pixel-level isolation to ensure features are extracted only from the target character, eliminating background interference.

| Input Video |  | Output Video |
| :---: | :---: | :---: |
| ![](index/demo.gif) | **SAM2** <br> ➔ | ![](index/demooutput.gif) |

- **Extracted Content**:
  - **Lip Embedding**: Lip movement features based on S3FD + lrw_resnet18_mstcn_video.
  - **EmoVA Facial Affect**: Facial dimensional emotion vectors based on S3FD + EmoNet.
- `python EmoVA_Lipreading.py`

### 📁 Directory Structure Reference
```
preprocessed_data/
├── features/                
│   ├── VA_features/             # EmoVA Facial Affect Vectors
│   ├── arc/                     # Director Guidance Vectors
│   ├── extrated_embedding_gray/ # Lip Embeddings Vectors
│   ├── face/                    # Facial Affect Vectors
│   ├── scene/                   # Environment Atmosphere Vectors
│   ├── text/                    # Textual Sentiment Vectors
│   ├── emotion/                 # Acoustic Sentiment Vectors
│   └── timbre/                  # Speaker Timbre Vectors
└── wavs/                        # Speech files
```


# Dependencies

```
pip3 install -r requirements.txt
```

# Training

We only need to train the two stages of Actor：

**Macro-Contextual Level (EmotionGateformer）** 

```
python EmotionGateformer/train.py --config EmotionGateformer/Configs/Config.yml
```

 **Micro-Performance Level (Dubber)**:  Adopts ProDubber's two-stage training architecture

```
python ProDubber/train_first.py -p Configs/config_stage1.yml
python ProDubber/train_second.py -p Configs/config.yml
```

---

## ⚖️ Disclaimer

* **Research Purpose Only**: This dataset is provided for non-commercial, academic research purposes only. 

* **Copyright**: All original movie materials (visuals, audio, and characters) are the intellectual property of their respective studios (Disney, Pixar, DreamWorks, etc.). The authors do not own the raw multimedia content.

* **Compliance**: Users are responsible for complying with local copyright laws when using this dataset. 

  
