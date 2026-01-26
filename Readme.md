[![Project Page](https://img.shields.io/badge/Project-Webpage-2ecc71?style=flat-square&logo=googlechrome&logoColor=white)](https://u-7d2e3f.github.io/DemoPage)


# 数据集

去除场景中的非主要说话人的语句，比如周围的欢呼声，一些无关紧要的路人的声音等语句


## 🛠️ 数据预处理

### 第一步：自动检测并插入新场景 (Auto Scene Insertion)

运行自动化脚本检测 `XXX_dataset.json` 中已有的场景数量，并自动生成下一个场景的模板。

- **命令**：`python auto_insert_scene.py`

### 第二步：场景视频切片 (Scene Extraction)

根据 JSON 定义的时间戳，从电影原片中提取出完整的场景片段。

- **命令**：`python scene.py`

### 第三步：音频降噪与背景音去除 (Audio Denoising)

为了消除电影背景音乐和音效对模型训练的干扰，我们采用 **UVR-MDX-NET-Voc_FT** 模型对原始素材进行处理。

- **命令**：`python preprocess.py`

### 第四步：文本标注与时间戳获取 (Text Annotation & Alignment)

在获取视频对应的台词及其精确时间戳时，我们提供以下两种灵活的方案：

1.针对拥有官方校对字幕的素材，可以直接解析 SRT 文件。

2.对于无SRT文件的素材，或者自定义数据构建，我们提供另一种方法：

​     使用 **Faster-Whisper-large-v3** 进行高精度转写并开启 Voice Activity Detection以确保时间戳剔除静音部分。

- **命令**：`python preprocess_txt.py`

### 第五步：手动修正角色与面部 (Manual Annotation) 

需手动打开 `XXX_dataset.json`，根据视频画面核对以下信息：

- **角色 ID (`char_id`)**：将 "Unknown" 更改为实际角色。
- **面部检测 (`if_face`)**：确认该句对白时，角色的脸部是否在镜头内。
- **场景描述 (`scene_description`)**：该片段场景的背景描述，以及场景中的人物描述。
- **角色音色**：将场景中角色音色手动插入

### 第六步：音视频稳定化切片 (Final Segmenting)

运行 `preprocess_wav.py` 对标注好的每一句台词的音频和视频进行物理切割，生成最终的训练单元。

- **命令**：`python preprocess_wav.py`

### 第七步：情感 VAD 特征标注 (Emotion Tagging)

使用**MERaLiON-SER**模型辅助标注音频的 Valence/Arousal/Dominance 数值。

- **命令**：`python preprocess_VAD.py`

### 第八步：导演指令特征提取 (Director's Arc Extraction)

用 **Qwen3-Omni** 多模态大模型，综合分析整个场景的视频画面、音频轨道以及上下文剧情，生成标准化的指导指令

- **命令**：`python preprocess_arc.py`

### ⚠️⚠️⚠️非常重要， 务必在标注完成后人工检差数据标注的正确性



### 📁 目录结构参考

```
movies/
├── XXX/                  # 电影数据集文件夹
│   ├── XXX_dataset.json  # 核心标注文件
│   ├── Scenes/           # 场景完整片段
│   ├── wavs/             # 最终音频切片 (16kHz)
│   └── video/            # 最终视频切片 (25fps)
└── preprocess_*.py       # 自动化处理脚本
```



## 🛠️ 特征离线化提取

为了提高训练效率并降低显存占用，我们采用**离线化提取（Offline Extraction）**策略。在训练开始前，所有多模态素材将被映射为高维特征向量并保存为 `.npy` 格式。

### 1. 语义与视觉描述特征 (Semantic & Visual Descriptions)

利用大语言模型与编码器提取台词语义及画面的深度描述特征。

- **模型**: **Emotion-RoBERTa-Large** & **VideoLLaMA3**
- **提取内容**:
  - **Textual Sentiment**: 利用Emotion-RoBERTa-Large提取文本台词和导演指导在语义层面的情感表征。
  - **Environment Atmosphere**: 利用 VideoLLaMA3 分析视频中的环境氛围并生成描述，随后映射为情感向量。
  - **Facial Affect**: 利用 VideoLLaMA3 深度解析角色面部微表情并生成描述，随后映射为情感向量。
- **命令**: `python Text_emos.py`

### 2. 音频音色与声学情感 (Audio Timbre & Emotion)

提取角色的固有音色特征及语音中的情感表现力向量。

- **模型**: **CAMPPlus** & **Wav2Vec2-Bert** & **UnifiedVoice**
- **提取内容**:
  - **Timbre Vector**: 基于 IndexTTS2 内部的 CAMPPlus 模块，提取 说话人音色嵌入。
  - **Acoustic Sentiment**: 利用 Wav2Vec2-Bert 2.0 提取音频的深层语义隐藏状态（Hidden States），随后将其输入 IndexTTS2 的 UnifiedVoice 模块，提取声学情感向量
- **命令**: `python Timbre.py`

### 3. 视觉同步与维度情感 (Visual Sync & VA Features)

通过像素级追踪技术提取帧级的唇部运动与面部维度情感。

- **模型**: **SAM 2.1** & **S3FD** &  **EmoNet** & **ResNet-18** 
- **关键技术**: 使用 **SAM 2.1** 进行全视频像素级隔离，确保特征仅提取自目标角色，排除背景干扰。
- **提取内容**:
  - **Lip Embedding**: 基于 S3FD+ lrw_resnet18_mstcn_video 的唇部运动特征。
  - **VA Features**: 基于 S3FD + EmoNet 的面部维度情感向量。
- **命令**: `python EmoVA_Lipreading.py`

### 📁 目录结构参考
```
preprocessed_data/
├── features/                  # 特征向量
│   ├── VA_features/           # 帧级面部情感向量
│   ├── arc/                   # 导演指导情感向量
│   ├── extrated_embedding_gray/ # 唇部向量 
│   ├── face/     # 面部情感向量 
│   ├── scene/    # 氛围情感向量
│   ├── text/     # 文本情感向量
│   ├── emotion/  # 音频情感向量
│   └── timbre/   #音色
└──wavs/      # 语音
```

