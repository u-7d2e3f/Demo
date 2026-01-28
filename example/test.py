import librosa
import librosa.display
import numpy as np
import torch
import matplotlib.pyplot as plt

def compare_mel_spectrograms(wav_path1, wav_path2):
    # 1. 设定参数 (来源于你的 preprocess.yaml)
    sr = 22050
    n_fft = 1024
    hop_length = 220
    win_length = 880
    n_mels = 80
    fmin = 0
    fmax = 8000

    # 2. 加载音频
    wav1, _ = librosa.load(wav_path1, sr=sr)
    wav2, _ = librosa.load(wav_path2, sr=sr)

    # 3. 提取 Mel 频谱
    # 注意：为了公平比较，通常取对数分贝值
    S1 = librosa.feature.melspectrogram(y=wav1, sr=sr, n_fft=n_fft, hop_length=hop_length, 
                                        win_length=win_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    S2 = librosa.feature.melspectrogram(y=wav2, sr=sr, n_fft=n_fft, hop_length=hop_length, 
                                        win_length=win_length, n_mels=n_mels, fmin=fmin, fmax=fmax)

    # 转换为分贝 (Log Scale)
    log_S1 = librosa.power_to_db(S1, ref=np.max)
    log_S2 = librosa.power_to_db(S2, ref=np.max)

    # 4. 长度对齐 (防止推理长度略微不同导致的报错)
    min_len = min(log_S1.shape[1], log_S2.shape[1])
    log_S1 = log_S1[:, :min_len]
    log_S2 = log_S2[:, :min_len]

    # 5. 计算差异
    mse = np.mean((log_S1 - log_S2) ** 2)
    mae = np.mean(np.abs(log_S1 - log_S2))

    print(f"--- 比较结果 ---")
    print(f"音频1: {wav_path1}")
    print(f"音频2: {wav_path2}")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")

    if mse < 1e-5:
        print("结论：这两个音频在 Mel 频谱上几乎完全一致。")
    elif mse < 5.0:
        print("结论：两个音频非常相似，存在微小预测偏差。")
    else:
        print("结论：两个音频存在明显差异。")

    # 6. 可视化差异
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    librosa.display.specshow(log_S1, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.title('Mel Spectrogram (Reconstructed/GT)')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(3, 1, 2)
    librosa.display.specshow(log_S2, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.title('Mel Spectrogram (Generated/Pred)')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(3, 1, 3)
    librosa.display.specshow(log_S1 - log_S2, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.title('Difference (S1 - S2)')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("example/mel_comparison.png")
    print("对比图已保存至 mel_comparison.png")


# --- 修改这里的路径后运行 ---
wav_pred = "example/ours/OZootopia004@Judy_00_Zootopia004_053.wav"
wav_rec = "example/truth/Zootopia004@Judy_00_Zootopia004_053.wav"
compare_mel_spectrograms(wav_rec, wav_pred)