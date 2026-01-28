import os
from pydub import AudioSegment

def trim_to_align_ends_inplace(file1_path, file2_path):
    # 1. 加载音频
    audio1 = AudioSegment.from_file(file1_path)
    audio2 = AudioSegment.from_file(file2_path)
    
    # 获取原文件的后缀名 (例如 '.mp3', '.wav')，去掉前面的点
    ext1 = os.path.splitext(file1_path)[1].lstrip('.')
    ext2 = os.path.splitext(file2_path)[1].lstrip('.')
    
    len1 = len(audio1)
    len2 = len(audio2)
    
    print(f"音频1 ({file1_path}): {len1}ms")
    print(f"音频2 ({file2_path}): {len2}ms")

    # 2. 剪切逻辑
    if len1 > len2:
        diff = len1 - len2
        audio1 = audio1[diff:] 
        print(f"-> 音频1较长，已剪掉开头 {diff}ms")
    elif len2 > len1:
        diff = len2 - len1
        audio2 = audio2[diff:]
        print(f"-> 音频2较长，已剪掉开头 {diff}ms")
    else:
        print("-> 时长已相等，无需剪切")
        return # 如果相等直接返回，不重新写文件

    # 3. 写回原位置 (覆盖保存)
    audio1.export(file1_path, format=ext1)
    audio2.export(file2_path, format=ext2)
    
    print(f"处理完成！文件已覆盖。最终时长均为: {len(audio1)}ms")

# 使用示例
trim_to_align_ends_inplace("example/StyleDubber/DragonII@Hiccup.wav", "example/truth/DragonII@Hiccup.wav")