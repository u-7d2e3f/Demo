import re
import os
import shutil
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# --- 配置 ---
LOG_FILE_PATH = "output/stage2/train.log"
# 统一保存到此根目录
BASE_LOG_DIR = "output/tensorboard"
# 定义具体的运行子目录，这样 TensorBoard 才能正确分类显示
RUN_DIR = os.path.join(BASE_LOG_DIR, "reconstructed_with_time")

# 你确认的：1 轮迭代 3200 次
STEPS_PER_EPOCH = 3200 

# 严格对齐 train_second.py 的 TensorBoard 路径映射
TAG_MAPPING = {
    "Loss": "train/mel_loss",
    "Gen Loss": "train/gen_loss",
    "Disc Loss": "train/d_loss",
    "Dur Loss": "train/dur_loss",
    "CE Loss": "train/ce_loss",
    "LM Loss": "train/slm_loss",
    "Norm Loss": "train/norm_loss",
    "F0 Loss": "train/F0_loss",
    "Sty Loss": "train/sty_loss",
    "Diff Loss": "train/diff_loss"
}

def reconstruct():
    if not os.path.exists(LOG_FILE_PATH):
        print(f"❌ 找不到日志: {LOG_FILE_PATH}")
        return

    # 环境准备：清理旧数据
    if os.path.exists(RUN_DIR):
        shutil.rmtree(RUN_DIR)
    os.makedirs(RUN_DIR, exist_ok=True)

    # 初始化 Writer
    writer = SummaryWriter(log_dir=RUN_DIR)
    
    # 核心正则：同时匹配时间戳、Epoch 和 Step
    # 示例行: INFO:2026-01-03 13:38:30,080: Epoch [1/3000], Step [100], Loss: ...
    main_pattern = re.compile(r"INFO:(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}):\s+Epoch\s+\[(\d+)/.*?Step\s+\[(\d+)\]")
    metric_pattern = re.compile(r"(\b[\w\s]*?Loss):\s+([\d.]+)")

    matched_rows = 0
    total_scalars = 0

    print(f"🚀 正在还原数据到 {RUN_DIR}，正在注入原始时间戳...")

    with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if "Epoch" not in line or "Step" not in line:
                continue

            # 1. 提取元数据（时间、轮次、步数）
            main_match = main_pattern.search(line)
            if main_match:
                time_str = main_match.group(1)      # 2026-01-03 13:38:30,080
                epoch = int(main_match.group(2))
                step_in_epoch = int(main_match.group(3))

                # 2. 将字符串时间转为 Unix 时间戳 (秒)
                # %f 对应日志中的毫秒部分
                dt_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S,%f")
                wall_time = dt_obj.timestamp()

                # 3. 计算全局连续步数 (Global Step)
                global_step = (epoch - 1) * STEPS_PER_EPOCH + step_in_epoch
                
                # 4. 提取该行内所有的 Loss 项
                metrics = metric_pattern.findall(line)
                
                row_has_data = False
                for label, value in metrics:
                    label = label.strip()
                    tag = TAG_MAPPING.get(label)
                    if tag:
                        # 核心修改：通过 walltime 强制指定该数据点的物理时间
                        writer.add_scalar(tag, float(value), global_step, walltime=wall_time)
                        total_scalars += 1
                        row_has_data = True
                
                if row_has_data:
                    matched_rows += 1

    # 强制刷盘并释放文件
    writer.flush()
    writer.close()
    
    print("-" * 40)
    print(f"✨ 还原完成！")
    print(f"✅ 成功提取真实时间坐标点: {matched_rows} 个节点")
    print(f"🔢 写入总计标量数值: {total_scalars} 个")
    print(f"📂 存储路径: {os.path.abspath(RUN_DIR)}")
    print(f"👉 启动指令: tensorboard --logdir={BASE_LOG_DIR} --port=6012")

if __name__ == "__main__":
    reconstruct()