import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，支持多进程
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.ndimage import zoom
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

# 设置字体
plt.rcParams['axes.unicode_minus'] = False

# 进程数
NUM_WORKERS = 4

# 输入输出目录
input_dir = Path('outputs')
output_dir = Path('output_spectrum')

# 频谱图最大尺寸（降采样目标）
MAX_FREQ_BINS = 512
MAX_TIME_BINS = 1000

def process_wav(wav_path_str):
    """处理单个 wav 文件"""
    wav_path = Path(wav_path_str)

    try:
        # 读取音频文件
        sample_rate, data = wavfile.read(wav_path)

        # 如果是立体声，取第一个声道
        if len(data.shape) > 1:
            data = data[:, 0]

        # 计算 spectrogram（与 image_to_audio 参数一致）
        nperseg = 4096
        hop_length = 512
        noverlap = nperseg - hop_length
        frequencies, times, Sxx = spectrogram(data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)

        # 转换为 dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # 对频谱图降采样以节省内存
        freq_ratio = min(1.0, MAX_FREQ_BINS / Sxx_db.shape[0])
        time_ratio = min(1.0, MAX_TIME_BINS / Sxx_db.shape[1])
        if freq_ratio < 1.0 or time_ratio < 1.0:
            Sxx_db = zoom(Sxx_db, (freq_ratio, time_ratio), order=1)
            frequencies = np.linspace(frequencies[0], frequencies[-1], Sxx_db.shape[0])
            times = np.linspace(times[0], times[-1], Sxx_db.shape[1])

        # 时间轴（用于时域波形）- 降采样以节省内存
        duration = len(data) / sample_rate
        downsample = max(1, len(data) // 5000)
        data_downsampled = data[::downsample]
        time_axis = np.linspace(0, duration, len(data_downsampled))

        # 释放原始数据
        del data
        gc.collect()

        # 绘制双子图：上方时域波形，下方频谱图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 6.6), height_ratios=[1, 6])

        # 上方：时域波形
        ax1.plot(time_axis, data_downsampled, linewidth=0.3, color='#1f77b4')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'{wav_path.name} (Sample Rate: {sample_rate} Hz)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(times[0], times[-1])
        ax1.set_xticklabels([])

        # 下方：时频谱图（使用 imshow 代替 pcolormesh，内存更省）
        ax2.imshow(Sxx_db, aspect='auto', origin='lower', cmap='inferno',
                   extent=[times[0], times[-1], frequencies[0]/1000, frequencies[-1]/1000])
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (kHz)')
        ax2.set_ylim(0, sample_rate / 2000)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05)

        # 保存图像
        out_dir = Path('output_spectrum')
        output_path = out_dir / f'{wav_path.stem}_spectrum.png'
        fig.savefig(output_path, dpi=100)  # 降低 dpi
        plt.close(fig)

        # 清理
        del Sxx_db, data_downsampled, time_axis, frequencies, times
        gc.collect()

        return True, wav_path.name

    except Exception as e:
        return False, f"{wav_path.name}: {e}"

if __name__ == '__main__':
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)

    # 获取所有 wav 文件
    wav_files = list(input_dir.glob('*.wav'))
    total = len(wav_files)
    print(f"找到 {total} 个音频文件，使用 {NUM_WORKERS} 进程处理")

    success_count = 0
    error_count = 0
    completed = 0

    # 转换为字符串列表
    wav_paths_str = [str(p) for p in wav_files]

    # maxtasksperchild=50 让进程定期重启，释放内存（需要用 multiprocessing.Pool）
    from multiprocessing import Pool

    with Pool(processes=NUM_WORKERS, maxtasksperchild=50) as pool:
        results = pool.imap_unordered(process_wav, wav_paths_str)

        for success, msg in results:
            completed += 1
            if success:
                success_count += 1
                print(f"[{completed}/{total}] 完成: {msg}")
            else:
                error_count += 1
                print(f"[{completed}/{total}] 错误: {msg}")

    print(f"\n完成! 成功: {success_count}, 失败: {error_count}")
    print(f"所有时频谱图已保存到 {output_dir}/")
