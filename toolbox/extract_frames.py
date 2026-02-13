"""
图片转频谱音频工具 - 线性频率版本
"""

import numpy as np
import cv2
from scipy.io import wavfile
from pathlib import Path


def image_to_audio(image_path: str, output_path: str,
                   sample_rate: int = 192000,
                   n_fft: int = 4096,
                   hop_length: int = 512):
    """
    将图片转换为频谱对应的音频（线性频率轴）

    图片坐标：y=0 (顶部) -> Nyquist, y=height-1 (底部) -> 0 Hz
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    n_freq_bins = n_fft // 2 + 1
    n_frames = width

    # 翻转图片（底部=低频，顶部=高频），缩放到 FFT bin 数
    gray_flipped = np.flipud(gray)
    magnitude = cv2.resize(gray_flipped, (n_frames, n_freq_bins),
                           interpolation=cv2.INTER_LINEAR).astype(np.float64) / 255.0

    # Griffin-Lim
    np.random.seed(42)
    phase = np.random.uniform(-np.pi, np.pi, (n_frames, n_freq_bins))
    window = np.hanning(n_fft)

    for it in range(60):
        stft = magnitude.T * np.exp(1j * phase)

        # ISTFT
        output_length = (n_frames - 1) * hop_length + n_fft
        audio = np.zeros(output_length)
        window_sum = np.zeros(output_length)
        for j in range(n_frames):
            start = j * hop_length
            frame = np.fft.irfft(stft[j], n_fft)
            audio[start:start + n_fft] += frame * window
            window_sum[start:start + n_fft] += window ** 2
        audio = audio / np.maximum(window_sum, 1e-8)

        # STFT
        stft_new = np.zeros((n_frames, n_freq_bins), dtype=np.complex128)
        for j in range(n_frames):
            start = j * hop_length
            if start + n_fft <= len(audio):
                frame = audio[start:start + n_fft] * window
            else:
                frame = np.zeros(n_fft)
                valid = len(audio) - start
                if valid > 0:
                    frame[:valid] = audio[start:] * window[:valid]
            stft_new[j] = np.fft.rfft(frame)

        phase = np.angle(stft_new)

    # 最终音频
    stft = magnitude.T * np.exp(1j * phase)
    output_length = (n_frames - 1) * hop_length + n_fft
    audio = np.zeros(output_length)
    window_sum = np.zeros(output_length)
    for j in range(n_frames):
        start = j * hop_length
        frame = np.fft.irfft(stft[j], n_fft)
        audio[start:start + n_fft] += frame * window
        window_sum[start:start + n_fft] += window ** 2
    audio = audio / np.maximum(window_sum, 1e-8)

    # 归一化并保存
    audio = audio / np.max(np.abs(audio)) * 0.9
    wavfile.write(output_path, sample_rate, (audio * 32767).astype(np.int16))


def process_task(args):
    image_to_audio(args[0], args[1])


if __name__ == "__main__":
    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor
    import os

    project_root = Path(__file__).parent.parent
    input_dir = project_root / "analyzed_image"
    output_dir = project_root / "output_audio"
    output_dir.mkdir(exist_ok=True)

    images = sorted(input_dir.glob("*.png"))
    print(f"找到 {len(images)} 张图片")

    # 准备任务列表
    tasks = [(str(img), str(output_dir / f"{img.stem}.wav")) for img in images]

    # 使用 12 个进程（保留 4 核给系统）
    num_workers = min(12, os.cpu_count() - 8)
    print(f"使用 {num_workers} 个进程并行处理")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(process_task, tasks), total=len(tasks), desc="转换进度"))
