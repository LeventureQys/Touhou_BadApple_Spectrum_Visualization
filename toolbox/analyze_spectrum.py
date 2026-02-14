import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取音频文件
sample_rate, data = wavfile.read('outputs/11_1.wav')
print(f"采样率: {sample_rate} Hz")
print(f"数据形状: {data.shape}")
print(f"时长: {len(data) / sample_rate:.2f} 秒")

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

# 时间轴（用于时域波形）
duration = len(data) / sample_rate
time_axis = np.linspace(0, duration, len(data))

# 绘制双子图：上方时域波形，下方频谱图
# 频谱图目标比例 900:560 = 1.6:1
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 6.6), height_ratios=[1, 6])

# 上方：时域波形
ax1.plot(time_axis, data, linewidth=0.3, color='#1f77b4')
ax1.set_ylabel('幅度')
ax1.set_title(f'11_1.wav (采样率: {sample_rate} Hz)')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(times[0], times[-1])
ax1.set_xticklabels([])

# 下方：时频谱图
mesh = ax2.pcolormesh(times, frequencies / 1000, Sxx_db, shading='gouraud', cmap='inferno')
ax2.set_xlabel('时间 (秒)')
ax2.set_ylabel('频率 (kHz)')
ax2.set_ylim(0, sample_rate / 2000)
ax2.set_xlim(times[0], times[-1])
ax2.set_aspect('auto')

fig.tight_layout()
fig.subplots_adjust(hspace=0.05)
plt.savefig('spectrum_11_1.png', dpi=150)
plt.show()
print("图像已保存为 spectrum_11_1.png")
