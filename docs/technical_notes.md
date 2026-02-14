# Bad Apple 频谱可视化 - 技术笔记

## 项目目标

把 Bad Apple 的视频画面转换成音频频谱图。播放生成的音频时，用频谱分析器查看，能看到原视频的画面。

## 处理流程

```
视频 -> 逐帧图片 -> 频谱音频 -> 频谱图片 -> 合成视频
         ↓            ↓           ↓           ↓
   extract_frames  image_to_audio  batch_spectrum  make_video
```

### 1. 视频拆帧

`toolbox/extract_frames.py`

把视频按帧拆成 PNG 图片。文件名格式是 `{秒}_{帧序号}.png`，比如 `5_12.png` 表示第 5 秒的第 12 帧。

用了多线程写文件，读帧还是单线程的（cv2.VideoCapture 不支持并行读取）。

### 2. 图片转音频

`toolbox/image_to_audio.py`

核心部分。把每张图片当作频谱图，反向生成对应的音频。

#### 坐标映射

图片的 Y 轴映射到频率：
- 图片底部 = 0 Hz（低频）
- 图片顶部 = Nyquist 频率（采样率的一半）

图片的 X 轴映射到时间：
- 图片左边 = 音频开始
- 图片右边 = 音频结束

像素亮度映射到该频率的能量强度，白色=强，黑色=无。

#### 参数

- 采样率 192kHz，Nyquist 是 96kHz，频率范围够用
- n_fft = 4096，频率分辨率约 47Hz
- hop_length = 512，时间分辨率约 2.7ms

#### Griffin-Lim 算法

频谱图只有幅度信息，没有相位。Griffin-Lim 是个迭代算法，通过反复做 ISTFT -> STFT 来估计相位，让生成的音频听起来比较自然。

用的 librosa 的实现，32 次迭代，比手写的快很多。

当然了，这并不会让频谱更好看，只是我的一点恶趣味。

### 3. 频谱分析

`toolbox/analyze_spectrum.py` - 单文件分析
`toolbox/batch_spectrum.py` - 批量分析

对生成的音频进行频谱分析，输出双子图：
- 上方：时域波形
- 下方：时频谱图（线性频率轴）

参数与 image_to_audio 保持一致：
- nperseg = 4096
- hop_length = 512
- noverlap = nperseg - hop_length

批量处理使用 multiprocessing.Pool，默认 4 进程并行，每 50 个任务重启进程以释放内存。为避免跨平台字体问题，标签使用英文。

### 4. 视频合成

`toolbox/make_video.py`

将频谱图片按时间顺序合成视频。

- 读取 `output_spectrum` 目录下所有 `{秒}_{帧}_spectrum.png` 格式的图片
- 按 (秒, 帧) 排序
- 以 30fps 合成为 MP4 视频

## 目录结构

```
raw_video/          原始视频
analyzed_image/     拆出来的帧
outputs/            生成的音频片段
output_spectrum/    频谱分析图
toolbox/            处理脚本
  ├── extract_frames.py   视频拆帧
  ├── image_to_audio.py   图片转音频
  ├── analyze_spectrum.py 单文件频谱分析
  ├── batch_spectrum.py   批量频谱分析
  └── make_video.py       合成视频
```

## 依赖

- opencv-python: 视频/图片处理
- librosa: Griffin-Lim 算法
- scipy: wav 文件读写、频谱计算
- numpy: 数值计算
- matplotlib: 频谱图绘制

## 已知问题

1. Griffin-Lim 生成的音频有点"金属感"，这是算法本身的限制
2. 高频部分细节会丢失，因为人耳对高频不敏感，算法也不会特别优化这部分
3. 处理速度主要受 CPU 限制，GPU 加速需要额外配置
4. 批量频谱分析内存占用较大，16GB 内存建议使用 4 进程
