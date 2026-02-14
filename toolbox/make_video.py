import cv2
from pathlib import Path
import re

# 输入输出
input_dir = Path('output_spectrum')
output_video = 'spectrum_video.mp4'

# 视频参数
FPS = 30

def parse_filename(filename):
    """解析文件名，返回 (秒, 帧) 元组"""
    # 格式: 秒_帧_spectrum.png
    match = re.match(r'(\d+)_(\d+)_spectrum\.png', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

# 获取所有图片并排序
image_files = list(input_dir.glob('*_spectrum.png'))
print(f"找到 {len(image_files)} 张图片")

# 按 (秒, 帧) 排序
image_files_sorted = []
for f in image_files:
    parsed = parse_filename(f.name)
    if parsed:
        image_files_sorted.append((parsed[0], parsed[1], f))

image_files_sorted.sort(key=lambda x: (x[0], x[1]))
print(f"有效图片: {len(image_files_sorted)} 张")

if not image_files_sorted:
    print("没有找到有效的图片!")
    exit(1)

# 读取第一张图片获取尺寸
first_img = cv2.imread(str(image_files_sorted[0][2]))
height, width = first_img.shape[:2]
print(f"图片尺寸: {width}x{height}")

# 创建视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, FPS, (width, height))

# 写入所有帧
for i, (sec, frame, filepath) in enumerate(image_files_sorted):
    img = cv2.imread(str(filepath))
    if img is not None:
        video_writer.write(img)

    if (i + 1) % 100 == 0:
        print(f"已处理 {i + 1}/{len(image_files_sorted)} 帧")

video_writer.release()
print(f"\n完成! 视频已保存为 {output_video}")
print(f"时长: {len(image_files_sorted) / FPS:.2f} 秒")
