"""
视频帧提取工具
将视频逐帧拆解，按照 秒_帧.png 格式保存
"""

import cv2
import os
from pathlib import Path


def extract_frames(video_path: str, output_dir: str):
    """
    从视频中提取所有帧

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录路径
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频帧率: {fps} FPS")
    print(f"总帧数: {total_frames}")
    print(f"开始提取帧...")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 计算当前秒数和该秒内的帧序号
        second = int(frame_count // fps)
        frame_in_second = int(frame_count % fps)

        # 保存帧，格式: 秒_帧.png
        filename = f"{second}_{frame_in_second}.png"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"已处理: {frame_count}/{total_frames} 帧")

    cap.release()
    print(f"完成! 共提取 {frame_count} 帧到 {output_dir}")


if __name__ == "__main__":
    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    video_path = project_root / "raw_video" / "1.mp4"
    output_dir = project_root / "analyzed_image"

    extract_frames(str(video_path), str(output_dir))
