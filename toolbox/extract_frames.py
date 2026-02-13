"""
视频帧提取工具
将视频逐帧拆解，按照 秒_帧.png 格式保存
"""

import cv2
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def save_frame(args):
    """保存单帧"""
    frame, output_path = args
    cv2.imwrite(output_path, frame)


def extract_frames(video_path: str, output_dir: str, num_threads: int = 8):
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
    print(f"开始提取帧 (使用 {num_threads} 线程)...")

    frame_count = 0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        tasks = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            second = int(frame_count // fps)
            frame_in_second = int(frame_count % fps)
            filename = f"{second}_{frame_in_second}.png"
            output_path = os.path.join(output_dir, filename)

            tasks.append(executor.submit(save_frame, (frame.copy(), output_path)))

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"已处理: {frame_count}/{total_frames} 帧")

        for task in tasks:
            task.result()

    cap.release()
    print(f"完成! 共提取 {frame_count} 帧到 {output_dir}")


if __name__ == "__main__":
    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    video_path = project_root / "raw_video" / "1.mp4"
    output_dir = project_root / "analyzed_image"

    extract_frames(str(video_path), str(output_dir))
