#!/usr/bin/env python3
"""
Converted from Jupyter notebook: Batch processing.ipynb

Usage:
  python batch_processing.py --action remove_audio --video_dir /path/to/videos --audio_removed_dir /path/to/audio_removed
  python batch_processing.py --action detect --audio_removed_dir /path/to/audio_removed --output_dir /path/to/output --model /path/to/best.pt
  python batch_processing.py --action detect_no_labels --video_dir /path/to/videos --output_dir /path/to/output --model /path/to/best.pt
  python batch_processing.py --action detect_color --video_dir /path/to/videos --output_dir /path/to/output --model /path/to/best.pt
  python batch_processing.py --action photos --video_dir /path/to/photos --output_dir /path/to/output --model /path/to/best.pt

All paths (--video_dir, --audio_removed_dir, --output_dir, --model) are required; fill in your own.
"""
import os
import subprocess
import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def remove_audio(video_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    for filename in os.listdir(video_directory):
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.mp4')):
            try:
                video_path = os.path.join(video_directory, filename)
                output_path = os.path.join(output_directory, filename)
                ffmpeg_command = ['ffmpeg', '-i', video_path, '-c', 'copy', '-an', output_path]
                subprocess.run(ffmpeg_command, check=True)
                print(f"Processed: {filename}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {filename}: {e}")
    print("All videos processed!")


def load_model(model_path):
    if YOLO is None:
        raise RuntimeError('ultralytics.YOLO not available; install ultralytics package')
    return YOLO(model_path)


def _prepare_boxes(r):
    # Return numpy arrays: boxes (N,4), classes (N,), confs (N,)
    if getattr(r, 'boxes', None) is None:
        return np.zeros((0, 4)), np.array([]), np.array([])
    try:
        boxes = r.boxes.xyxy.cpu().numpy()
    except Exception:
        boxes = np.array([])
    try:
        classes = r.boxes.cls.cpu().numpy()
    except Exception:
        classes = np.array([])
    try:
        confs = r.boxes.conf.cpu().numpy()
    except Exception:
        confs = np.array([])
    return boxes, classes, confs


def process_video(model_path, video_path, output_path, tracker_config='bytetrack.yaml', draw_labels=True, color_map=None):
    model = load_model(model_path)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    results = model.track(source=str(video_path), tracker=tracker_config, save=False, save_txt=False, stream=True, verbose=False)
    for r in results:
        frame = r.orig_img
        boxes, classes, confs = _prepare_boxes(r)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            cls = int(classes[i]) if classes.size > i else None
            conf = float(confs[i]) if confs.size > i else None
            color = (0, 255, 0)
            if color_map is not None and cls is not None:
                color = tuple(map(int, color_map.get(int(cls), (0, 255, 0))))
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            if draw_labels and cls is not None:
                label = f"{model.names[int(cls)]} {conf:.2f}" if conf is not None else f"{model.names[int(cls)]}"
                cv2.putText(frame, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()


def process_videos_in_dir(model_path, video_dir, output_dir, tracker='bytetrack.yaml', pattern='*.MP4', draw_labels=True, color_map=None):
    os.makedirs(output_dir, exist_ok=True)
    for video_file in Path(video_dir).glob(pattern):
        video_path = str(video_file)
        output_path = os.path.join(output_dir, video_file.name)
        print(f'Processing {video_file.name} -> {output_path}')
        process_video(model_path, video_path, output_path, tracker_config=tracker, draw_labels=draw_labels, color_map=color_map)
    print('Processing complete.')


def get_color_map(num_classes):
    np.random.seed(42)
    return {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(num_classes)}


def process_photos(model_path, photo_dir, output_dir, tracker_config='bytetrack.yaml'):
    model = load_model(model_path)
    os.makedirs(output_dir, exist_ok=True)
    for photo_file in Path(photo_dir).glob('*.*'):
        photo_path = str(photo_file)
        image = cv2.imread(photo_path)
        if image is None:
            print(f"Error: Could not read image {photo_path}")
            continue
        results = model.track(source=photo_path, tracker=tracker_config, save=False, save_txt=False, stream=True)
        for r in results:
            frame = r.orig_img
            boxes, classes, confs = _prepare_boxes(r)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                cls = int(classes[i]) if classes.size > i else None
                conf = float(confs[i]) if confs.size > i else None
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        output_path = os.path.join(output_dir, photo_file.name)
        cv2.imwrite(output_path, frame)
        print(f'Processed and saved {output_path}')
    print('Processing complete.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['remove_audio', 'detect', 'detect_no_labels', 'detect_color', 'photos'], required=True)
    parser.add_argument('--video_dir', required=True, help='Directory containing input videos (or photos, for the photos action)')
    parser.add_argument('--audio_removed_dir', required=True, help='Directory for audio-removed videos (output of remove_audio, input of detect)')
    parser.add_argument('--output_dir', required=True, help='Directory to write processed output to')
    parser.add_argument('--model', required=True, help='Path to YOLO model weights (.pt)')
    parser.add_argument('--tracker', default='bytetrack.yaml')
    args = parser.parse_args()

    if args.action == 'remove_audio':
        remove_audio(args.video_dir, args.audio_removed_dir)

    elif args.action == 'detect':
        process_videos_in_dir(args.model, args.audio_removed_dir, args.output_dir, tracker=args.tracker, pattern='*.MP4', draw_labels=True)

    elif args.action == 'detect_no_labels':
        process_videos_in_dir(args.model, args.video_dir, args.output_dir, tracker=args.tracker, pattern='*.mp4', draw_labels=False)

    elif args.action == 'detect_color':
        num_classes = 80
        try:
            model = load_model(args.model)
            num_classes = len(model.names)
        except Exception:
            pass
        color_map = get_color_map(num_classes)
        process_videos_in_dir(args.model, args.video_dir, args.output_dir, tracker=args.tracker, pattern='*.mp4', draw_labels=True, color_map=color_map)

    elif args.action == 'photos':
        process_photos(args.model, args.video_dir, args.output_dir, tracker_config=args.tracker)


if __name__ == '__main__':
    main()
