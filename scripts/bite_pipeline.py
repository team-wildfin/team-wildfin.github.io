#!/usr/bin/env python3
"""Converted from bite_pipeline.ipynb into a single script.

Usage:
  python bite_pipeline.py --video_folder /path/to/videos --model_path /path/to/best.pt --bite_model_path /path/to/bite.pt --output_dir /path/to/output
"""
import argparse
import os
import sys
import time
import shutil
import pickle
import torch
import subprocess
from tqdm import tqdm
import zipfile
import multiprocessing
import math

def extract_and_process_frames(video_path, model, batch_size=2048):
    import cv2
    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    frames = []
    results = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if len(frames) == batch_size:
            batch_results = model.predict(source=frames, verbose=False)
            results.extend(batch_results)
            frames = []

    if frames:
        batch_results = model.predict(source=frames, verbose=False)
        results.extend(batch_results)

    cap.release()
    return results


def process_videos_list(args, video_list, device_str=None):
    # set device environment variable if provided (used for worker processes)
    if device_str is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_str)

    # ensure script path so we can import process_video
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from cut_chromis_track_vids import process_video
    except Exception as e:
        print("Could not import process_video in worker:", e)
        raise

    # Debug: print GPU and device info for this worker
    try:
        import torch as _torch_temp
        print("WORKER DEBUG: CUDA_VISIBLE_DEVICES=", os.environ.get('CUDA_VISIBLE_DEVICES', None))
        print("WORKER DEBUG: torch.cuda.is_available()=", _torch_temp.cuda.is_available())
        try:
            print("WORKER DEBUG: torch.cuda.device_count()=", _torch_temp.cuda.device_count())
            if _torch_temp.cuda.is_available():
                cur = _torch_temp.cuda.current_device()
                print("WORKER DEBUG: current_device=", cur)
                try:
                    print("WORKER DEBUG: device_name=", _torch_temp.cuda.get_device_name(cur))
                except Exception:
                    pass
                print("WORKER DEBUG: memory_allocated=", _torch_temp.cuda.memory_allocated(cur))
                print("WORKER DEBUG: memory_reserved=", _torch_temp.cuda.memory_reserved(cur))
        except Exception as e:
            print("WORKER DEBUG: error querying cuda devices:", e)
    except Exception:
        print("WORKER DEBUG: torch import failed in worker (non-fatal)")

    # Load models only if not running in cut-only mode
    model = None
    bite_model = None
    _torch = None
    if not getattr(args, 'cut_only', False):
        from ultralytics import YOLO
        import torch as _torch
        print(f"Worker starting on device {device_str}, processing {len(video_list)} videos", flush=True)
        model = YOLO(args.model_path)
        bite_model = YOLO(args.bite_model_path)

        # Device setup
        if args.device:
            device = _torch.device(args.device)
        else:
            device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

        try:
            model.to(device=device)
            bite_model.to(device=device)
        except Exception:
            pass
    else:
        print(f"Worker (cut-only) starting, processing {len(video_list)} videos", flush=True)

    for video_filename in video_list:
        video_path = os.path.join(args.video_folder, video_filename)
        video_name = os.path.splitext(video_filename)[0]
        print(f"STARTING: {video_name} -> {video_path} (worker device {device_str})", flush=True)

        # If a matching .pkl exists in --pkl_dir, load it instead of running tracking
        pkl_loaded = False
        if args.pkl_dir:
            candidate = os.path.join(args.pkl_dir, f"{video_name}.pkl")
            if os.path.exists(candidate):
                print(f"Loading existing pkl: {candidate}")
                with open(candidate, 'rb') as f:
                    all_bboxes = pickle.load(f)
                new_file_path = os.path.join(args.output_dir, f"{video_name}.pkl")
                shutil.copy(candidate, new_file_path)
                pkl_loaded = True

        if not pkl_loaded:
            results = model.track(source=video_path, stream=True, verbose=False)

            all_bboxes = []
            for result in results:
                frame_data = []
                boxes = getattr(result, 'boxes', None)
                box_ids = None
                if boxes is not None:
                    box_ids = getattr(boxes, 'id', None)
                if boxes is not None and box_ids is not None:
                    for bbox, track_id in zip(boxes.xyxy, box_ids):
                        bbox = [bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()]
                        frame_data.append({"track_id": int(track_id.item()), "bbox": bbox})
                else:
                    frame_data.append({"track_id": None, "bbox": None})
                all_bboxes.append(frame_data)

            new_file_path = os.path.join(args.output_dir, f"{video_name}.pkl")
            with open(new_file_path, 'wb') as f:
                pickle.dump(all_bboxes, f)

        first_appearances = {}
        for frame_index, frame in enumerate(all_bboxes):
            for track in frame:
                track_id = track['track_id']
                if track_id not in first_appearances:
                    first_appearances[track_id] = frame_index

        # per-video clip directory inside output_dir (keeps clips next to the .pkl)
        clip_dir = os.path.join(args.output_dir, f"{video_name}_clips")
        if os.path.exists(clip_dir):
            shutil.rmtree(clip_dir)
        os.makedirs(clip_dir, exist_ok=True)

        process_video(video_path, new_file_path, extract_dim=(220, 220), n_pixels_crop=2, output_dir=clip_dir)

        # Stream-add clips to a zip while processing to keep disk usage low
        base_zip = os.path.join(args.output_dir, f"{video_name}")
        zip_path = base_zip + '.zip'

        if getattr(args, 'cut_only', False):
            # In cut-only mode we only need to archive the clips (and optionally delete them).
            if not args.keep_clips:
                zipf = zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED)
                for video_file in os.listdir(clip_dir):
                    if not video_file.endswith('.mp4'):
                        continue
                    clip_path = os.path.join(clip_dir, video_file)
                    try:
                        zipf.write(clip_path, arcname=os.path.basename(clip_path))
                    except Exception as e:
                        print(f"Warning: failed to add {clip_path} to zip: {e}")
                    try:
                        os.remove(clip_path)
                    except Exception as e:
                        print(f"Warning: failed to remove {clip_path}: {e}")
                zipf.close()
            else:
                # keep clips; do nothing else
                pass

            # No bite inference or .txt generation in cut-only mode
            print(f"FINISHED (cut-only): {video_name} -> clips at {clip_dir} {'(zipped)' if not args.keep_clips else ''}", flush=True)
            continue

        zipf = None
        if not args.keep_clips:
            zipf = zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED)

        output_txt_path = os.path.join(args.output_dir, f"{video_name}.txt")
        with open(output_txt_path, 'w') as file:
            for video_file in os.listdir(clip_dir):
                if not video_file.endswith('.mp4'):
                    continue
                clip_path = os.path.join(clip_dir, video_file)
                results = extract_and_process_frames(clip_path, bite_model)

                for i, result in enumerate(results, 1):
                    max_prob_class = 'NA'
                    try:
                        probs = getattr(result, 'probs', None)
                        if probs is not None:
                            max_prob_class = int(probs.data.argmax())
                        else:
                            boxes = getattr(result, 'boxes', None)
                            if boxes is not None and hasattr(boxes, 'cls'):
                                max_prob_class = int(boxes.cls.data.argmax())
                    except Exception:
                        max_prob_class = 'NA'

                    try:
                        track_id = int(video_file.split('_')[1].split('.')[0])
                    except Exception:
                        track_id = None

                    first_frame = first_appearances.get(track_id, 0)
                    global_frame_number = i + first_frame

                    mean_x = mean_y = bb_area = 'NA'
                    if global_frame_number < len(all_bboxes):
                        frame_data = all_bboxes[global_frame_number]
                        if frame_data:
                            for bbox_data in frame_data:
                                if bbox_data and bbox_data['track_id'] == track_id:
                                    x1, y1, x2, y2 = map(float, bbox_data['bbox'])
                                    bb_area = (x2 - x1) * (y2 - y1)
                                    mean_x = (x2 + x1) / 2
                                    mean_y = (y2 + y1) / 2
                                    break

                    file.write(f"{track_id}, {global_frame_number+1}, {max_prob_class}, {mean_x}, {mean_y}, {bb_area}\n")

                if zipf is not None:
                    try:
                        zipf.write(clip_path, arcname=os.path.basename(clip_path))
                    except Exception as e:
                        print(f"Warning: failed to add {clip_path} to zip: {e}")
                if not args.keep_clips:
                    try:
                        os.remove(clip_path)
                    except Exception as e:
                        print(f"Warning: failed to remove {clip_path}: {e}")

        if zipf is not None:
            zipf.close()

        print(f"FINISHED: {video_name} -> wrote {output_txt_path} and {zip_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", required=True, help="Directory of input videos to process")
    parser.add_argument("--model_path", required=True, help="Path to the tracking YOLO model weights (.pt)")
    parser.add_argument("--bite_model_path", required=True, help="Path to the bite-classification YOLO model weights (.pt)")
    parser.add_argument("--output_dir", required=True, help="Directory to write tracks, clips, and results to")
    parser.add_argument("--pkl_dir", default=None, help="directory with existing .pkl annotation files (optional)")
    parser.add_argument("--clips_root", default=None, help="optional directory of pre-extracted clips (unused by default)")
    parser.add_argument("--keep_clips", action='store_true', help="If set, keep per-track clip mp4s; otherwise they will be zipped and removed to save space")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes to spawn (one per GPU ideally)")
    parser.add_argument("--gpu_list", default=None, help="Comma-separated GPU ids to assign to workers (e.g. '0,1,2')")
    parser.add_argument("--device", default=None, help="torch device, e.g. cuda:0 (overrides worker GPU assignment)")
    parser.add_argument("--cut_only", action='store_true', help="Only cut clips from existing .pkl files; do not load tracking or bite models")
    args = parser.parse_args()

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from cut_chromis_track_vids import process_video
    except Exception as e:
        print("Could not import process_video from cut_chromis_track_vids:", e)
        raise

    os.makedirs(args.output_dir, exist_ok=True)

    # Check ultralytics availability without importing YOLO (avoid allocating GPU memory in parent)
    try:
        import importlib.util
        if importlib.util.find_spec("ultralytics") is None:
            raise ImportError("ultralytics package not found")
    except Exception as e:
        print("ultralytics YOLO availability check failed:", e)
        raise

    # Use module-level `process_videos_list(args, video_list, device_str)` implementation

    # start timer
    start = time.time()

    # Run sequentially (no multiprocessing)
    all_videos = [f for f in os.listdir(args.video_folder) if f.upper().endswith('.MP4')]
    process_videos_list(args, all_videos, device_str=None)

    end = time.time()
    print("Elapsed (s):", end - start)


if __name__ == '__main__':
    main()
