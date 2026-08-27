#!/usr/bin/env python3
"""
YOLO training and validation via the Ultralytics CLI.

Usage:
  python yolo_training_and_vallidation.py --action train --data /path/to/dataset/data.yaml --model yolov8s.pt --imgsz 1920 --batch 32 --epochs 50
  python yolo_training_and_vallidation.py --action val --model /path/to/best.pt --data /path/to/dataset/data.yaml --conf 0.5
"""
import argparse
import subprocess


def train(data, model, imgsz, batch, epochs):
    cmd = [
        'yolo', 'task=detect', 'mode=train',
        f'model={model}',
        f'data={data}',
        f'imgsz={imgsz}',
        f'batch={batch}',
        f'epochs={epochs}',
        'plots=True',
    ]
    subprocess.run(cmd, check=True)


def validate(model, data, conf):
    cmd = [
        'yolo', 'task=detect', 'mode=val',
        f'model={model}',
        f'data={data}',
        f'conf={conf}',
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['train', 'val'], required=True)
    parser.add_argument('--data', required=True, help='Path to the dataset data.yaml')
    parser.add_argument('--model', default='yolov8s.pt', help='Pretrained model to start from (train), or trained weights to evaluate (val)')
    parser.add_argument('--imgsz', type=int, default=1920, help='Training image size (train only)')
    parser.add_argument('--batch', type=int, default=32, help='Batch size (train only)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (train only)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (val only)')
    args = parser.parse_args()

    if args.action == 'train':
        train(args.data, args.model, args.imgsz, args.batch, args.epochs)
    elif args.action == 'val':
        validate(args.model, args.data, args.conf)


if __name__ == '__main__':
    main()
