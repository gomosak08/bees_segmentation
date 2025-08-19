#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive YOLO pipeline script.

- Asks whether to TRAIN or PREDICT
- Prompts for model, dataset, and output paths
- Runs the chosen task using Ultralytics YOLO interface

Requirements:
    pip install ultralytics
"""

import os
from ultralytics import YOLO

def ask(msg: str, default: str | None = None) -> str:
    """Prompt with optional default."""
    if default:
        msg = f"{msg} [{default}]: "
    else:
        msg = f"{msg}: "
    val = input(msg).strip()
    return val if val else (default or "")

def do_train():
    print("\n=== TRAIN MODE ===")
    model_path = ask("Path to model (.pt)", "yolo11n-seg.pt")
    data_yaml = ask("Path to dataset YAML", "/home/gomosak/abejas/abejas_segmentation/data.yaml")
    epochs = int(ask("Number of epochs", "100"))
    imgsz = int(ask("Image size", "640"))

    print("\nStarting training...")
    model = YOLO(model_path)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz
    )
    print("\n✅ Training finished.")

def do_predict():
    print("\n=== PREDICT MODE ===")
    model_path = ask("Path to trained model (.pt)", "/home/gomosak/abejas/runs/segment/train/weights/best.pt")
    source = ask("Source directory or file", "/home/gomosak/abejas/abejas_segmentation/test/images")
    project = ask("Project output folder", "/home/gomosak/abejas/outputs")
    name = ask("Experiment name", "test_predict")
    conf = float(ask("Confidence threshold", "0.25"))
    iou = float(ask("IoU threshold", "0.7"))

    print("\nStarting prediction...")
    model = YOLO(model_path)
    model.predict(
        source=source,
        task="segment",
        save=True,
        conf=conf,
        iou=iou,
        save_txt=True,
        project=project,
        name=name,
        exist_ok=True
    )
    print("\n✅ Prediction finished. Results saved in:", os.path.join(project, name))

def main():
    print("YOLO Interactive Script")
    choice = ask("Do you want to [train] or [predict]?", "train").lower()

    if choice.startswith("t"):
        do_train()
    elif choice.startswith("p"):
        do_predict()
    else:
        print("❌ Invalid choice. Please run again and select 'train' or 'predict'.")

if __name__ == "__main__":
    main()
