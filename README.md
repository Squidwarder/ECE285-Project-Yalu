# Inventory Scanner - Track your inventory

> A ECE 285 Project by Yalu Ouyang


## Project Notes

### Inference speed

The inference speed of YOLO models change pretty drastically. 

Running on the CPU, the model trained on YOLOv8n has inference time of roughly 100ms, whilst
the YOLOv8s and YOLOv8m models all took more than 200ms to inference video frames.

This made the footage noticeably choppy, and not ideal for use in realworld situation.

Using a GPU, the inference time dropped noticeably, down to ~30 ms for all three models.

## Dependencies:

```
ultralytics
torch
PySimpleGUI
cv2
numpy
os
```