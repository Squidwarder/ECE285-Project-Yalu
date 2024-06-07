# Inventory Scanner - Track your inventory

> A ECE 285 Project by Yalu Ouyang


## Project Notes

### Boxes overlap

In some models I've trained I noticed that multiple boxes were drawn for a single instance of inventory object.
To fix this issue, I used the `iou` argument during YOLO detection. `iou` stands for intersection over union.


### YOLO model notes

Some metrics about YOLO models I used in my project. The number of paramters are measured by the info from training.

| | YOLOv8n | YOLOv8s | YOLOv8m|
|-|---|----|---|
|params (M) | 3.2 | 11.4 | 25.9 |

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

## Citations:

The original YOLO paper from 2015 (revised in 2016):

```
@misc{redmon2016look,
      title={You Only Look Once: Unified, Real-Time Object Detection}, 
      author={Joseph Redmon and Santosh Divvala and Ross Girshick and Ali Farhadi},
      year={2016},
      eprint={1506.02640},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


The YOLOv10 codebase is built with ultralytics and RT-DETR.

```
@article{wang2024yolov10,
  title={YOLOv10: Real-Time End-to-End Object Detection},
  author={Wang, Ao and Chen, Hui and Liu, Lihao and Chen, Kai and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  journal={arXiv preprint arXiv:2405.14458},
  year={2024}
}
```