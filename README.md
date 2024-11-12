
<h1 align="center"><span>YOLOv8 + Pose estimation+ ByteTracker</span></h1>

This project integrates YOLOv8 (Object detection and pose estimation)and ByteTracker for real-time, TensorRT-optimized object detection and tracking, extending the existing [TensorRT-YOLOv8](https://github.com/spacewalk01/tensorrt-YOLOv8) implementation.

<p align="center" margin: 0 auto;>
  <img src="assets/demo.gif" width="360px" />
</p>

## Usage


- CPP(TensorRT):
1.Clone [yolov8](https://github.com/ultralytics/ultralytics/tree/v8.2.103) and download [yolov8s-pose.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-pose.pt)
2.Export onnx model
``` python
from ultralytics import YOLO
# Load a model
model = YOLO("yolov8s-pose.pt")  # load an official model
# Export the model
model.export(format="onnx")
``` 
3.Convert model to trt
``` shell
trtexec.exe --onnx=yolov8s-pose.onnx --saveEngine=yolov8s-pose.engine
```

4.Inference
``` shell
cd build/release
yolov8-pose-bytetrack-trt.exe yolov8s-pose.engine test.mp4 # the video path
```

## What is next?

- [ ] Python(TensorRT)

## Setup:Build project by using the following commands or  **cmake-gui**(Windows).

**CPP:**

1. Windows:
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

2. Linux(not tested):
```bash
mkdir build
cd build && mkdir out_dir
cmake ..
make
```
## Requirement
   - TensorRT
   - CUDA, CudaNN
   - Eigen 3.3
   - C++ compiler with C++17 or higher support
   - CMake 3.14 or higher
   - OpenCV
     
## Acknowledgement

This project is based on the following awesome projects:
- [YOLOv8](https://github.com/ultralytics/ultralytics) - YOLOv8: Ultralytics.
- [ByteTrack](https://github.com/Vertical-Beach/ByteTrack-cpp) - C++ implementation of ByteTrack algorithm. 
- [yolov9-bytetrack-tensorrt](https://github.com/spacewalk01/yolov9-bytetrack-tensorrt) - C++ implementation of YOLOv9 and bytetrack using TensorRT API.
