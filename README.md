### Description
This repository is a python2 ros package that is made similar to the [darknet_ros](https://github.com/leggedrobotics/darknet_ros) package with only difference that it uses tensorrt for acceleration.

It is faster on devices like jetson nano, tx2 or xavier.

### Installation
TensorRT is in requirements, but can't be donloaded using pip. Make sure that tensorrt is installed on your system. For x86 systems you can download the file from nvidia website, and for jetson - it should be already preinstalled on your system.

After that:

```
  python -m pip install -r requirements.txt
```

### Performance

| Model type      | Device  | Inference time | max FPS |
| --------------- | ------  | -------------- | ------- |
| yolov3-tiny-288 | GTX1080 | 10ms           | 100     |
| yolov3-tiny-416 | GTX1080 | 13ms           | 75      |
| yolov3-416      | GTX1080 | 34ms           | 29      |
| yolov3-608      | GTX1080 | 60ms           | 16      |


### Licenses
--------

I referenced source code of [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) samples to develop this repository.  Those NVIDIA samples are under [Apache License 2.0](https://github.com/NVIDIA/TensorRT/blob/master/LICENSE).

Also, a big thanks to [jkjung-avt](https://github.com/jkjung-avt/) and his project with tensorrt samples without him it would hard.
