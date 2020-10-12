## YOLOv3 with TensorRT acceleration on ROS
This repository is a full `python` ros package that is made similar to the [darknet_ros](https://github.com/leggedrobotics/darknet_ros) with only difference that it uses tensorrt acceleration.

So, it is faster on devices like jetson nano, rtx2080, tx2 or xavier.

And it is easier to maintain because fully written in python.

Works only with `tensorrt v6`, for jetsons it means `jeptack<=4.2.1`

---
## Installation
First you need to install `tensorrt` and `opencv-python`. Those are platform specific and a bit tricky to isntall, so I didn't put those in requirements.txt.

### x86
* To get opencv run `python -m pip opencv-python` or build it yourself.
* Installation of tensorrt is also easy: download appropriate tensorrt version from official webiste [tensorrt_download_link](https://developer.nvidia.com/nvidia-tensorrt-download) ("download now" on webiste is a lie - you need to login) and run:
```
python -m pip install $TENSORRT_PATH/python/my_trt_version_cp27_none_linux_x86_x64.whl
```

---
### jetsons
* If you are using global python shame on you, use something like [pyenv](http://www.github.com/pyenv/pyenv) + [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) instead, but luckily you don't need to do anything. If you are using virtual environment - link tensorrt to your site-packages like this:
``` Bash
$ ln -s /usr/local/lib/python2.7/dist-packages/tensorrt $VIRTUAL_ENV/lib/site-packages/
```
* For opencv I suggest build it yourself, check [buildOpencv](https://github.com/JetsonHacksNano/buildOpenCV). And then link libraries as you did with tensorrt in the previous step.

---
### Later for both systems:

``` Bash
$ python -m pip install -r requirements.txt
```

## Performance

You should expect performance for one processed image around the numbers in the table. I checked forums and some users said you can get better numbers if you will switch to `INT8` precision and if you will use `opencv3.8` on jetsons. If the numbers are not good enough check those ideas first. Probably I will try it out later if you are interested.

Also, you can check runtime yourself, a message is published with debug flag.


| Model           | GTX1080 | Xavier  | nano   |
| --------------- | ------- | ------- | ------ |
| yolov3-tiny-288 | 10 ms   | 25 ms   | 80 ms  |
| yolov3-tiny-416 | 13 ms   | 30 ms   | 135 ms |
| yolov3-416      | 33 ms   | 95 ms   | 660 ms |
| yolov3-608      | 62 ms   | 195 ms  | x ms   |


### License and references
--------

I don't really care if you will reference me or will take the code without it. But from my side I think it is good to thank these people:

I referenced source code of [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT). Those NVIDIA samples are under [Apache License 2.0](https://github.com/NVIDIA/TensorRT/blob/master/LICENSE).

And thanks to [jkjung-avt](https://github.com/jkjung-avt/) and his project with tensorrt samples. I took some parts from there.
