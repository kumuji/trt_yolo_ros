from __future__ import division, print_function

import os
import re

import cv2
import numpy as np
import pycuda.autoinit  # nescessary for initializing cuda cards
import pycuda.driver as cuda
import tensorrt as trt

from yolov3_to_onnx import build_onnx_engine
from postprocessing import PostprocessYOLO, Visualization
from utils import read_json


class DarknetTRT(object):
    def __init__(
        self,
        obj_threshold=0.6,
        nms_threshold=0.7,
        yolo_type="yolov3-416",  # also supported v3-tiny
        weights_path="./weights/",
        config_path="./config/",
        label_filename="coco_labels.txt",
        postprocessor_cfg="yolo_postprocess_config.json",
        cuda_device=0,
        show_image=False,
    ):
        """
        DarknetTRT is a class for using yolo tensorrt detector

        Parameters
        ----------
        obj_threshold : float
            Minimum value for threshold for which the object is considered detected
        nms_threshold : float
            Maximum value for the intersection over union for non-max suppression of bounding_boxes
        yolo_type : str
            yolo architecture to run [yolotype](-tiny)-[Size]
        weights_path : str
            path where weights for yolo are located: "./weights/"
        config_path : str
            path where cfg files for yolo architecture are located: "./config/"
        label_filename : str
            path to the label file "coco_labels.txt"
        postprocessor_cfg : str
            name of the file with information about yolo structure
        cuda_device : int
            device on which you want to run the code
        show_image : str
            return image with drawn bounding boxes, visualisation

        """
        yolo_input_dim = int(re.search(r"\d+$", yolo_type).group())
        self.yolo_h_w = (yolo_input_dim, yolo_input_dim)
        self.input_h, self.input_w = 0, 0
        # allocating memmory on gpu
        self.trt_logger = trt.Logger()
        self.engine = self.get_engine(weights_path, config_path, yolo_type)
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        self.context = self.engine.create_execution_context()
        # postprocessing
        self.categories = [
            line.rstrip("\n")
            for line in open(os.path.join(config_path, label_filename))
        ]
        self.postprocessor = PostprocessYOLO(
            yolo_type=yolo_type,
            config_path=os.path.join(config_path, postprocessor_cfg),
            obj_threshold=obj_threshold,
            nms_threshold=nms_threshold,
            input_resolution=self.yolo_h_w,
            class_num=len(self.categories),
        )
        self.drawer = None
        if show_image:
            self.drawer = Visualization()

    def __call__(self, image):
        """ Main part where all the magic happens """
        image_prepared = self._prepare_image(image)
        trt_outputs = []
        self.inputs[0].host = image_prepared
        trt_outputs = self.do_inference(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream,
        )
        input_resolution_padded = max(self.input_h, self.input_w)
        boxes, classes, scores = self.postprocessor.process(
            trt_outputs, (input_resolution_padded, input_resolution_padded)
        )
        if boxes is not None:
            boxes = self._boxes2angles(boxes)
            classes = [self.categories[label] for label in classes]
        obj_detected_img = None
        if self.drawer is not None:
            obj_detected_img = self.drawer(image, boxes, scores, classes)
        return boxes, classes, scores, obj_detected_img

    def get_engine(self, weights_path, configs_path, yolo_type):
        """ Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
        engine_file_name = "%s.trt" % yolo_type
        engine_file_path = os.path.join(weights_path, engine_file_name)
        if not os.path.exists(engine_file_path):
            self.build_trt_from_onnx(weights_path, configs_path, yolo_type)
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f:
            with trt.Runtime(self.trt_logger) as runtime:
                return runtime.deserialize_cuda_engine(f.read())

    def build_trt_from_onnx(self, weights_path, configs_path, yolo_type):
        """ Takes an ONNX file and creates a TensorRT engine to run inference with """
        print("Building new trt engine file")
        onnx_file_name = "%s.onnx" % yolo_type
        engine_file_name = "%s.trt" % yolo_type
        onnx_file_path = os.path.join(weights_path, onnx_file_name)
        engine_file_path = os.path.join(weights_path, engine_file_name)
        if not os.path.exists(onnx_file_path):
            build_onnx_engine(weights_path, configs_path, yolo_type)
        # building trt from onnx
        with trt.Builder(self.trt_logger) as builder:
            with builder.create_network() as network:
                with trt.OnnxParser(network, self.trt_logger) as parser:
                    builder.max_workspace_size = 1 << 28  # 256MiB
                    builder.max_batch_size = 1
                    # Parse model file
                    print("Loading ONNX file from path {}...".format(onnx_file_path))
                    with open(onnx_file_path, "rb") as model:
                        print("Beginning ONNX file parsing")
                        parser.parse(model.read())
                    print("Completed parsing of ONNX file")
                    print("Building an engine this may take a while...")
                    engine = builder.build_cuda_engine(network)
                    print("Completed creating engine, saving it.")
                    with open(engine_file_path, "wb") as f:
                        f.write(engine.serialize())

    def _prepare_image(self, img_raw):
        """ Padd image, resize it and save initial parameters """
        height, width, _ = img_raw.shape
        if (height != self.input_h) or (width != self.input_w):
            self.input_h = height
            self.input_w = width
        # Determine image to be used
        border_w = (max(self.input_w, self.input_h) - self.input_w) // 2
        border_h = (max(self.input_w, self.input_h) - self.input_h) // 2
        input_img = cv2.copyMakeBorder(
            img_raw,
            top=border_h,
            bottom=border_h,
            left=border_w,
            right=border_w,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )
        input_img = cv2.resize(input_img, self.yolo_h_w) / 255
        input_img = input_img.transpose((2, 0, 1))  # HWC to CHW format
        input_img = np.array(input_img, dtype=np.float32, order="C")
        input_img = np.expand_dims(input_img, axis=0)  # CHW to NCHW format
        return input_img

    def _allocate_buffers(self):
        """ nvidia function to allocate memory on the device """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:
            size = (
                trt.volume(self.engine.get_binding_shape(binding))
                * self.engine.max_batch_size
            )
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    @staticmethod
    def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
        """ nvidia function for doing the inference on the device """
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        context.execute_async(
            batch_size=batch_size, bindings=bindings, stream_handle=stream.handle
        )
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        stream.synchronize()
        return [out.host for out in outputs]

    def _boxes2angles(self, boxes):
        """ Takes a list of boxes in xywh format and transforms it to left.bottom.right.top format. """
        for i in range(len(boxes)):
            x, y, width, height = boxes[i]
            x = x - ((max(self.input_w, self.input_h) - self.input_w) // 2)
            y = y - ((max(self.input_w, self.input_h) - self.input_h) // 2)
            left = max(0, np.floor(x + 0.5).astype(int))
            top = max(0, np.floor(y + 0.5).astype(int))
            right = min(self.input_w, np.floor(x + width + 0.5).astype(int))
            bottom = min(self.input_h, np.floor(y + height + 0.5).astype(int))
            boxes[i] = [left, bottom, right, top]
        return boxes


class HostDeviceMem(object):
    """ Simple helper data class that's a little nicer to use than a 2-tuple """

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

