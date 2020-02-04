from __future__ import division, print_function

import os

import rospy

import cv2
import numpy as np
import pycuda.autoinit  # nescessary for initializing cuda cards
import pycuda.driver as cuda
import tensorrt as trt
from utils import read_json
from yolov3_to_onnx import build_onnx_engine


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
        # input to the node node
        dim = int(yolo_type[-3:])
        self.yolo_h_w = (dim, dim)
        self.h = 0
        self.w = 0
        self.show_image = show_image
        # input to the model

        # allocating memmory on gpu
        self.trt_logger = trt.Logger()
        self.engine = self.get_engine(weights_path, config_path, yolo_type)
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        self.context = self.engine.create_execution_context()
        self.all_categories = [
            line.rstrip("\n")
            for line in open(os.path.join(config_path, label_filename))
        ]

        # reading yolo parameters for processing
        yolo_type = "yolov3-tiny" if "tiny" in yolo_type else "yolov3"
        postprocessor_cfg = os.path.join(config_path, postprocessor_cfg)
        postprocessor_cfg = read_json(postprocessor_cfg)[yolo_type]
        postprocessor_cfg.update(
            {
                "obj_threshold": obj_threshold,  # Threshold for object coverage, float value between 0 and 1
                "nms_threshold": nms_threshold,  # Threshold for non-max suppression algorithm, float value between 0 and 1
                "yolo_input_resolution": self.yolo_h_w,
                "class_num": len(self.all_categories),
            }
        )
        self.postprocessor = PostprocessYOLO(postprocessor_cfg)
        self.drawer = None
        self.drawer = Visualization(self.all_categories)

    def __call__(self, image):
        padded_h_w, image_prepared = self.image_preparation(image)

        trt_outputs = []
        self.inputs[0].host = image_prepared
        trt_outputs = self.do_inference(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream,
        )

        boxes, classes, scores = self.postprocessor.process(trt_outputs, padded_h_w)
        # Draw the bounding boxes onto the original input image and save it as a PNG file
        obj_detected_img = image
        if boxes is None:
            return boxes, classes, scores, obj_detected_img
        for i in range(len(boxes)):
            x, y, width, height = boxes[i]
            x = x - ((padded_h_w[1] - self.w) // 2)
            y = y - ((padded_h_w[0] - self.h) // 2)
            left = max(0, np.floor(x + 0.5).astype(int))
            top = max(0, np.floor(y + 0.5).astype(int))
            right = min(self.w, np.floor(x + width + 0.5).astype(int))
            bottom = min(self.h, np.floor(y + height + 0.5).astype(int))
            boxes[i] = [left, bottom, right, top]
        obj_detected_img = self.drawer(image, boxes, scores, classes)
        return boxes, classes, scores, obj_detected_img

    def get_engine(self, weights_path, configs_path, yolo_type):
        """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
        engine_file_name = "%s.trt" % yolo_type
        engine_file_path = os.path.join(weights_path, engine_file_name)
        if not os.path.exists(engine_file_path):
            self.build_trt_engine(weights_path, configs_path, yolo_type)
        # If a serialized engine exists, use it instead of building an engine.
        rospy.logdebug("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def build_trt_engine(self, weights_path, configs_path, yolo_type):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        rospy.logdebug("Building trt engine file")
        onnx_file_name = "%s.onnx" % yolo_type
        engine_file_name = "%s.trt" % yolo_type
        onnx_file_path = os.path.join(weights_path, onnx_file_name)
        engine_file_path = os.path.join(weights_path, engine_file_name)
        if not os.path.exists(onnx_file_path):
            build_onnx_engine(weights_path, configs_path, yolo_type)
        with trt.Builder(
            self.trt_logger
        ) as builder, builder.create_network() as network, trt.OnnxParser(
            network, self.trt_logger
        ) as parser:
            builder.max_workspace_size = 1 << 28  # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            rospy.loginfo("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                rospy.loginfo("Beginning ONNX file parsing")
                parser.parse(model.read())
            rospy.loginfo("Completed parsing of ONNX file")
            rospy.loginfo(
                "Building an engine from file {}; this may take a while...".format(
                    onnx_file_path
                )
            )
            engine = builder.build_cuda_engine(network)
            rospy.loginfo("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())

    def _allocate_buffers(self):
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
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        context.execute_async(
            batch_size=batch_size, bindings=bindings, stream_handle=stream.handle
        )
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        stream.synchronize()
        return [out.host for out in outputs]

    def image_preparation(self, img_raw):
        """ Extract image and shape """
        height, width, channels = img_raw.shape
        if (height != self.h) or (width != self.w):
            self.h = height
            self.w = width

            # Determine image to be used
        padded_image = np.zeros(
            (max(self.h, self.w), max(self.h, self.w), channels)
        ).astype(float)

        # Add padding
        if self.w > self.h:
            padded_image[
                (self.w - self.h) // 2 : self.h + (self.w - self.h) // 2, :, :
            ] = img_raw
        else:
            padded_image[
                :, (self.h - self.w) // 2 : self.w + (self.h - self.w) // 2, :
            ] = img_raw
        height, width, _ = padded_image.shape

        # Resize and normalize
        input_img = cv2.resize(padded_image, self.yolo_h_w) / 255
        input_img = input_img.transpose((2, 0, 1))  # HWC to CHW format
        input_img = np.array(input_img, dtype=np.float32, order="C")
        input_img = np.expand_dims(input_img, axis=0)  # CHW to NCHW format
        return (height, width), input_img


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class PostprocessYOLO(object):
    """Class for post-processing the three outputs tensors from YOLO."""

    def __init__(self, postprocessor_args):
        """Initialize with all values that will be kept when processing several frames.
        Assuming 3 outputs of the network in the case of (large) YOLOv3.

        Keyword arguments:
        yolo_masks -- a list of 3 three-dimensional tuples for the YOLO masks
        yolo_anchors -- a list of 9 two-dimensional tuples for the YOLO anchors
        object_threshold -- threshold for object coverage, float value between 0 and 1
        nms_threshold -- threshold for non-max suppression algorithm, float value between 0 and 1
        input_resolution_yolo -- input resolution in HW order
        """
        # initializing parameters for the postprocessing node
        postprocessor_args["output_shapes"] = [
            tuple(
                [
                    a,
                    b,
                    postprocessor_args["yolo_input_resolution"][0] // c,
                    postprocessor_args["yolo_input_resolution"][1] // d,
                ]
            )
            for a, b, c, d in postprocessor_args["output_shapes"]
        ]
        self.masks = postprocessor_args["masks"]
        self.anchors = postprocessor_args["anchors"]
        self.object_threshold = postprocessor_args["obj_threshold"]
        self.nms_threshold = postprocessor_args["nms_threshold"]
        self.input_resolution_yolo = postprocessor_args["yolo_input_resolution"]
        self.class_num = postprocessor_args["class_num"]
        self.output_shapes = postprocessor_args["output_shapes"]

    def process(self, outputs, resolution_raw):
        """Take the YOLOv3 outputs generated from a TensorRT forward pass, post-process them
        and return a list of bounding boxes for detected object together with their category
        and their confidences in separate lists.

        Keyword arguments:
        outputs -- outputs from a TensorRT engine in NCHW format
        resolution_raw -- resolution of input image in format HW
        """
        outputs = [
            output.reshape(shape) for output, shape in zip(outputs, self.output_shapes)
        ]

        outputs_reshaped = list()
        for output in outputs:
            outputs_reshaped.append(self._reshape_output(output))

        boxes, categories, confidences = self._process_yolo_output(
            outputs_reshaped, resolution_raw
        )

        if (boxes is None) or len(boxes) < 1:
            return boxes, categories, confidences

        return boxes, categories, confidences

    def _reshape_output(self, output):
        """Reshape a TensorRT output from NCHW to NHWC format (with expected C=255),
        and then return it in (height,width,3,85) dimensionality after further reshaping.

        Keyword argument:
        output -- an output from a TensorRT engine after inference
        """
        output = np.transpose(output, [0, 2, 3, 1])
        _, height, width, _ = output.shape
        dim1, dim2 = height, width
        dim3 = 3  # probably color
        dim4 = 4 + 1 + self.class_num  # output classes and box anker residuals
        return np.reshape(output, (dim1, dim2, dim3, dim4))

    def _process_yolo_output(self, outputs_reshaped, resolution_raw):
        """Take in a list of three reshaped YOLO outputs in (height,width,3,85) shape and return
        return a list of bounding boxes for detected object together with their category and their
        confidences in separate lists.

        Keyword arguments:
        outputs_reshaped -- list of three reshaped YOLO outputs as NumPy arrays
        with shape (height,width,3,85)
        resolution_raw -- resolution of input image in format HW
        """

        # E.g. in YOLOv3-608, there are three output tensors, which we associate with their
        # respective masks. Then we iterate through all output-mask pairs and generate candidates
        # for bounding boxes, their corresponding category predictions and their confidences:
        boxes, categories, confidences = list(), list(), list()
        for output, mask in zip(outputs_reshaped, self.masks):
            box, category, confidence = self._process_feats(output, mask)
            box, category, confidence = self._filter_boxes(box, category, confidence)
            boxes.append(box)
            categories.append(category)
            confidences.append(confidence)

        boxes = np.concatenate(boxes)
        categories = np.concatenate(categories)
        confidences = np.concatenate(confidences)

        # Scale boxes back to original image shape:
        height, width = resolution_raw
        image_dims = [width, height, width, height]
        boxes = boxes * image_dims

        # Using the candidates from the previous (loop) step, we apply the non-max suppression
        # algorithm that clusters adjacent bounding boxes to a single bounding box:
        nms_boxes, nms_categories, nscores = list(), list(), list()
        for category in set(categories):
            idxs = np.where(categories == category)
            box = boxes[idxs]
            category = categories[idxs]
            confidence = confidences[idxs]

            keep = self._nms_boxes(box, confidence)

            nms_boxes.append(box[keep])
            nms_categories.append(category[keep])
            nscores.append(confidence[keep])

        if not nms_categories and not nscores:
            return None, None, None

        boxes = np.concatenate(nms_boxes)
        categories = np.concatenate(nms_categories)
        confidences = np.concatenate(nscores)

        return boxes, categories, confidences

    def _process_feats(self, output_reshaped, mask):
        """Take in a reshaped YOLO output in height,width,3,85 format together with its
        corresponding YOLO mask and return the detected bounding boxes, the confidence,
        and the class probability in each cell/pixel.
        Keyword arguments:
        output_reshaped -- reshaped YOLO output as NumPy arrays with shape (height,width,3,85)
        mask -- 2-dimensional tuple with mask specification for this output
        """

        def sigmoid_v(array):
            return np.reciprocal(np.exp(-array) + 1.0)

        def exponential_v(array):
            return np.exp(array)

        grid_h, grid_w, _, _ = output_reshaped.shape

        anchors = [self.anchors[i] for i in mask]

        # Reshape to N, height, width, num_anchors, box_params:
        anchors_tensor = np.reshape(anchors, [1, 1, len(anchors), 2])
        box_xy = sigmoid_v(output_reshaped[..., 0:2])
        box_wh = exponential_v(output_reshaped[..., 2:4]) * anchors_tensor
        box_confidence = sigmoid_v(output_reshaped[..., 4:5])
        box_class_probs = sigmoid_v(output_reshaped[..., 5:])

        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)

        box_xy += grid
        box_xy /= (grid_w, grid_h)
        box_wh /= self.input_resolution_yolo
        box_xy -= box_wh / 2.0
        boxes = np.concatenate((box_xy, box_wh), axis=-1)

        # boxes: centroids, box_confidence: confidence level, box_class_probs:
        # class confidence
        return boxes, box_confidence, box_class_probs

    def _filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Take in the unfiltered bounding box descriptors and discard each cell
        whose score is lower than the object threshold set during class initialization.

        Keyword arguments:
        boxes -- bounding box coordinates with shape (height,width,3,4); 4 for
        x,y,height,width coordinates of the boxes
        box_confidences -- bounding box confidences with shape (height,width,3,1); 1 for as
        confidence scalar per element
        box_class_probs -- class probabilities with shape (height,width,3,CATEGORY_NUM)

        """
        box_scores = box_confidences * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        pos = np.where(box_class_scores >= self.object_threshold)

        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]

        return boxes, classes, scores

    def _nms_boxes(self, boxes, box_confidences):
        """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding boxes with their
        confidence scores and return an array with the indexes of the bounding boxes we want to
        keep (and display later).

        Keyword arguments:
        boxes -- a NumPy array containing N bounding-box coordinates that survived filtering,
        with shape (N,4); 4 for x,y,height,width coordinates of the boxes
        box_confidences -- a Numpy array containing the corresponding confidences with shape N
        """
        x_coord = boxes[:, 0]
        y_coord = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        areas = width * height
        ordered = box_confidences.argsort()[::-1]

        keep = list()
        while ordered.size > 0:
            # Index of the current element:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
            yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
            xx2 = np.minimum(
                x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]]
            )
            yy2 = np.minimum(
                y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]]
            )

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = areas[i] + areas[ordered[1:]] - intersection

            # Compute the Intersection over Union (IoU) score:
            iou = intersection / union

            # The goal of the NMS algorithm is to reduce the number of adjacent bounding-box
            # candidates to a minimum. In this step, we keep only those elements whose overlap
            # with the current bounding box is lower than the threshold:
            indexes = np.where(iou <= self.nms_threshold)[0]
            ordered = ordered[indexes + 1]

        keep = np.array(keep)
        return keep


class Visualization(object):
    def __init__(
        self,
        possible_lables,
        font_scale=0.8,
        thickness=2,
        font=cv2.FONT_HERSHEY_SIMPLEX,
    ):
        self.possible_lables = possible_lables
        self.font_scale = font_scale
        self.thickness = thickness
        self.font = font
        self.classes_colors = {}

    def __call__(self, image_raw, angles, confidences, labels):
        if angles is None:
            return image_raw
        # Copy image and visualize
        for box, score, label in zip(angles, confidences, labels):
            label = self.possible_lables[label]
            left, bottom, right, top = box

            # Find class color
            if label in self.classes_colors.keys():
                color = self.classes_colors[label]
            else:
                # Generate a new color if first time seen this label
                color = np.random.randint(0, 255, 3)
                self.classes_colors[label] = color

            # Create rectangle
            cv2.rectangle(
                image_raw,
                (int(left), int(top)),
                (int(right), int(bottom)),
                (color[0], color[1], color[2]),
                self.thickness,
            )
            text = ("{:s}: {:.2f}").format(label, score)
            cv2.putText(
                image_raw,
                text,
                (int(left), int(bottom + 20)),
                self.font,
                self.font_scale,
                (255, 255, 255),
                self.thickness,
                cv2.LINE_AA,
            )
        return image_raw
