#!/usr/bin/env python
from __future__ import division, print_function

import rospy
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes
from sensor_msgs.msg import Image

import numpy as np
from trt_package.detector import DarknetTRT


class Detector(object):
    def __init__(self):
        self._bridge = CvBridge()
        self._read_params()

        # Load subscription topics
        self.image_topic = rospy.get_param("~image_topic", "/camera/rgb/image_raw")
        # Load publisher topics
        self.detected_objects_topic = rospy.get_param("~detected_objects_topic")
        self.published_image_topic = rospy.get_param("~detections_image_topic")

        self._init()
        self.model = DarknetTRT(
            obj_threshold=0.6,
            nms_threshold=0.7,
            h=self.yolo_input_h,
            w=self.yolo_input_w,
            label_file_path=self.label_path,
            trt_engine=self.trt_engine_path,
            onnx_engine=self.onnx_engine_path,
            show_image=self.publish_image,
        )
        rospy.loginfo("[detector] loaded and ready")

    # def _read_subscriber_param(self, name):
    #     """
    #     @brief      Convenience function to read subscriber parameter.
    #     """
    #     topic = rospy.get_param("~subscriber/" + name + "/topic")
    #     queue_size = rospy.get_param("~subscriber/" + name + "/queue_size")
    #     return topic, queue_size

    # def _read_publisher_param(self, name):
    #     """
    #     @brief      Convenience function to read publisher parameter.
    #     """
    #     topic = rospy.get_param("~publisher/" + name + "/topic")
    #     queue_size = rospy.get_param("~publisher/" + name + "/queue_size")
    #     latch = rospy.get_param("~publisher/" + name + "/latch")
    #     return topic, queue_size, latch

    def _init(self):
        # Define subscribers
        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self._image_callback,
            queue_size=1,
            buff_size=2 ** 24,
        )
        # Define publishers
        self._pub = rospy.Publisher(
            self.detected_objects_topic, BoundingBoxes, queue_size=10
        )
        if self.publish_image:
            self._pub_viz = rospy.Publisher(
                self.published_image_topic, Image, queue_size=10
            )
        rospy.loginfo("[detector] subsribers and publishers created")
        # """
        # @brief      Initialize ROS connection.
        # """
        # # Publisher
        # topic, queue_size, latch = self._read_publisher_param("word")
        # self._word_pub = rospy.Publisher(
        #         topic, String, queue_size=queue_size, latch=latch)

        # topic, queue_size, latch = self._read_publisher_param("image")
        # self._image_pub = rospy.Publisher(
        #         topic, Image, queue_size=queue_size, latch=latch)

        # # Subscriber
        # topic, queue_size = self._read_subscriber_param("word")
        # self._word_sub = rospy.Subscriber(
        #         topic, String, self._word_callback, queue_size=queue_size)

        # topic, queue_size = self._read_subscriber_param("image")
        # self._image_sub = rospy.Subscriber(
        #         topic, Image, self._image_callback, queue_size=queue_size)

    def _read_params(self):
        self.publish_image = rospy.get_param("~publish_image", False)
        self.trt_engine_path = rospy.get_param("~trt_engine_path", "yolo.engine")
        self.onnx_engine_path = rospy.get_param("~onnx_engine_path", "yolo.onnx")
        self.label_path = rospy.get_param("~label_path", "coco_labels.txt")
        self.yolo_input_h = rospy.get_param("~yolo_input_h", 608)
        self.yolo_input_w = rospy.get_param("~yolo_input_w", 608)
        self.obj_threshold = rospy.get_param("~obj_threshold", 0.6)
        self.nms_threshold = rospy.get_param("~nms_threshold", 0.7)
        self.cuda_device = rospy.get_param("~cuda_device", 0)
        rospy.loginfo("[detector] parameters read")

    def _image_callback(self, msg):
        self.msg = None
        try:
            self.image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            self.msg = msg
        except CvBridgeError as e:
            print(e)

    def process_frame(self):
        start_time = rospy.get_rostime()
        last_publish = rospy.get_rostime()
        rospy.logdebug("[detector] processing frame")
        if self.msg is None:
            return
        # Initialize detection results
        detection_results = BoundingBoxes()
        detection_results.header = self.msg.header
        detection_results.image_header = self.msg.header
        boxes, classes, scores, obj_detected_img = self.model(self.image)
        if boxes is not None:
            for box, score, category in zip(boxes, scores, classes):
                x_coord, y_coord, width, height = box
                left = max(0, np.floor(x_coord + 0.5))
                top = max(0, np.floor(y_coord + 0.5))
                right = min(self.image.shape[1], np.floor(x_coord + width + 0.5))
                bottom = min(self.image.shape[0], np.floor(y_coord + height + 0.5))
                # Populate darknet message
                detection_msg = BoundingBox()
                detection_msg.xmin = int(left)
                detection_msg.xmax = int(right)
                detection_msg.ymin = int(bottom)
                detection_msg.ymax = int(top)
                detection_msg.probability = score
                detection_msg.Class = self.model.all_categories[int(category)]
                detection_results.bounding_boxes.append(detection_msg)

        try:
            rospy.logdebug("[detector] publishing")
            self._pub.publish(detection_results)
            if self.publish_image:
                self._pub_viz.publish(
                    self._bridge.cv2_to_imgmsg(obj_detected_img, "bgr8")
                )
            last_publish = rospy.get_rostime()
        except CvBridgeError as e:
            print(e)
        delay = (last_publish.nsecs - start_time.nsecs) / 1000000
        rospy.logdebug("[detector] interference time for callback[ms]={}".format(delay))


if __name__ == "__main__":
    rospy.init_node("detector")
    rospy.loginfo("[detector] starting the node")
    rospy.Rate(10)
    network = Detector()
    while not rospy.is_shutdown():
        network.process_frame()
