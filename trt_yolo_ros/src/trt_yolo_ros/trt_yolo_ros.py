from __future__ import division, print_function

import rospy
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes
from sensor_msgs.msg import Image

from trt_yolo.detector import DarknetTRT
from utils import timeit_ros


class YOLORos(object):
    def __init__(self):
        self._bridge = CvBridge()
        self._read_params()
        self._init_topics()
        self.model = DarknetTRT(
            obj_threshold=self.obj_threshold,
            nms_threshold=self.nms_threshold,
            yolo_type=self.yolo_type,
            weights_path=self.weights_path,
            config_path=self.config_path,
            label_filename=self.label_filename,
            postprocessor_cfg=self.postprocessor_cfg,
            cuda_device=self.cuda_device,
            show_image=self.publish_image,
        )
        self.msg = None
        rospy.loginfo("[trt_yolo_ros] loaded and ready")

    def _read_params(self):
        """ Reading parameters for YOLORos from launch or yaml files """
        self.publish_image = rospy.get_param("~publish_image", False)
        # default paths to weights from different sources
        self.weights_path = rospy.get_param("~weights_path", "./weights/")
        self.config_path = rospy.get_param("~config_path", "./config/")
        self.label_filename = rospy.get_param("~label_filename", "coco_labels.txt")
        # parameters of yolo detector
        self.yolo_type = rospy.get_param("~yolo_type", "yolov3-416")
        self.postprocessor_cfg = rospy.get_param(
            "~postprocessor_cfg", "yolo_postprocess_config.json"
        )
        self.obj_threshold = rospy.get_param("~obj_threshold", 0.6)
        self.nms_threshold = rospy.get_param("~nms_threshold", 0.3)
        # default cuda device
        self.cuda_device = rospy.get_param("~cuda_device", 0)
        rospy.logdebug("[trt_yolo_ros] parameters read")

    @staticmethod
    def _read_subscriber_param(name):
        """ reading subscriber parameters from launch or yaml files """
        topic = rospy.get_param("~subscriber/" + name + "/topic")
        queue_size = rospy.get_param("~subscriber/" + name + "/queue_size", 10)
        return topic, queue_size

    @staticmethod
    def _read_publisher_param(name):
        """ reading publisher parameters from launch or yaml files """
        topic = rospy.get_param("~publisher/" + name + "/topic")
        queue_size = rospy.get_param("~publisher/" + name + "/queue_size", 1)
        latch = rospy.get_param("~publisher/" + name + "/latch", False)
        return topic, queue_size, latch

    def _init_topics(self):
        """ This function is initializing node publisher and subscribers for the node """
        # Publisher
        topic, queue_size, latch = self._read_publisher_param("bounding_boxes")
        self._pub = rospy.Publisher(
            topic, BoundingBoxes, queue_size=queue_size, latch=latch
        )
        topic, queue_size, latch = self._read_publisher_param("image")
        self._pub_viz = rospy.Publisher(
            topic, Image, queue_size=queue_size, latch=latch
        )
        # Subscriber
        topic, queue_size = self._read_subscriber_param("image")
        self._image_sub = rospy.Subscriber(
            topic, Image, self._image_callback, queue_size=queue_size, buff_size=2 ** 24
        )
        rospy.logdebug("[trt_yolo_ros] publishers and subsribers initialized")

    def _image_callback(self, msg):
        """ Main callback which is saving the last received image """
        self.msg = None
        try:
            self.image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            self.msg = msg
            rospy.logdebug("[trt_yolo_ros] image received")
        except CvBridgeError as e:
            print(e)

    def _write_message(self, detection_results, boxes, scores, classes):
        """ populate output message with input header and bounding boxes information """
        if boxes is None:
            return None
        for box, score, category in zip(boxes, scores, classes):
            # Populate darknet message
            left, bottom, right, top = box
            detection_msg = BoundingBox()
            detection_msg.xmin = left
            detection_msg.xmax = right
            detection_msg.ymin = bottom
            detection_msg.ymax = top
            detection_msg.probability = score
            detection_msg.Class = category
            detection_results.bounding_boxes.append(detection_msg)
        return detection_results

    @timeit_ros
    def process_frame(self):
        """ Main function to process the frame and run the infererence """
        if self.msg is None:
            return
        # Initialize detection results
        detection_results = BoundingBoxes()
        detection_results.header = self.msg.header
        detection_results.image_header = self.msg.header
        self.msg = None
        rospy.logdebug("[trt_yolo_ros] processing frame")
        boxes, classes, scores, visualization = self.model(self.image)
        # and deleting the processed message from memmory

        # construct message
        self._write_message(detection_results, boxes, scores, classes)

        # send message
        try:
            rospy.logdebug("[trt_yolo_ros] publishing")
            self._pub.publish(detection_results)
            if self.publish_image:
                self._pub_viz.publish(self._bridge.cv2_to_imgmsg(visualization, "bgr8"))
        except CvBridgeError as e:
            print(e)

