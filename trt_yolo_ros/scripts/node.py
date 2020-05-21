#!/usr/bin/env python

import rospy
from trt_yolo_ros.trt_yolo_ros import YOLORos


if __name__ == "__main__":
    rospy.init_node("trt_yolo_ros", log_level=rospy.DEBUG)
    rospy.loginfo("[trt_yolo_ros] starting the node")
    try:
        network = YOLORos()
    except rospy.ROSInterruptException:
        pass
    publish_rate = rospy.get_param("~publish_rate", 10)
    sleep_time = rospy.Rate(publish_rate)
    while not rospy.is_shutdown():
        network.process_frame()
        sleep_time.sleep()

