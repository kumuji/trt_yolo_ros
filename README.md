## FIXME LIST
    * cleanup the code so it wouldn't publish garbage - done
    * include darknet_ros_msgs as part of package
    * include code for generating onnx models
    * publish only person class and give the correct label - done
    * draw bounding boxes correctly for visualization and also fix the rgb->bgr - done
    * make it compatible also with yolo 218, 416
    * rewrite post processing step so it is not depending on numba - done
    * also add option of running pytorch model
    * make yaml configs for topics and other params
    * add new roslaunch file that would include yamls
