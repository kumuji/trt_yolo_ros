import rospy

def timeit_ros(method):
    def timed(*args, **kw):
        ts = rospy.get_rostime()
        result = method(*args, **kw)
        te = rospy.get_rostime()
        rospy.logdebug(
            "{} execution time: {:2.2f} ms".format(
                method.__name__, ((te - ts).nsecs * 0.000001)
            )
        )
        return result

    return timed
