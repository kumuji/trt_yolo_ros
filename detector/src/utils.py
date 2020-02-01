import rospy


def timeit(method):
    def timed(*args, **kw):
        ts = rospy.get_rostime().nsecs
        result = method(*args, **kw)
        te = rospy.get_rostime().nsecs
        rospy.logdebug(
            "{} execution time: {:2.2f} ms".format(
                method.__name__, ((te - ts) * 0.000001)
            )
        )
        return result

    return timed
