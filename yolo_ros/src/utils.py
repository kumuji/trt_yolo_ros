import rospy


def timeit(method):
    def timed(*args, **kw):
        ts = rospy.get_rostime().nsecs
        result = method(*args, **kw)
        te = rospy.get_rostime().nsecs
        rospy.logdebug(
            "{} execution time: {:2.2} ms".format(
                method.__name__, ((te - ts) * 1000000.0)
            )
        )
        return result

    return timed
