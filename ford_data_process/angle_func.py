# functions to process angles


def convert_body_yaw_to_360(yaw_body):
    yaw_360 = 0
    # if (yaw_body >= 0.0) and (yaw_body <=90.0):
    #     yaw_360 = 90.0 - yaw_body

    if (yaw_body >90.0) and (yaw_body <=180.0):
        yaw_360 = 360.0 - yaw_body + 90.0
    else:
        yaw_360 = 90.0 - yaw_body

    # if (yaw_body >= -90) and (yaw_body <0.0):
    #     yaw_360 = 90.0 - yaw_body
    #
    # if (yaw_body >= -180) and (yaw_body < -90):
    #     yaw_360 = 90.0 - yaw_body
    return yaw_360