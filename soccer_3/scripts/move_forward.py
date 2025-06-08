#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64

def main():
    rospy.init_node('move_forward')

    pub_lf = rospy.Publisher('/left_front_wheel_velocity_controller/command', Float64, queue_size=10)
    pub_rf = rospy.Publisher('/right_front_wheel_velocity_controller/command', Float64, queue_size=10)
    pub_lb = rospy.Publisher('/left_back_wheel_velocity_controller/command', Float64, queue_size=10)
    pub_rb = rospy.Publisher('/right_back_wheel_velocity_controller/command', Float64, queue_size=10)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        pub_lf.publish(1.0)
        pub_rf.publish(1.0)
        pub_lb.publish(1.0)
        pub_rb.publish(1.0)
        rate.sleep()

if __name__ == '__main__':
    main()

