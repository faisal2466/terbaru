#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

class OmniEffortController:
    def __init__(self):
        rospy.init_node('omni_effort_controller')

        # Radius roda dan jarak dari pusat ke roda (ubah sesuai robotmu)
        self.r = rospy.get_param('~wheel_radius', 0.05)  # meter
        self.l = rospy.get_param('~robot_radius', 0.2)   # meter

        # Gain pengali effort (untuk atur kecepatan)
        self.gain = rospy.get_param('~effort_gain', 0.5)

        # Publisher untuk setiap roda
        self.pub_fl = rospy.Publisher('/flwheel_effort_controller/command', Float64, queue_size=10)
        self.pub_fr = rospy.Publisher('/frwheel_effort_controller/command', Float64, queue_size=10)
        self.pub_bl = rospy.Publisher('/blwheel_effort_controller/command', Float64, queue_size=10)
        self.pub_br = rospy.Publisher('/brwheel_effort_controller/command', Float64, queue_size=10)

        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        
        # Log informasi tentang node yang berjalan
        rospy.loginfo("Omni effort controller is running with gain: %.2f", self.gain)
        rospy.loginfo("Wheel radius: %.2f, Robot radius: %.2f", self.r, self.l)

        rospy.spin()

    def cmd_vel_callback(self, msg):
        # Ambil kecepatan dari cmd_vel
        vx = msg.linear.x
        vy = msg.linear.y
        wz = msg.angular.z

        # Log input kecepatan
        rospy.loginfo("Received cmd_vel - linear: [%.2f, %.2f], angular: %.2f", vx, vy, wz)

        # Kinematika inverse untuk omni wheel 4 roda
        w_fl = self.gain * (1 / self.r) * (vx - vy - (self.l * wz))
        w_fr = self.gain * (1 / self.r) * (vx + vy + (self.l * wz))
        w_bl = self.gain * (1 / self.r) * (vx + vy - (self.l * wz))
        w_br = self.gain * (1 / self.r) * (vx - vy + (self.l * wz))

        # Log kecepatan roda yang dihitung
        rospy.loginfo("Wheel efforts - FL: %.2f, FR: %.2f, BL: %.2f, BR: %.2f", w_fl, w_fr, w_bl, w_br)

        # Kirim effort ke masing-masing roda
        self.pub_fl.publish(w_fl)
        self.pub_fr.publish(w_fr)
        self.pub_bl.publish(w_bl)
        self.pub_br.publish(w_br)

if __name__ == '__main__':
    try:
        OmniEffortController()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS Interrupt Exception occurred!")

