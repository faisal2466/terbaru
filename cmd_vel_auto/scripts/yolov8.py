#!/usr/bin/env python3

import rospy
import cv2
import os
import torch
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from ultralytics import YOLO

class YoloV8Controller:
    def __init__(self):
        rospy.init_node("yolov8_controller_node", anonymous=True)

        self.bridge = CvBridge()

        # Load YOLOv8 model (force to CPU)
        model_path = os.path.expanduser('~/catkin/src/cmd_vel_auto/scripts/best.pt')
        self.model = YOLO(model_path)
        self.model.fuse()  # ✅ Optimalisasi: fuses Conv + BatchNorm
        self.model.to("cpu")  # ✅ Paksa pakai CPU

        # ROS interfaces
        self.image_sub = rospy.Subscriber("/final/camera1/image_raw", Image, self.image_callback, queue_size=1)
        self.image_pub = rospy.Publisher("/yolov8/detections/image", Image, queue_size=1)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        rospy.loginfo("✅ YOLOv8 Controller Node (CPU optimized) started.")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return

        # ✅ Nonaktifkan tracking, percepat inferensi
        results = self.model.predict(frame, verbose=False, conf=0.3, iou=0.5)[0]

        h, w, _ = frame.shape
        twist = Twist()
        detected = False

        for box in results.boxes:
            cls = int(box.cls[0])
            name = self.model.names[cls]

            if name == "bolla":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Anotasi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Kendali
                if cx < w // 3:
                    twist.linear.x = 0.0
                    twist.angular.z = -1.5
                elif cx > 2 * w // 3:
                    twist.linear.x = 0.0
                    twist.angular.z = 1.5
                else:
                    twist.linear.x = 10.0
                    twist.angular.z = 0.0

                detected = True
                break

        if not detected:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.image_pub.publish(image_msg)


if __name__ == '__main__':
    try:
        YoloV8Controller()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

