#!/usr/bin/env python3

import rospy
import cv2
import os
import torch
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class YoloV5BallDetector:
    def __init__(self):
        rospy.init_node("yolov5_ball_detector", anonymous=True)

        self.bridge = CvBridge()

        # Ambil path model dari parameter ROS, default di bawah ini, lalu expand ~ jadi full path
        model_path = rospy.get_param("~weights", "~/catkin/src/cmd_vel_auto/scripts/best.pt")
        model_path = os.path.expanduser(model_path)

        # Load model YOLOv5 custom, paksa pakai CPU
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        self.model.to("cpu")
        self.model.eval()

        # ROS subscriber dan publisher
        self.image_sub = rospy.Subscriber("/final/camera1/image_raw", Image, self.image_callback, queue_size=1)
        self.image_pub = rospy.Publisher("/yolov5/detections/image", Image, queue_size=1)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        rospy.loginfo("✅ YOLOv5 Ball Detector Node started.")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"CV Bridge error: {e}")
            return

        # Jalankan deteksi
        results = self.model(frame)
        detections = results.xyxy[0]  # tensor [x1, y1, x2, y2, conf, cls]

        h, w, _ = frame.shape
        twist = Twist()
        detected = False

        for *box, conf, cls in detections:
            name = self.model.names[int(cls)]
            if name == "bolla":  # Ganti sesuai nama kelas pada model custom-mu
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Gambar bounding box dan pusat deteksi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Kendali robot berdasarkan posisi bola di frame
                if cx < w // 3:
                    twist.linear.x = 0.0
                    twist.angular.z = -1.5  # belok kiri
                elif cx > 2 * w // 3:
                    twist.linear.x = 0.0
                    twist.angular.z = 1.5  # belok kanan
                else:
                    twist.linear.x = 0.5  # maju
                    twist.angular.z = 0.0

                detected = True
                break  # hanya fokus satu bola

        if not detected:
            # Stop robot jika bola tidak terdeteksi
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        # Publish perintah gerak
        self.cmd_pub.publish(twist)

        # Publish gambar hasil deteksi
        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.image_pub.publish(image_msg)


if __name__ == '__main__':
    try:
        node = YoloV5BallDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

