#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import os
import time
import csv
from datetime import datetime

class YoloV8Detector:
    def __init__(self):
        rospy.init_node("yolov8_detector", anonymous=True)

        self.bridge = CvBridge()
        model_path = os.path.expanduser('~/catkin/src/cmd_vel_auto/scripts/best.pt')
        self.model = YOLO(model_path)

        self.image_sub = rospy.Subscriber("/final/camera1/image_raw", Image, self.image_callback)
        self.image_pub = rospy.Publisher("/yolov8/detections/image", Image, queue_size=1)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        self.detected_before = False
        self.start_time = None
        self.log_count = 1

        self.log_file = os.path.expanduser("~/catkin/src/cmd_vel_auto/deteksi_waktu.csv")
        self.ensure_log_file_exists()

        rospy.loginfo("✅ YOLOv8 ROS node with timer and logger started.")

    def ensure_log_file_exists(self):
        # Buat folder kalau belum ada
        if not os.path.exists(os.path.dirname(self.log_file)):
            os.makedirs(os.path.dirname(self.log_file))
        # Buat file dan tulis header kalau belum ada
        if not os.path.isfile(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['No', 'Start Time', 'End Time', 'Duration (s)'])

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return

        results = self.model(frame)[0]
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

                # Gambar kotak dan titik tengah
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Kontrol robot berdasarkan posisi bola
                if cx < w // 3:
                    twist.linear.x = 0.0
                    twist.angular.z = -1.5
                elif cx > 2 * w // 3:
                    twist.linear.x = 0.0
                    twist.angular.z = 1.5
                else:
                    twist.linear.x = 7.0
                    twist.angular.z = 0.0

                detected = True
                break

        if detected and not self.detected_before:
            self.start_time = time.time()
            rospy.loginfo("🎯 Bola terdeteksi! Timer dimulai.")
            self.detected_before = True
            self.image_pub.publish(image_msg)

        elif not detected and self.detected_before:
            end_time = time.time()
            duration = end_time - self.start_time
            start_str = datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S")
            end_str = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")

            # Tulis ke CSV sesuai kolom yang benar
            with open(self.log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.log_count,    # No
                    start_str,         # Start Time
                    end_str,           # End Time
                    f"{duration:.2f}"  # Duration (s)
                ])
            self.log_count += 1

            rospy.loginfo(f"⏱️ Bola tidak terdeteksi lagi. Durasi: {duration:.2f} detik.")
            self.detected_before = False
            self.start_time = None

        if not detected:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.image_pub.publish(image_msg)


if __name__ == '__main__':
    try:
        YoloV8Detector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
