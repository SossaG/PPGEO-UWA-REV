import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torchvision import datasets, models, transforms

from std_msgs.msg import Float32, Int32MultiArray, String, Int32, Float32MultiArray
from python_nodes_interfaces.msg import BoolStamped, Int32Stamped, Float32MultiArrayStamped#, StringMultiArrayStamped
from geometry_msgs.msg import Twist, TwistStamped
from cv_bridge import CvBridge
import os
import time

import sys
sys.path.append('/workspace/ros_mamba_ws/visionMambaPaper')

from shuttlebusTrain import ViM


bridge = CvBridge()
#driving_mode node 0: manual 1: GPS 2: lane following 3: pullin 4: reverse 5: dual steering
# Script_Path = os.path.dirname(os.path.abspath(__file__))
Model_Path = "/workspace/ros_mamba_ws/models"

appliedTransform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Resize((66, 200)),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        
    ]
)

class vim_cmd_publisher(Node):
    def __init__(self):
        super().__init__('nn_vim_publisher')
        self.img_subscription = self.create_subscription(
            Image,
            "/CameraFront",
            self.img_listener_callback,
            10)
        self.mode_subscription = self.create_subscription(
            Int32Stamped,
            "/driving_mode_stamped",
            self.mode_listener_callback,
            10)
        self.speed_mode_subscription = self.create_subscription(
            Int32Stamped,
            "speed_mode_stamped",
            self.speed_mode_listener_callback,
            10)
        self.img_subscription  # prevent unused variable warning
        self.mode_subscription  # prevent unused variable warning
        self.lane_following_cmd0 = "vim_shuttle_lane_following.pth"
        self.model_path = os.path.join(Model_Path, self.lane_following_cmd0)
        self.lane_following_cmd1 = "vim_shuttle_pullin.pth"
        self.model_pullin_path = os.path.join(Model_Path, self.lane_following_cmd1)
        self.lane_following_cmd2 = "vim_shuttle_reverse.pth"
        self.model_reverse_path = os.path.join(Model_Path, self.lane_following_cmd2)
        self.lane_following_cmd3 = "vim_shuttle_lane_following.pth"
        self.model_dual_steering_path = os.path.join(Model_Path, self.lane_following_cmd3)
        # print(sys.path)
        self.model = ViM(
            img_size = (160, 320),
            patch_size = (20, 20), #8, #(11, 10)
            # stride=(10, 20), #(5, 5),
            # stride=() # smaller for more detail (5, 5)
            num_classes = 2,
            dim = 1024, #1000, #2
            depth = 8, #24, #4-8
            channels=1,
            dropout = 0.1,
            # emb_dopout = 0.1,
            embed_dim=512, #48, #512
            d_state=16, #40, #16
        )
        # print(self.model)
        self.model_pullin = ViM(
            img_size = (160, 320),
            patch_size = (20, 20), #8, #(11, 10)
            # stride=(10, 20), #(5, 5),
            # stride=() # smaller for more detail (5, 5)
            num_classes = 2,
            dim = 1024, #1000, #2
            depth = 8, #24, #4-8
            channels=1,
            dropout = 0.1,
            # emb_dopout = 0.1,
            embed_dim=512, #48, #512
            d_state=16, #40, #16
        )

        self.model_reverse = ViM(
            img_size = (160, 320),
            patch_size = (20, 20), #8, #(11, 10)
            # stride=(10, 20), #(5, 5),
            # stride=() # smaller for more detail (5, 5)
            num_classes = 2,
            dim = 1024, #1000, #2
            depth = 8, #24, #4-8
            channels=1,
            dropout = 0.1,
            # emb_dopout = 0.1,
            embed_dim=512, #48, #512
            d_state=16, #40, #16
        )

        self.model_dual_steering = ViM(
            img_size = (160, 320),
            patch_size = (20, 20), #8, #(11, 10)
            # stride=(10, 20), #(5, 5),
            # stride=() # smaller for more detail (5, 5)
            num_classes = 2,
            dim = 1024, #1000, #2
            depth = 8, #24, #4-8
            channels=1,
            dropout = 0.1,
            # emb_dopout = 0.1,
            embed_dim=512, #48, #512
            d_state=16, #40, #16
        )

        self.nn_linear=0.1
        self.nn_angular=0.0
        self.driving_mode=0
        self.speed_multi=4.0
        
        self.model.load_state_dict(torch.load(self.model_path, weights_only=True)['model_state_dict'])
        self.model_pullin.load_state_dict(torch.load(self.model_pullin_path, weights_only=True)['model_state_dict'])
        self.model_reverse.load_state_dict(torch.load(self.model_reverse_path, weights_only=True)['model_state_dict'])
        self.model_dual_steering.load_state_dict(torch.load(self.model_dual_steering_path, weights_only=True)['model_state_dict'])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print(self.model.state_dict())
        self.model.eval()
        self.model_pullin.eval()
        self.model_reverse.eval()
        self.model_dual_steering.eval()
        self.model.to(device)
        self.model_pullin.to(device)
        self.model_reverse.to(device)
        self.model_dual_steering.to(device)

        self.publisher1 = self.create_publisher(Twist, "nn_cmd_vel", 10)
        self.publisher2 = self.create_publisher(TwistStamped, "nn_cmd_vel_stamped", 10)
        self.publisher3 = self.create_publisher(Float32MultiArrayStamped, "scene_cat_stamped", 10)
        self.publisher4 = self.create_publisher(String, "camera_node_info", 10)
        self.publisher5 = self.create_publisher(String, "nn_model_names", 10)
        nn_model_names_list = [
            f"cmd lan_follow: {self.lane_following_cmd0}",
            f"cmd pullin: {self.lane_following_cmd1}",
            f"cmd reverse: {self.lane_following_cmd2}",
        ]
        nn_model_names_msg = String()
        nn_model_names_msg.data = ",".join(nn_model_names_list)
        self.publisher5.publish(nn_model_names_msg)

        #scene_cat node 0: empty bay 1: roundabout give away
        self.timer = self.create_timer(1/6, self.publish_msg)

        self.pre_img = np.zeros([1,240,400])
        self.cur_img = np.zeros([1,240,400])


    def img_listener_callback(self, msg):
        #print("Got something!")
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        cv_img_h, cv_img_w = cv_img.shape
        #print(cv_img.shape)
        resize_h=int(cv_img_h*0.2)
        img=cv2.resize(cv_img[resize_h:,:],(480,240))
        img=img[80:, 80:400]
        img= np.expand_dims(img, axis=0)
        img = img.astype('float32')
        # self.get_logger().info("image shape")
        # print(img.shape)

        img = appliedTransform(img)
        img = rearrange(img, 'w c h -> 1 c h w')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        img = img.to(device)
        
        self.cur_img = img
        speed = 0.0
        steering_angle = 0.0

        if self.driving_mode == 7:

            speed, steering_angle = self.model(img)
            speed = speed.item() * self.speed_multi
            steering_angle = steering_angle.item() * 0.3
            # print(speed)
            #self.get_logger().info("CV Image resize_h %d" % (resize_h))
            self.get_logger().info(f"speed: {speed}, steering: {steering_angle}")
        if self.driving_mode == 8:

            speed, steering_angle = self.model_pullin(img)
            speed = speed.item() * self.speed_multi
            steering_angle = steering_angle.item() * 0.3
            # print(speed)
            #self.get_logger().info("CV Image resize_h %d" % (resize_h))
            self.get_logger().info(f"speed: {speed}, steering: {steering_angle}")
        if self.driving_mode == 9:

            speed, steering_angle = self.model_reverse(img)
            speed = speed.item() * self.speed_multi
            steering_angle = steering_angle.item() * 0.3
            # print(speed)
            #self.get_logger().info("CV Image resize_h %d" % (resize_h))
            self.get_logger().info(f"speed: {speed}, steering: {steering_angle}")
        if self.driving_mode == 10:

            speed, steering_angle = self.model_dual_steering(img)
            speed = speed.item() * self.speed_multi
            steering_angle = steering_angle.item() * 0.3
            # print(speed)
            #self.get_logger().info("CV Image resize_h %d" % (resize_h))
            self.get_logger().info(f"speed: {speed}, steering: {steering_angle}")

        self.nn_linear = speed
        self.nn_angular = steering_angle
        

    def mode_listener_callback(self, msg):
        #print("Got something!")
        self.driving_mode=msg.data


    def speed_mode_listener_callback(self, msg):
        #print("Got something!")
        self.speed_mode=msg.data
        if self.speed_mode==0:
            self.speed_multi=4.0
        else:
            self.speed_multi=5.4

    def publish_msg(self):
        msg = Twist()
        # print(self.nn_linear)
        # scene_cat_msg=Float32MultiArrayStamped()
        if False:# not np.any(self.cur_img - self.pre_img): 
            # image is dumb as current image equals to previous image, stop the bus
            self.get_logger().info("Same image, set speed to 0")
            msg.linear.x=0.0
            msg.angular.z=0.0
            self.camera_text_info = "Camera frozen"
        else:
        # self.get_logger().info("Camera is working")
        # self.camera_text_info = "Camera working"
            msg.linear.x=self.nn_linear
            msg.angular.z=self.nn_angular
        self.pre_img = self.cur_img
        self.publisher1.publish(msg)
        msg2=TwistStamped()
        msg2.twist=msg
        msg2.header.stamp=rclpy.clock.Clock().now().to_msg()
        self.publisher2.publish(msg2)

def main(args=None):
    rclpy.init(args=args)
    nn_cmd_publisher = vim_cmd_publisher()
    rclpy.spin(nn_cmd_publisher)

    nn_cmd_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()