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
import random
import PIL

import sys
sys.path.append('/workspace/ros_mamba_ws/ivanModel')

# from shuttlebusTrain import ViM
from new_resnet_model import PPGeoNavModelGray, build_model_for_eglinton_gray



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

class ppgeo_publisher(Node):
    def __init__(self):
        super().__init__('nn_ppgeo_publisher')
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
        self.lane_following_cmd0 = "ResNet34_shuttlebus_imagenet_unfrozen_lane_following_finetune_1.0.pth"
        self.model_path = os.path.join(Model_Path, self.lane_following_cmd0)
        self.lane_following_cmd1 = "ResNet34_shuttlebus_custom_ppgeo_unfrozen_lane_following_finetune_1.0.pth"
        self.model_pullin_path = os.path.join(Model_Path, self.lane_following_cmd1)
        self.lane_following_cmd2 = "ResNet34_shuttlebus_ppgeo_partial_lane_following_finetune_1.0.pth"
        self.model_reverse_path = os.path.join(Model_Path, self.lane_following_cmd2)
        self.lane_following_cmd3 = "ResNet34_shuttle_custom_ppgeo_frozen_lane_following_finetune_1.0_40_0.0021_0.4455_0.2174.pth"
        self.model_dual_steering_path = os.path.join(Model_Path, self.lane_following_cmd3)
        # print(sys.path)
        self.model = build_model_for_eglinton_gray(
                pretrain_type="scratch",   # important: we're restoring from your .pth, not re-pretraining here
                freeze_mode="unfrozen",
                normalize=False
            ).to(device)
        # print(self.model)
        self.model_pullin =  build_model_for_eglinton_gray(
                pretrain_type="scratch",   # important: we're restoring from your .pth, not re-pretraining here
                freeze_mode="unfrozen",
                normalize=False
            ).to(device)

        self.model_reverse = build_model_for_eglinton_gray(
            pretrain_type="scratch",   # important: we're restoring from your .pth, not re-pretraining here
            freeze_mode="unfrozen",
            normalize=False
        ).to(device) 

        self.model_dual_steering = re

        self.nn_linear=0.1
        self.nn_angular=0.0
        self.driving_mode=7
        self.speed_multi=4.0
        
        self.model.load_state_dict(torch.load(self.model_path)['model_state_dict'])
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
            f"cmd7: {self.lane_following_cmd0}",
            f"cmd8: {self.lane_following_cmd1}",
            f"cmd9: {self.lane_following_cmd2}",
        ]
        nn_model_names_msg = String()
        nn_model_names_msg.data = ",".join(nn_model_names_list)
        self.publisher5.publish(nn_model_names_msg)

        #scene_cat node 0: empty bay 1: roundabout give away
        self.timer = self.create_timer(1/6, self.publish_msg)

        self.pre_img = np.zeros([1,240,400])
        self.cur_img = np.zeros([1,240,400])

        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)
        if device == "cuda":
            torch.cuda.manual_seed_all(1234)

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision('high')


    def img_listener_callback(self, msg):
        #print("Got something!")
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
        img=cv2.resize(cv_img[104:,:],(480,240))
            
        cropped_img = img[60:, 40:440]
        cropped_img = PIL.Image.fromarray(np.uint8(cropped_img), mode='L')

        cropped_img = appliedTransform(cropped_img)
        cropped_img = rearrange(cropped_img, "c h w -> 1 c h w")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        img = cropped_img.to(device)
        
        self.cur_img = img
        speed = 0.0
        steering_angle = 0.0

        with torch.inference_mode(), torch.autocast('cuda', enabled=False):

            if self.driving_mode == 7:
    
                speed, steering_angle = self.model(img)
                speed = speed.item() * self.speed_multi
                steering_angle = steering_angle.item() * 0.3
                # print(speed)
                #self.get_logger().info("CV Image resize_h %d" % (resize_h))
                self.get_logger().info(f"speed: {speed}, steering: {steering_angle}")
                self.nn_linear = speed
                self.nn_angular = steering_angle
            if self.driving_mode == 8:
    
                speed, steering_angle = self.model_pullin(img)
                speed = speed.item() * self.speed_multi
                steering_angle = steering_angle.item() * 0.3
                # print(speed)
                #self.get_logger().info("CV Image resize_h %d" % (resize_h))
                self.get_logger().info(f"speed: {speed}, steering: {steering_angle}")
                self.nn_linear = speed
                self.nn_angular = steering_angle
            if self.driving_mode == 9:
    
                speed, steering_angle = self.model_reverse(img)
                speed = speed.item() * self.speed_multi
                steering_angle = steering_angle.item() * 0.3
                # print(speed)
                #self.get_logger().info("CV Image resize_h %d" % (resize_h))
                self.get_logger().info(f"speed: {speed}, steering: {steering_angle}")
                self.nn_linear = speed
                self.nn_angular = steering_angle
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
            self.speed_multi=2.0
        else:
            self.speed_multi=3.0

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
    nn_cmd_publisher = ppgeo_publisher()
    rclpy.spin(nn_cmd_publisher)

    nn_cmd_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()