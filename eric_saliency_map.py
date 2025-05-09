#!/usr/bin/env python
import numpy as np
import os
import cv2 
from os.path import join, exists, dirname, abspath, split
import shutil
import math
import time

# Constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
FILE_TEXT_POSITION = (50,200)
FONT_SCALE = 0.6
FONT_COLOR = (250)
FONT_COLOR2 = (50)
LINE_TYPE = 1
LINE_TYPE2 = LINE_TYPE+1
START_POINT = (200, 230)
end_point = (200, 190)
ACTUAL_START_POINT = (200-10, 230)
PREDICTION_START_POINT = (200+10, 230)
actual_end_point = (200, 190)
prediction_end_point = (200, 190)

LINE_MAX_LEN = 60
LINE_MAX_ANGLE = np.pi/6
#positive left
ACTUAL_COLOR = (200)
PREDICTION_COLOR = (250)
THICKNESS = 4
THICKNESS2 = THICKNESS +1

MAP_WIDTH_PIX=15444
MAP_HEIGHT_PIX=5269
MAP_RESOLUTION=8.73
MAP_HEIGHT_METERS=603.55097
MAP_LOCS = {'Flying Fox Park': (1416.8958, MAP_HEIGHT_METERS-300.6873),
        'Amberton Beach': (212.7148, MAP_HEIGHT_METERS-468.8431)
    }   
def Image_Processing(image):
    height, width, _ = image.shape
    image = image[int(height/3):, :, :]
    #image = cv2.resize(image, (200, 66))
    #now is (240 ,480 , H, W)
    #image = image[:, 40:440, :]
    #(240 ,400 , H, W)
    return image
def euler_from_quaternion_z(z, w):
        t3 = +2.0 * (w * z)
        t4 = +1.0 - 2.0 * (z * z)
        yaw_z = math.atan2(t3, t4)
        return yaw_z # in radians

def get_path_dict(base_folder):
    return {subfolder_name: {subsubfolder_name: join(join(base_folder, subfolder_name), subsubfolder_name) for subsubfolder_name in os.listdir(join(base_folder, subfolder_name))} for subfolder_name in os.listdir(base_folder)}

#paths
script_path = dirname(abspath(__file__))
raw_data_path = join(script_path, "raw_eglinton_data")
#sorted_data
sorted_data_path = join(script_path, "sorted_eglinton_data")
#print("getting path dict")
path_dict = get_path_dict(sorted_data_path)
#print("got path dict")
dataset_dict={0: "CIL_Dual_Cam_Stage1", 1: "CIL_Dual_Cam_Stage2", 2: "CIL_Dual_Cam_Stage2_B", 3: "three_roundabouts_front_cam_only", 4: "CIL_Dual_Cam_Stage2_First_Half", 5: "CIL_Dual_Cam_Stage2_Second_Half"}
#trained model
trained_model_path = join(script_path, "Trained_model")
#----------------------------Data Sorting----------------------------------------------------------
class Data_Sorting():
    def __init__(self):
        self.cp = 'Flying Fox Park'
        self.cp_xy_cords = MAP_LOCS[self.cp]
        #dataset_dict{0: "CIL_Dual_Cam_Stage1", 1: "CIL_Dual_Cam_Stage2", 2: "CIL_Dual_Cam_Stage2_B", 3: "three_roundabouts_front_cam_only"}
        self.selected_path_dict = path_dict[dataset_dict[2]]
        self.keep_value = False
        
    def Stage1_Auto_Sorting(self):    # sourcery skip: low-code-quality
        #self.data_folder_path="/media/erik/Linux_ST2/eglinton_img_data"
        #stage 1 filters
        filters_path = join(script_path, "filters")
        filters_dict = {filter_name: np.int32(np.divide(np.array(cv2.imread(join(filters_path, filter_name), cv2.IMREAD_GRAYSCALE)), 255))*255 for filter_name in os.listdir(filters_path)}
        self.data_folder_path=raw_data_path
        self.selected_path_dict = path_dict["CIL_Dual_Cam_Stage1"]
        if not exists(self.data_folder_path):
            return
        for self.bag_folder in sorted(os.listdir(self.data_folder_path)):
            self.folder_path = join(self.data_folder_path, self.bag_folder)
            self.sorted_file_list=sorted(os.listdir(self.folder_path), key=lambda x: int(x.split("_")[1]))
            print(self.folder_path)
            print(len(os.listdir(self.folder_path)))
            self.idx_begin, self.idx_end = 0, 0
            pre_map_pose=(255,255)
            append_flag=False
            linear_buffer, angular_buffer = [], []
            for self.idx in range(len(self.sorted_file_list)):
                linear = self.extract_npy_data()[2]
                angular = self.extract_npy_data()[3]
                map_pose = self.extract_npy_data()[8]
                #print("map_pose: ",map_pose)
                if map_pose is None:
                    map_pose=pre_map_pose
                if (map_pose[0] < MAP_WIDTH_PIX and map_pose[1] < MAP_HEIGHT_PIX) and (pre_map_pose[0] < MAP_WIDTH_PIX and pre_map_pose[1] < MAP_HEIGHT_PIX):
                    #print(map_pose[0],map_pose[1])
                    for filter_name in filters_dict:
                        #print("filter_ shape: ", filter_.shape)
                        filter_=filters_dict[filter_name]
                        if filter_[map_pose[1]][map_pose[0]]==0 and filter_[pre_map_pose[1]][pre_map_pose[0]]==255:
                            #print("Entering filter_")
                            self.idx_begin=self.idx
                            filter_name_begin=filter_name
                            append_flag=True
                        elif filter_[map_pose[1]][map_pose[0]]==255 and filter_[pre_map_pose[1]][pre_map_pose[0]]==0:
                            #print("Exiting filter_")
                            self.idx_end=self.idx
                            filter_name_end=filter_name
                            append_flag=False
                        elif self.idx==(len(self.sorted_file_list)-1) and filter_[map_pose[1]][map_pose[0]]==0 and filter_[pre_map_pose[1]][pre_map_pose[0]]==0:
                            self.idx_end=self.idx
                            filter_name_end=filter_name
                            append_flag=False
                if append_flag:    
                    linear_buffer.append(linear)
                    angular_buffer.append(angular)
                    #orien_buffer.append(orien)
                if ((self.idx_end-self.idx_begin)> 100) and (filter_name_begin==filter_name_end):
                    filter_name_begin_=filter_name_begin.split(".")[0]
                    selected_filtered_data_path=self.selected_path_dict[f"{filter_name_begin_}ed"]
                    self.move_files(selected_filtered_data_path)
                    #reset
                    self.idx_begin, self.idx_end = self.idx, 0
                    linear_buffer, angular_buffer = [], []
                pre_map_pose = map_pose

    def Stage2_Auto_Sorting(self):
        sort_parallel_parking_flag=1
        sort_cul_de_sac_flag=1
        #sort_intersection_flag=0
        #sort_right_hand_roundabout_flag=0
        #sort_small_carpark_flag=0
        sort_standard_carpark_flag=1
        sort_standard_roundabout_flag=1
        data_folder_path=path_dict["CIL_Dual_Cam_Stage1"]["parallel_parking_filtered"]
        if exists(data_folder_path) and sort_parallel_parking_flag:
            for self.bag_folder in sorted(os.listdir(data_folder_path)):
                self.folder_path = join(data_folder_path, self.bag_folder)
                self.sorted_file_list=sorted(os.listdir(self.folder_path), key=lambda x: int(x.split("_")[1]))
                print(self.folder_path)
                print(len(os.listdir(self.folder_path)))
                self.idx, self.new_idx_begin = 0, 0
                self.idx_begin, self.idx_end = 0, None
                forward_flag, reverse_flag, pre_linear = 0, 0, 0

                for self.idx in range(len(self.sorted_file_list)):
                    linear = self.extract_npy_data()[2]
                    if linear>0.5:
                        forward_flag=self.idx
                    if pre_linear>=0 and linear<0:
                        reverse_flag=self.idx
                    if forward_flag < reverse_flag:
                        self.idx_end=self.idx-1
                        self.move_files(self.selected_path_dict["pullin"])
                        #reset
                        #self.idx_begin, self.idx_end = 0, None
                        forward_flag, reverse_flag=self.idx, self.idx
                    elif (self.idx == len(self.sorted_file_list)-1):
                        self.idx_end=self.idx
                        if self.idx_begin==0:
                            self.move_files(self.selected_path_dict["lane_bay_pass"])
                        else:
                            self.move_files(self.selected_path_dict["pullout"])
                    pre_linear=linear
        data_folder_path=path_dict["CIL_Dual_Cam_Stage1"]["cul_de_sac_filtered"]
        if exists(data_folder_path) and sort_cul_de_sac_flag:
            for self.bag_folder in sorted(os.listdir(data_folder_path)):
                self.folder_path = join(data_folder_path, self.bag_folder)
                self.sorted_file_list=sorted(os.listdir(self.folder_path), key=lambda x: int(x.split("_")[1]))
                print(self.folder_path)
                print(len(os.listdir(self.folder_path)))
                self.idx, self.new_idx_begin = 0, 0
                self.idx_begin, self.idx_end = 0, None
                pre_gear=1
                
                for self.idx in range(len(self.sorted_file_list)):
                    gear=self.extract_npy_data()[5]
                    if pre_gear==1 and gear==2:
                        self.idx_begin=self.idx
                    elif pre_gear==2 and gear==1:
                        self.idx_end=self.idx-1
                        self.move_files(self.selected_path_dict["cul_de_sac_dualsteering"])
                    pre_gear=gear
        data_folder_path=path_dict["CIL_Dual_Cam_Stage1"]["standard_carpark_filtered"]
        if exists(data_folder_path) and sort_standard_carpark_flag: 
            for self.bag_folder in sorted(os.listdir(data_folder_path)):
                self.folder_path = join(data_folder_path, self.bag_folder)
                self.sorted_file_list=sorted(os.listdir(self.folder_path), key=lambda x: int(x.split("_")[1]))
                print(self.folder_path)
                print(len(os.listdir(self.folder_path)))
                self.idx, self.new_idx_begin = 0, 0
                self.idx_begin, self.idx_end = 0, None
                pre_gear=1

                if len(self.sorted_file_list)<400:
                    self.move_files(self.selected_path_dict["carpark_pass"],move_folder_flag=True)
                else:        
                    for self.idx in range(len(self.sorted_file_list)):
                        gear=self.extract_npy_data()[5]
                        if pre_gear==1 and gear==2:
                            self.idx_begin=self.idx
                        elif pre_gear==2 and gear==1:
                            self.idx_end=self.idx-1
                            self.move_files(self.selected_path_dict["carpark_dualsteering"])
                        pre_gear=gear
        data_folder_path=path_dict["CIL_Dual_Cam_Stage1"]["standard_roundabout_filtered"]
        if exists(data_folder_path) and sort_standard_roundabout_flag: 
            roundabout_to_beach_point_list=[(7803,3184),(13318,2169),(10616,2933)]
            roundabout_to_office_point_list=[(8163,2972),(11000,2771),(13603,1873)]
            for self.bag_folder in sorted(os.listdir(data_folder_path)):
                self.folder_path = join(data_folder_path, self.bag_folder)
                self.sorted_file_list=sorted(os.listdir(self.folder_path), key=lambda x: int(x.split("_")[1]))
                print(self.folder_path)
                print(len(os.listdir(self.folder_path)))
                self.idx, self.new_idx_begin = 0, 0
                self.idx_begin, self.idx_end = 0, None
                pre_map_pose=(0,0)
                angular_buffer=[]
                for self.idx in range(len(self.sorted_file_list)):
                    angular=self.extract_npy_data()[3]
                    map_pose=self.extract_npy_data()[8]
                    if map_pose is None:
                        map_pose=pre_map_pose
                    angular_buffer.append(angular)
                    pre_map_pose=map_pose
                if np.mean(np.sort(np.partition(angular_buffer, 50)[:50]))>-0.7:
                    self.move_files(self.selected_path_dict["roundabout_straight"],move_folder_flag=True)
                else:
                    for roundabout_to_beach_point in roundabout_to_beach_point_list:
                        if math.dist(map_pose,roundabout_to_beach_point)<100:
                            self.move_files(self.selected_path_dict["roundabout_turn_around_to_beach"],move_folder_flag=True)
                    for roundabout_to_office_point in roundabout_to_office_point_list:
                        if math.dist(map_pose,roundabout_to_office_point)<100:
                            self.move_files(self.selected_path_dict["roundabout_turn_around_to_office"],move_folder_flag=True)
    def Section_Auto_Sorting(self):
        pass
    def manual_manipulating(self, mode_idx=4):
        mani_mode_list=["Sorting","Modifying","Comparing","Classifying","Saliency"]
        mani_mode=mani_mode_list[mode_idx]
        self.selected_path_dict = path_dict[dataset_dict[4]]
        self.data_folder_path=self.selected_path_dict["lane_following"] 
        #self.data_folder_path="/media/erik/Linux_Data/eglinton_data_sorting_dual/sorted_eglddinton_data/CIL_Dual_Cam_Stage1/main_filtered"
        print(self.data_folder_path) 
        if not exists(self.data_folder_path):
            return
        sorted_folder_list = sorted(os.listdir(self.data_folder_path))
        step, f_idx = 100, 0  # folder index 
        key_path_mapping = {
            ord('k'): "bad_behaviors_to_delete",
            #ord('k'): "reverse_bad_behaviors_to_delete",
            #ord('k'): "reverse_manual",
            #ord('k'): "tmp_folder_to_second_half",
            #ord('k'): "pullout_bad_behaviors_to_delete",
            ord('y'): "lane_following",
            ord('u'): "lane_bay_pass",
            ord('t'): "lane_empty_bay",
            ord('i'): "pullin",
            ord('I'): "pullin_stops",
            ord('-'): "reverse",
            ord('_'): "reverse_manual",
            ord('o'): "pullout",
            ord('O'): "pullout_stops",
            ord('p'): "carpark_pass",
            ord('['): "startpoint_in",
            ord(']'): "startpoint_out",
            ord('f'): "roundabout_turn_around_to_beach",
            ord('g'): "roundabout_straight",
            ord('h'): "roundabout_turn_around_to_office",
            ord('c'): "roundabout_right_turn",
            ord(';'): "shed_in",
            ord("'"): "shed_out",
            ord('x'): "others",
            ord('b'): "carpark_dual_steering",
            ord('n'): "carpark_entry",
            ord('='): "lane_empty_bay_first_half",
            ord('v'): "lane_empty_bay_second_half",
            ord(','): "intersection_lane_following",
            ord('.'): "intersection_turn_around_to_beach",
            ord('/'): "intersection_turn_around_to_office",
            ord('z'): "cul_de_sac_dualsteering",
        }
        if mani_mode=="Comparing":
            import tensorflow as tf
            state = 0
            interpreter, output_0, output_1, input_0 = [], [], [], []
            tflite_model_list = [
                join(script_path, "saved_models/old_models/BW_Scan_B1_Cmd0_Single_2025-03-20-20.55.46/PilotNet_BW_Scan_B1_Cmd0_2025-03-21-07.16.37.tflite"),
                join(script_path, "saved_models/old_models/BW_Scan_B1_Cmd1_Single_2025-03-21-16.54.27/PilotNet_BW_Scan_B1_Cmd1_2025-03-22-01.01.26.tflite"),
                join(script_path, "saved_models/old_models/BW_Scan_B1_Cmd2_Single_2025-03-21-11.15.26/PilotNet_BW_Scan_B1_Cmd2_2025-03-21-16.32.37.tflite"),
                #join(script_path, "saved_models/old_models/BW_Scan_B1_Cmd1_Single_2025-03-21-07.16.50/PilotNet_BW_Scan_B1_Cmd1_2025-03-21-11.15.13.tflite")
                join(script_path, "saved_models/SGrayResPilotNetv0.0_2025-03-25-15.41.07/SGrayResPilotNetv0.0_cmd_0_2025-03-25-15.43.42.tflite")
            ]
            with tf.device('/device:GPU:0'):
                for model in tflite_model_list:
                    interp = tf.lite.Interpreter(model)
                    interp.allocate_tensors()
                    interpreter.append(interp)
                    output_0.append(interp.get_output_details()[0])
                    output_1.append(interp.get_output_details()[1])
                    input_0.append(interp.get_input_details()[0])
                    #print(f"input_0: {interp.get_input_details()[0]}")
        elif mani_mode=="Classifying":
            import tensorflow as tf
            model_path = join(trained_model_path, "BW_Empty_Bay_A0_Single_2025-02-18-20.34.47/PilotNet_BW_Empty_Bay_A0_2025-02-18-20.47.22.tflite")
            interpreter = tf.lite.Interpreter(model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
        elif mani_mode=="Saliency":
            import tensorflow as tf
            from tf_keras_vis.saliency import Saliency
            from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
            from tf_keras_vis.utils.scores import Score
            state = 0
            interpreter= []
            keras_model_list = [
                join(script_path, "saved_models_logs/old_models/BW_Scan_B1_Cmd0_Single_2025-03-20-20.55.46/BW_Scan_B1_Cmd0_2025-03-20-20.55.46.keras"),
                join(script_path, "saved_models_logs/old_models/BW_Scan_B1_Cmd1_Single_2025-03-21-16.54.27/BW_Scan_B1_Cmd1_2025-03-21-16.54.27.keras"),
                join(script_path, "saved_models_logs/old_models/BW_Scan_B1_Cmd2_Single_2025-03-21-11.15.26/BW_Scan_B1_Cmd2_2025-03-21-11.15.26.keras"),
                #join(script_path, "saved_models_logs/old_models/BW_Scan_B1_Cmd1_Single_2025-03-21-07.16.50/BW_Scan_B1_Cmd1_2025-03-21-07.16.50.keras")
                join(script_path, "saved_models_logs/old_models/BW_Scan_B1_Cmd0_Single_2025-03-20-20.55.46/BW_Scan_B1_Cmd0_2025-03-20-20.55.46.keras"),
            ]

            # --- Define score function for the selected output ---
            class OutputScore(Score):
                def __call__(self, output):
                    return tf.reduce_mean(output)


            # --- Modify model output for visualization ---
            modifier = ReplaceToLinear()

            score_function = [OutputScore(),OutputScore()]
            with tf.device('/device:GPU:0'):
                for model in keras_model_list:
                    interp = tf.keras.models.load_model(model, compile=False)
                    # --- Create Saliency object ---
                    saliency = Saliency(interp, model_modifier=modifier, clone=True)
                    interpreter.append([interp,saliency])

        while f_idx < len(sorted_folder_list):
            self.bag_folder = sorted_folder_list[f_idx]
            self.folder_path = join(self.data_folder_path, self.bag_folder)
            self.sorted_file_list=sorted(os.listdir(self.folder_path), key=lambda x: int(x.split("_")[1]))
            print(self.folder_path)
            print(len(os.listdir(self.folder_path)))

            if not self.sorted_file_list:
                os.removedirs(self.folder_path)
                continue

            self.idx, self.new_idx_begin, move_file = 0, 0, 1
            modified_linear, modified_angular = -1, 0
            modified_linear_begin, modified_angular_begin = -1, 0
            modified_linear_end, modified_angular_end = 0, 0
            self.idx_begin, self.idx_end = 0, None
            last_time, last_linear, stop_flag, self.keep_value = time.time(), 0, 0, False
            
            key=cv2.waitKey()
            if key==ord(' '): 
                break

            while move_file:
                start_time = time.time()
                (img_front, img_rear, linear, angular, driving_mode, gear, speed_mode, orien, map_pose, 
                 modified_linear, modified_angular, file) = self.extract_npy_data(modified_linear, modified_angular)
                if mani_mode=="Comparing":
                    image=img_front
                    img_input = np.expand_dims(image, axis=0).astype('float32')
                    #img_input = np.expand_dims(img_input, axis=0).astype('float32')
                    with tf.device('/device:GPU:0'):
                        interpreter[state].set_tensor(input_0[state]['index'], img_input)
                        interpreter[state].invoke()
                        nn_linear=float(interpreter[state].get_tensor(output_1[state]['index']))
                        nn_angular=float(interpreter[state].get_tensor(output_0[state]['index']))
                    #print("--- %s seconds ---" % (time.time() - start_time))            
                    #print("self.nn_linear: ",nn_linear)
                    #print("self.nn_angular: ",nn_angular)
                elif mani_mode=="Classifying":
                    image=img_front
                    img_input = np.expand_dims(image, axis=0).astype('float32')
                    interpreter.set_tensor(input_details[0]['index'], img_input)
                    interpreter.invoke()
                    output_data = float(interpreter.get_tensor(output_details[0]['index']))
                    #print("output_data: ",output_data)
                elif mani_mode=="Saliency":
                    image=img_front
                    img_input = np.expand_dims(image, axis=-1).astype('float32')
                    img_input = np.expand_dims(img_input, axis=0).astype('float32')
                    with tf.device('/device:GPU:0'):
                        saliency_map = interpreter[state][1](score_function, img_input)
                        saliency_map = np.uint8(saliency_map[0] * 255)
                        saliency_map = cv2.resize(saliency_map, (img_front.shape[1], img_front.shape[0]))
                        saliency_map = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
                        img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                        saliency_map = cv2.addWeighted(img_color, 0.5, saliency_map, 0.5, 0)
                        nn_linear = float(interpreter[state][0].predict(img_input)[0])
                        nn_angular = float(interpreter[state][0].predict(img_input)[1]) 

                text_data = [
                    ("step: ", step, (50, 20)),
                    ("idx_begin: ", self.idx_begin, (50, 40)),
                    ("idx_end: ", self.idx_end, (50, 60)),
                    ("driving_mode: ", driving_mode, (50, 80)),
                    ("gear: ", gear, (200, 40)),
                    ("speed_mode: ", speed_mode, (200, 20)),
                    ("file idx: ", file.split("_")[1], (50, 100)),
                    #("file: ", file, FILE_TEXT_POSITION),
                    ("pose: ", map_pose, (50, 120)),
                    ("orien: ", orien, (50, 140)),
                    #("linear: ", round(linear, 3), (50, 180)),
                    #("angular: ", round(angular, 3), (50, 200)),
                    #("linear2: ", modified_linear, (50, 220)),
                    #("angular2: ", modified_angular, (50, 240)),
                ]
                for text, value, pos in text_data:
                    img_front = self.text_on_image(img_front, f"{text}{value}", pos)
                #img_front=self.text_on_image(img_front, file, FILE_TEXT_POSITION)
                img_front_=np.concatenate((img_front, np.zeros((60, img_front.shape[1]),dtype=img_front.dtype)), axis=0)
                if mani_mode=="Modifying":
                    text_modify_mode = f"mdfy_sp: {round(modified_linear, 3)} mdfy_ag: {round(modified_angular, 3)}"
                    img_front_=self.text_on_image(img_front_, text_modify_mode, (50,280))
                    actual_end_point=(tuple(map(lambda x, y: round(x) - round(y), ACTUAL_START_POINT, (LINE_MAX_LEN*linear*np.sin(LINE_MAX_ANGLE*np.sign(linear)*angular),LINE_MAX_LEN*linear*np.cos(LINE_MAX_ANGLE*np.sign(linear)*angular)))))
                    img_front_ = self.line_on_image(img_front_, ACTUAL_START_POINT, actual_end_point)
                    prediction_end_point=(tuple(map(lambda x, y: round(x) - round(y), PREDICTION_START_POINT, (LINE_MAX_LEN*modified_linear*np.sin(LINE_MAX_ANGLE*np.sign(modified_linear)*modified_angular),LINE_MAX_LEN*modified_linear*np.cos(LINE_MAX_ANGLE*np.sign(modified_linear)*modified_angular)))))
                    img_front_ = self.line_on_image(img_front_, PREDICTION_START_POINT, prediction_end_point)
                    #img_front_ = self.vel_measurements_on_image(img_front_, 5)
                    img_front_ = self.hori_measurements_on_image(img_front_, 10 )
                elif mani_mode=="Comparing":
                    text_nn_mode = f"state {state} nn_sp: {round(nn_linear, 3)} nn_ag: {round(nn_angular, 3)}"
                    img_front_=self.text_on_image(img_front_, text_nn_mode, (20,280))
                    #actual_end_point=(tuple(map(lambda x, y: round(x) - round(y), ACTUAL_START_POINT, (LINE_MAX_LEN*linear*np.sin(LINE_MAX_ANGLE*angular),LINE_MAX_LEN*linear*np.cos(LINE_MAX_ANGLE*angular)))))
                    actual_end_point=(tuple(map(lambda x, y: round(x) - round(y), ACTUAL_START_POINT, (LINE_MAX_LEN*modified_linear*np.sin(LINE_MAX_ANGLE*np.sign(modified_linear)*modified_angular),LINE_MAX_LEN*modified_linear*np.cos(LINE_MAX_ANGLE*np.sign(modified_linear)*modified_angular)))))
                    img_front_ = self.line_on_image(img_front_, ACTUAL_START_POINT, actual_end_point)
                    prediction_end_point=tuple(map(lambda x, y: round(x) - round(y), PREDICTION_START_POINT, (LINE_MAX_LEN*nn_linear*np.sin(LINE_MAX_ANGLE*np.sign(nn_linear)*nn_angular),LINE_MAX_LEN*nn_linear*np.cos(LINE_MAX_ANGLE*np.sign(nn_linear)*nn_angular))))
                    img_front_ = self.line_on_image(img_front_, PREDICTION_START_POINT, prediction_end_point)
                    img_front_ = self.vel_measurements_on_image(img_front_, 5)
                elif mani_mode=="Classifying":
                    text_output = f"Empty bay: {round(output_data*100,2)}%"  
                    img_front_=self.text_on_image(img_front_, text_output, (50,280))
                    end_point=(tuple(map(lambda x, y: round(x) - round(y), START_POINT, (LINE_MAX_LEN*np.sign(linear)*linear*np.sin(LINE_MAX_ANGLE*angular),LINE_MAX_LEN*np.sign(linear)*linear*np.cos(LINE_MAX_ANGLE*angular)))))
                    img_front_=self.line_on_image(img_front_, START_POINT, end_point)
                elif mani_mode=="Saliency":
                    text_nn_mode = f"state {state} nn_sp: {round(nn_linear, 3)} nn_ag: {round(nn_angular, 3)}" 
                    img_front_=self.text_on_image(img_front_, text_nn_mode , (50,280))
                    end_point=(tuple(map(lambda x, y: round(x) - round(y), START_POINT, (LINE_MAX_LEN*nn_linear*np.sin(LINE_MAX_ANGLE*nn_angular),LINE_MAX_LEN*nn_linear*np.cos(LINE_MAX_ANGLE*nn_angular)))))
                    img_front_=self.line_on_image(img_front_, START_POINT, end_point)
                else:
                    text_mode = f"sp: {round(linear, 3)} ag: {round(angular, 3)}"
                    img_front_=self.text_on_image(img_front_, text_mode, (50,280))
                    end_point=(tuple(map(lambda x, y: round(x) - round(y), START_POINT, (LINE_MAX_LEN*linear*np.sin(LINE_MAX_ANGLE*angular),LINE_MAX_LEN*linear*np.cos(LINE_MAX_ANGLE*angular)))))
                    img_front_=self.line_on_image(img_front_, START_POINT, end_point)
                img_front_=cv2.resize(img_front_, (600, 450))
                if img_rear is None:
                    #print("imge size: ",img_front_.shape)
                    cv2.imshow('record',img_front_)
                    if mani_mode=="Saliency":
                        saliency_map_ = np.concatenate((saliency_map, np.zeros((60, saliency_map.shape[1],3 ),dtype=saliency_map.dtype)), axis=0)
                        saliency_map_ = cv2.resize(saliency_map_, (600, 450))
                        cv2.imshow('saliency', saliency_map_)
                else:
                    img_rear_ = np.concatenate((img_rear, np.zeros((60, img_rear.shape[1]),dtype=img_rear.dtype)), axis=0)
                    #print("imge size: ",img_rear_.shape)
                    img_rear_=cv2.resize(img_rear_, (600, 450))
                    cv2.imshow('record',np.concatenate((img_front_, img_rear_), axis=1))
                    if mani_mode=="Saliency":
                        saliency_map_ = np.concatenate((saliency_map, np.zeros((60, saliency_map.shape[1],3),dtype=saliency_map.dtype)), axis=0)
                        saliency_map_ = cv2.resize(saliency_map_, (600, 450))
                        cv2.imshow('saliency', saliency_map_)
                key=cv2.waitKey()
                #print("key: ",key)
                curr_time = time.time()
                key_interval=curr_time-last_time
                #print("key_interval: ",key_interval)
                last_time = curr_time
                if key== ord('d'):
                    if mani_mode=="Modifying":
                        #print("key_interval: ",key_interval)
                        if key_interval<0.1:
                            if np.sign(linear)!=np.sign(last_linear):
                                print("linear sign change")
                                stop_flag=stop_flag+1
                            elif stop_flag==1:
                                    self.idx=self.idx-1
                        else:
                            stop_flag=0
                        if not stop_flag:
                            self.idx+=step
                    else:
                        self.idx+=step
                elif key== ord('a'):
                    if mani_mode=="Modifying":
                        if key_interval<0.1:
                            if np.sign(linear)!=np.sign(last_linear):
                                print("linear sign change")
                                stop_flag=stop_flag+1
                            elif stop_flag==1:
                                    self.idx=self.idx+1
                        else:
                            stop_flag=0
                        if not stop_flag:
                            self.idx-=step
                    else:
                        self.idx-=step
                elif key== ord('w'):
                    step+=1
                elif key== ord('s'):
                    step-=1   
                elif key== ord('j'):
                    self.idx_begin=self.idx
                    if mani_mode=="Modifying":
                        self.keep_value=False
                        modified_linear_begin=modified_linear
                        modified_angular_begin=modified_angular
                elif key== ord('r') and mani_mode=="Modifying":
                    self.keep_value=True
                elif key== ord('q'):
                    step=1
                elif key== ord('e'):
                    step=20
                elif key in (ord("0"), ord("1"), ord("2"), ord("3")) and mani_mode in ("Comparing","Saliency"):
                    state = int(chr(key))
                elif key in (ord('6'),83) and mani_mode=="Modifying":#right arrow
                    self.keep_value=False
                    modified_angular-=0.05
                elif key in (ord('8'),82) and mani_mode=="Modifying":#up arrow
                    self.keep_value=False
                    modified_linear+=0.05
                elif key in (ord('2'),84) and mani_mode=="Modifying":#down arrow
                    self.keep_value=False
                    modified_linear-=0.05
                elif key in (ord('4'),81) and mani_mode=="Modifying":#left arrow
                    self.keep_value=False
                    modified_angular+=0.05
                elif key in (ord(' '),233) and mani_mode=="Modifying":#Space, Left Alt
                    if self.idx_begin is not None and self.idx_end is not None:
                        files_to_modify=self.sorted_file_list[self.idx_begin:self.idx_end+1]
                        if key == 233:#Left Alt
                            print(f"modified_linear_and_angular_begin: {modified_linear_begin}, {modified_angular_begin}")
                            modified_linear_list=np.linspace(modified_linear_begin,modified_linear_end,len(files_to_modify))
                            modified_angular_list=np.linspace(modified_angular_begin,modified_angular_end,len(files_to_modify))
                        print(f"modified_linear_and_angular_end: {modified_linear_end}, {modified_angular_end}")
                        for file_to_modify in files_to_modify:
                            file_to_modify_path=join(self.folder_path, file_to_modify)
                            data_array = np.load(file_to_modify_path, allow_pickle=True)

                            img_front, img_rear, linear, angular, driving_mode, gear, speed_mode, map_pos_orien = [None]*8
                            if key == 233:#Left Alt, fill in the values
                                current_modified_linear=modified_linear_list[files_to_modify.index(file_to_modify)]
                                current_modified_angular=modified_angular_list[files_to_modify.index(file_to_modify)]
                            else:#Space, same values
                                current_modified_linear=modified_linear_end
                                current_modified_angular=modified_angular_end
                            if len(data_array)==3:
                                modified_data_array=np.array([data_array[0],img_rear, data_array[1],data_array[2],driving_mode,gear,speed_mode,map_pos_orien,current_modified_linear,current_modified_angular],dtype=object)
                            elif len(data_array)==5:
                                modified_data_array=np.array([data_array[0],img_rear, data_array[1],data_array[2],data_array[3],data_array[4],speed_mode,map_pos_orien,current_modified_linear,current_modified_angular],dtype=object)
                            elif len(data_array) in (8,10):
                                modified_data_array=np.array([data_array[0],data_array[1],data_array[2],data_array[3],data_array[4],data_array[5],data_array[6],data_array[7],current_modified_linear,current_modified_angular],dtype=object)
                            np.save(file_to_modify_path,modified_data_array)
                    self.idx_begin, self.idx_end = None, None
                    #break
                elif key== ord('l'):
                    self.idx_end=self.idx
                    if mani_mode=="Modifying":
                        self.keep_value=False
                        modified_linear_end=modified_linear
                        modified_angular_end=modified_angular
                elif key in key_path_mapping:
                    target_folder_path = self.selected_path_dict[key_path_mapping[key]]
                    move_file=self.move_files(target_folder_path)
                elif key== ord('`') and f_idx > 0:
                    f_idx -= 2
                    break
                elif key==27: break
                last_linear=linear
                self.idx=max(self.new_idx_begin,min(self.idx,len(self.sorted_file_list)-1))
                step=max(1,min(step,40))
                modified_linear=max(-1.0,min(modified_linear,1.0))
                modified_angular=max(-1.0,min(modified_angular,1.0))  
            f_idx += 1          
        cv2.destroyAllWindows()
    def move_files(self, output_path , move_folder_flag=False):
        # need self.idx, self.folder_path, self.sorted_file_list, self.idx_begin, self.idx_end, self.new_idx_begin, self.bag_folder
        if ((self.idx + 1) == len(self.sorted_file_list) and self.idx_end is None) or move_folder_flag:
            os.rename(self.folder_path,join(output_path, self.bag_folder))
            print(f"moved folder {self.bag_folder} to folder {join(output_path, self.bag_folder)}")
        else:
            if self.idx_begin!=None and self.idx_end !=None:
                file_begin=self.sorted_file_list[self.idx_begin].split("_")[1]
                file_end=self.sorted_file_list[self.idx_end].split("_")[1]
                new_bag_folder="_".join(self.bag_folder.split("_")[:7])
                folder_name = f"{new_bag_folder}_{file_begin}-{file_end}"
                folder_move=join(output_path, folder_name)
                if not exists(folder_move):
                    os.mkdir(folder_move)
                for file_move in self.sorted_file_list[self.idx_begin:self.idx_end+1]:
                    if exists(join(self.folder_path, file_move)):
                        os.rename(join(self.folder_path, file_move),join(folder_move, file_move))
                    print(f"moved list index {self.idx_begin} to {self.idx_end}")
                    print(f"moved file {self.sorted_file_list[self.idx_begin]} to {self.sorted_file_list[self.idx_end]}") 
                    print(f"to folder {folder_move}")
                if (self.idx_end+1)==len(self.sorted_file_list):
                    return 0
                self.new_idx_begin=self.idx_end+1
            self.idx_begin, self.idx_end=None, None
            return 1
    def text_on_image(self,image,text,position):
        image=cv2.putText(image, text, position, FONT, FONT_SCALE,FONT_COLOR2,LINE_TYPE2)
        image=cv2.putText(image, text, position, FONT, FONT_SCALE,FONT_COLOR,LINE_TYPE)
        return image
    def line_on_image(self,image,start_point,end_point):
        image = cv2.line(image, start_point, end_point, FONT_COLOR2, THICKNESS2)
        image = cv2.line(image, start_point, end_point, ACTUAL_COLOR, THICKNESS)
        return image
    def vel_measurements_on_image(self,image,splits):
        distance = 80/splits
        image = cv2.line(image, (200,240), (200,160), (50), 4)
        image = cv2.line(image, (200,240), (200,160), (225), 2)
        image = cv2.line(image, (200-3,240-6), (200,240), (50), 4)
        image = cv2.line(image, (200+3,240-6), (200,240), (50), 4)
        image = cv2.line(image, (200-3,240-6), (200,240), (225), 2)
        image = cv2.line(image, (200+3,240-6), (200,240), (225), 2)
        for i in range(1,splits):
            image = cv2.line(image, (200-3,240-int(distance*i)), (200+3,240-int(distance*i)), (50), 4)
            image = cv2.line(image, (200-3,240-int(distance*i)), (200+3,240-int(distance*i)), (255-int(2*distance*i)), 2)
            image = cv2.putText(image, str(8-2*i), (200+15,240-int(distance*i)+5), FONT, 0.5, (250), 1)
        return image
    def hori_measurements_on_image(self,image,splits):
        distance = 160 /splits
        for i in range(0 ,splits):
            image = cv2.line(image, (200-int(distance*i),240-3), (200-int(distance*i),240+3), (50), 4)
            image = cv2.line(image, (200-int(distance*i),240-3), (200-int(distance*i),240+3), (255-int(2*distance*i)), 2)
            image = cv2.line(image, (200+int(distance*i),240-3), (200+int(distance*i),240+3), (50), 4)
            image = cv2.line(image, (200+int(distance*i),240-3), (200+int(distance*i),240+3), (255-int(2*distance*i)), 2)
            image = cv2.putText(image, str(i), (200-int(distance*i),240+15), FONT, 0.3, (250), 1)
            image = cv2.putText(image, str(i), (200+int(distance*i),240+15), FONT, 0.3, (250), 1)
        return image 
    def extract_npy_data(self,modified_linear=0, modified_angular=0): 
        # need self.idx, self.folder_path, self.sorted_file_list, self.keep_value
        file=self.sorted_file_list[self.idx]
        file_path=join(self.folder_path, file)
        data_array = np.load(file_path, allow_pickle=True)
        img_front, img_rear, linear, angular, driving_mode, gear, speed_mode, map_pos_orien, linear2, angular2 = [None]*10
        if len(data_array)==10:
            img_front, img_rear, linear, angular, driving_mode, gear, speed_mode, map_pos_orien, linear2, angular2 = data_array
            if self.keep_value: 
                modified_linear, modified_angular = linear2, angular2
        else:
            if len(data_array)==3: 
                img_front, linear, angular = data_array
            elif len(data_array)==5: 
                img_front, linear, angular, driving_mode, gear = data_array
            elif len(data_array)==8: 
                img_front, img_rear, linear, angular, driving_mode, gear, speed_mode, map_pos_orien = data_array
            if self.keep_value: 
                modified_linear, modified_angular = linear, angular
        img_front=img_front[:, 40:440]      
        if img_rear is not None:
            img_rear=img_rear[:, 40:440]
        if map_pos_orien is None:
            orien, map_pose = None, None
        else:
            delta_x, delta_y = map_pos_orien[:2]
            orien = euler_from_quaternion_z(map_pos_orien[2], map_pos_orien[3])
            #change orien range from -pi to pi to 0 to 2pi, (0 is east, pi/2 is north, pi is west, 3pi/2 is south)
            #orien = (math.pi*2 + orien) if orien < 0 else orien
            #rad to degree
            orien = round(orien * 180 / math.pi,2)
            tree_x = self.cp_xy_cords[0] + delta_x
            tree_y = self.cp_xy_cords[1] + delta_y
            #print("tree_x: ",tree_x,"tree_y: ",tree_y)
            map_pose=(round(tree_x*MAP_RESOLUTION),MAP_HEIGHT_PIX-round(tree_y*MAP_RESOLUTION))
        return img_front, img_rear, linear, angular, driving_mode, gear, speed_mode, orien, map_pose, modified_linear, modified_angular, file

    def Stops_Auto_Sorting(self):
        Stage2_Path=join(sorted_data_path, "CIL_Dual_Cam_Stage2_B")
        for Behavior_Folder_Name in os.listdir(Stage2_Path):
            Behavior_Folder=join(Stage2_Path, Behavior_Folder_Name)
            if "stops" not in Behavior_Folder_Name:
                Behavior_Folder_Move = join(
                    Stage2_Path, f"{Behavior_Folder_Name}_stops"
                )
                if not exists(Behavior_Folder_Move):
                    os.mkdir(Behavior_Folder_Move)
                for Folder_Name in os.listdir(Behavior_Folder):
                    self.folder_path = join(Behavior_Folder, Folder_Name)
                    print(self.folder_path)
                    Folder_Move = join(Behavior_Folder_Move, f"{Folder_Name}_stops")
                    print(Folder_Move)
                    for file in os.listdir(self.folder_path):
                        File_path=join(self.folder_path, file)
                        data_array = np.load(join(self.folder_path, file), allow_pickle=True)
                        Speed=data_array[2]
                        if ((Speed < 0.01) and (Speed > -0.01)):
                            if not exists(Folder_Move):
                                os.mkdir(Folder_Move)
                            os.rename(join(self.folder_path, file),join(Folder_Move, file))
    # def Correct_File_Name(self):
    #     Stage2_Path=join(sorted_data_path, "CIL_Dual_Cam_Stage2_B")
    #     for Behavior_Folder_Name in os.listdir(Stage2_Path):
    #         if "stops" in Behavior_Folder_Name:
    #             Behavior_Folder=join(Stage2_Path, Behavior_Folder_Name)
    #             for Folder_Name in os.listdir(Behavior_Folder):
    #                 self.folder_path = join(Behavior_Folder, Folder_Name)
    #                 index_str=self.folder_path.split("_")[-1]
    #                 index_str2=self.folder_path.split("_")[-2]
    #                 if index_str == "stops" and index_str2 == "stops":
    #                     New_Folder_Name="_".join(Folder_Name.split("_")[:-1])
    #                     New_Folder_Path=join(Behavior_Folder, New_Folder_Name)
    #                     #print(self.folder_path)
    #                     #print(New_Folder_Path)
    #                     os.rename(self.folder_path,New_Folder_Path) 
    def Extract_Reverse_From_Pullout(self):
        Behavior_Folder=self.selected_path_dict["pullout"]
        Behavior_Folder_Move=self.selected_path_dict["reverse"]
        for Folder_Name in os.listdir(Behavior_Folder):
            self.folder_path = join(Behavior_Folder, Folder_Name)
            print(self.folder_path)
            Folder_Move = join(Behavior_Folder_Move, f"{Folder_Name}_reverse")
            print(Folder_Move)
            for file in os.listdir(self.folder_path):
                File_path=join(self.folder_path, file)
                data_array = np.load(join(self.folder_path, file), allow_pickle=True)
                if len(data_array)<8:
                    Speed=data_array[1]
                else:
                    Speed=data_array[2]
                if (Speed < 0.0):
                    if not exists(Folder_Move):
                        os.mkdir(Folder_Move)
                    os.rename(join(self.folder_path, file),join(Folder_Move, file))
    # def Recover_Reverse_To_Pullout(self):
    #     Behavior_Folder=self.selected_path_dict["reverse"]
    #     Behavior_Folder_Move=self.selected_path_dict["pullout"]
    #     for Folder_Name in os.listdir(Behavior_Folder):
    #         self.folder_path = join(Behavior_Folder, Folder_Name)
    #         #print(self.folder_path)
    #         New_Folders=Folder_Name.replace("_reverse","")
    #         Folder_Move = join(Behavior_Folder_Move, New_Folders)
    #         print(Folder_Move)
    #         for file in os.listdir(self.folder_path):
    #             os.rename(join(self.folder_path, file),join(Folder_Move, file))
    # def Simplify_File_Names(self):
    #     for Stage_Folder_Name in os.listdir(sorted_data_path):
    #         Stage_Folder=join(sorted_data_path, Stage_Folder_Name)
    #         print("Stage_Folder: ",Stage_Folder)
    #         for Behavior_Folder_Name in os.listdir(Stage_Folder):
    #             Behavior_Folder=join(Stage_Folder, Behavior_Folder_Name)
    #             print("Behavior_Folder: ",Behavior_Folder)
    #             for Folder_Name in os.listdir(Behavior_Folder):
    #                 self.folder_path = join(Behavior_Folder, Folder_Name)
    #                 for file in os.listdir(self.folder_path):
    #                     File_path=join(self.folder_path, file)
    #                     New_File_Name="_".join(file.split("_")[:2])+"_"+file.split("_")[-1]
    #                     New_File_Path=join(self.folder_path, New_File_Name)
    #                     os.rename(File_path,New_File_Path)
                

    def Folder_Auto_Merging(self):#merge the Folder_Name in the same behavior and same place
        Stage2_Path=join(sorted_data_path, "three_roundabouts_front_cam_only")
        last_file_begin=0
        last_file_end=0
        last_search_folder=None
        for Behavior_Folder_Name in sorted(os.listdir(Stage2_Path)):
            Behavior_Folder=join(Stage2_Path, Behavior_Folder_Name)
            print("Behavior_Folder: ",Behavior_Folder)
            for Folder in sorted(os.listdir(Behavior_Folder)):
                self.folder_path = join(Behavior_Folder, Folder)
                index_str=self.folder_path.split("_")[-1]
                index_str2=self.folder_path.split("_")[-2]
                if index_str not in ["0", "recovered","0recovered"] and index_str2 not in ["0", "recovered","0recovered"]:
                    if index_str in ["stops","reverse"]:
                        file_begin=int(index_str2.split("-")[0])
                        file_end=int(index_str2.split("-")[1])
                    else:
                        file_begin=int(index_str.split("-")[0])
                        file_end=int(index_str.split("-")[1])
                    if file_begin==last_file_end+1:
                        if index_str in ["stops"] and last_idex_str in ["stops"]:
                            Merged_Folder="_".join(Folder.split("_")[:7])+"_"+str(last_file_begin)+"-"+str(file_end)+"_stops"
                        elif index_str in ["reverse"] and last_idex_str in ["reverse"]:
                            Merged_Folder="_".join(Folder.split("_")[:7])+"_"+str(last_file_begin)+"-"+str(file_end)+"_reverse"
                        else:
                            Merged_Folder="_".join(Folder.split("_")[:7])+"_"+str(last_file_begin)+"-"+str(file_end)
                        Merged_Folder_Path=join(Behavior_Folder, Merged_Folder)
                        print("Merged_Folder: ",Merged_Folder)
                        #print("self.folder_path: ",self.folder_path)
                        #print("last_search_folder: ",last_search_folder)
                        if not exists(Merged_Folder_Path):
                            os.mkdir(Merged_Folder_Path)
                        for file in os.listdir(self.folder_path):
                            os.rename(join(self.folder_path, file),join(Merged_Folder_Path, file))
                        os.rmdir(self.folder_path)
                        for file in os.listdir(last_search_folder):
                            os.rename(join(last_search_folder, file),join(Merged_Folder_Path, file))
                        os.rmdir(last_search_folder)
                        file_begin=last_file_begin
                        file_end=file_end
                        self.folder_path=Merged_Folder_Path
                    last_file_begin=file_begin
                    last_file_end=file_end
                    last_idex_str=index_str
                    last_search_folder=self.folder_path
    def Pullin_Trigger(self):
        if not exists(self.selected_path_dict["lane_empty_bay_first_half"]):
            return
        Behavior_Folder=self.selected_path_dict["lane_empty_bay_first_half"]
        Behavior_Folder_Move=self.selected_path_dict["lane_empty_bay_first_half_pullin_trigger"]
        for Folder_Name in os.listdir(Behavior_Folder):
            self.folder_path = join(Behavior_Folder, Folder_Name)
            print(self.folder_path)
            Folder_Move = join(Behavior_Folder_Move, f"{Folder_Name}_pullin_trigger")
            if not exists(Folder_Move):
                os.mkdir(Folder_Move)
            print(Folder_Move)
            self.sorted_file_list=sorted(os.listdir(self.folder_path), key=lambda x: int(x.split("_")[1]))
            folder_len=len(self.sorted_file_list)
            for self.idx in range (folder_len):
                file=self.sorted_file_list[self.idx]
                File_path=join(self.folder_path, file)
                data_array = np.load(File_path, allow_pickle=True)
                if len(data_array)==3:
                    driving_mode=None
                    gear=None
                else:
                    driving_mode=data_array[3]
                    gear=data_array[4]
                angular=data_array[2]
                linear=data_array[1]
                image=data_array[0]
                linear=0.7-(0.5*(self.idx/(folder_len-1)))
                linear=max(linear,0.1)
                angular=angular+(1.0*(self.idx/(folder_len-1)))
                angular=min(angular,1.0)
                New_Data_Array_File_Name=file.split("_")[0]+"_"+file.split("_")[1]+"_"+str(f"{linear:.3f}")+"_"+str(f"{angular:.3f}")+"_"+str(driving_mode)+"_"+str(gear)+".npy"
                New_Data_Array=np.array([image,linear,angular,driving_mode,gear],dtype=object)
                np.save(join(Folder_Move, New_Data_Array_File_Name), New_Data_Array)
    def NN_to_Modified_CMD(self):
        Behavior_Folder=[]
        if not exists(Behavior_Folder):
            return
        import tensorflow as tf
        model_path = join(trained_model_path, "BW_Scan_B0_Cmd2_Single_2025-03-08-11.03.17/PilotNet_BW_Scan_B0_Cmd2_2025-03-08-12.56.40.tflite")
        interpreter = tf.lite.Interpreter(model_path)
        interpreter.allocate_tensors()
        output_0 = interpreter.get_output_details()[0]
        output_1 = interpreter.get_output_details()[1]
        input_0 = interpreter.get_input_details()[0]
        for self.bag_folder in sorted(os.listdir(Behavior_Folder)):
            self.folder_path = join(Behavior_Folder, self.bag_folder)
            self.sorted_file_list=sorted(os.listdir(self.folder_path), key=lambda x: int(x.split("_")[1]))
            print(self.folder_path)
            print(len(os.listdir(self.folder_path)))
            self.idx_begin, self.idx_end = 0, 0
            for self.idx in range(len(self.sorted_file_list)):
                data_array=np.load(join(self.folder_path, self.sorted_file_list[self.idx]), allow_pickle=True)
                img_front, img_rear, linear, angular, driving_mode, gear, speed_mode, map_pose = [None]*8
                img_front=data_array[0]
                img_front=img_front[:, 40:440]
                img_input = np.expand_dims(img_front, axis=0).astype('float32')
                interpreter.set_tensor(input_0['index'], img_input)
                interpreter.invoke()
                nn_linear=float(interpreter.get_tensor(output_1['index']))
                nn_angular=float(interpreter.get_tensor(output_0['index']))
                if nn_linear>0.0:
                    print("file: ",self.sorted_file_list[self.idx])
                    print("nn_linear: ",nn_linear)
                if len(data_array)==3:
                    modified_data_array=np.array([data_array[0],img_rear, data_array[1],data_array[2],driving_mode,gear,speed_mode,map_pose,nn_linear,nn_angular],dtype=object)
                elif len(data_array)==5:
                    modified_data_array=np.array([data_array[0],img_rear, data_array[1],data_array[2],data_array[3],data_array[4],speed_mode,map_pose,nn_linear,nn_angular],dtype=object)
                elif len(data_array) in (8,10):
                    modified_data_array=np.array([data_array[0],data_array[1],data_array[2],data_array[3],data_array[4],data_array[5],data_array[6],data_array[7],nn_linear,nn_angular],dtype=object)
                np.save(join(self.folder_path, self.sorted_file_list[self.idx]),modified_data_array)
                
                

    def Split_Stage2(self):
        Stage2_Path=join(sorted_data_path, "CIL_Dual_Cam_Stage2")
        Stage2_First_Half=join(sorted_data_path, "CIL_Dual_Cam_Stage2_First_Half")
        Stage2_Second_Half=join(sorted_data_path, "CIL_Dual_Cam_Stage2_Second_Half")
        for Behavior_Folder_Name in os.listdir(Stage2_Path):
            self.data_folder_path=join(Stage2_Path, Behavior_Folder_Name)
            Behavior_Folder_Move_First_Half = join(Stage2_First_Half, Behavior_Folder_Name)
            Behavior_Folder_Move_Second_Half = join(Stage2_Second_Half, Behavior_Folder_Name)
            if Behavior_Folder_Name in ["roundabout_straight","roundabout_turn_around_to_beach","roundabout_turn_around_to_office","roundabout_turn_around_to_beach_stops","roundabout_turn_around_to_office_stops"]:
                os.rename(self.folder_path,join(Behavior_Folder_Move_First_Half, self.bag_folder))
            else:
                for self.bag_folder in sorted(os.listdir(self.data_folder_path)):
                    self.folder_path = join(self.data_folder_path, self.bag_folder)
                    self.sorted_file_list=sorted(os.listdir(self.folder_path), key=lambda x: int(x.split("_")[1]))
                    print(self.folder_path)
                    print(len(os.listdir(self.folder_path)))
                    self.idx_begin, self.idx_end = 0, 0
                    map_pose_x_list=[]
                    pre_map_pose=(12000,2000)#Just a point on the first half of the map
                    for self.idx in range(len(self.sorted_file_list)):
                        map_pose = self.extract_npy_data()[8]
                        #print("map_pose: ",map_pose)
                        if map_pose is None:
                            map_pose=pre_map_pose
                        map_pose_x_list.append(map_pose[0])
                        pre_map_pose=map_pose
                    if np.all(np.array(map_pose_x_list) ==12000):
                        pass
                    elif np.all(np.array(map_pose_x_list) > 8000):
                        os.rename(self.folder_path,join(Behavior_Folder_Move_First_Half, self.bag_folder))
                    elif np.all(np.array(map_pose_x_list) < 8000):   
                        os.rename(self.folder_path,join(Behavior_Folder_Move_Second_Half, self.bag_folder))
                    elif np.mean(np.array(map_pose_x_list))>8000 and len(map_pose_x_list)>100:
                        os.rename(self.folder_path,join(Behavior_Folder_Move_First_Half, self.bag_folder))
                

if __name__ == '__main__':
    #Data_Sorting().Pullin_Trigger()
    Data_Sorting().manual_manipulating()
    #Data_Sorting().Extract_Reverse_From_Pullout()
    #Data_Sorting().Stage1_Auto_Sorting()
    #Data_Sorting().Stops_Auto_Sorting()
    #Data_Sorting().Stage2_Auto_Sorting()
    #Data_Sorting().Folder_Auto_Merging()
    #Data_Sorting().Recover_Reverse_To_Pullout()
    #Data_Sorting().Split_Stage2()
    #Data_Sorting().NN_to_Modified_CMD()
    #Data_Sorting().Correct_File_Name()
    #Data_Sorting().Simplify_File_Names()