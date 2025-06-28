import torch
from torch import nn
from functools import partial

from torchvision import datasets, models, transforms
from PIL import Image

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models_ivan import ResNet34PilotNet

from torch.utils.tensorboard import SummaryWriter

import numpy as np

import cv2
import os
import sys
from os.path import join, exists, dirname, abspath

from sklearn.model_selection import train_test_split

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import argparse
from torchsummary import summary
import time
import random

import yaml

writer = SummaryWriter()

Images_All = []
Speeds_All = []
Steering_Angles_All = []

def pair(t):

    return t if isinstance(t, tuple) else (t,t)


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2.5))], 0.05)
    ]
)

class dataset(torch.utils.data.Dataset):
    def __init__(self, file_list, label_list1, label_list2, transform=None):
        self.file_list = file_list
        self.label_list1 = label_list1
        self.label_list2 = label_list2
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img = self.file_list[idx]
        label1 = self.label_list1[idx]
        label2 = self.label_list2[idx]
        img_transformed = self.transform(img)
        return img_transformed, label1, label2

def load_latest_data(Search_Folder):
    # for Files in os.listdir(Search_Folder):
        # print(i)
        # print(Files)
    # print(Search_Folder)
    # print(Search_Folder[0].split("_")[-6])
    # print(Search_Folder[1].split("_")[-6])
    
    try:
        if(int(Search_Folder[1].split("_")[-6]) - int(Search_Folder[0].split("_")[-6]) != 6):
            # print(int(Search_Folder[1].split("_")[-6]) - int(Search_Folder[0].split("_")[-6]))
            return
    except:
        return
    try:
        data = np.load(Search_Folder[1], allow_pickle=True)
        past_data = np.load(Search_Folder[0], allow_pickle=True)
    except EOFError:
        return
    except:
        # print("skipping file in {}: {}", Search_Folder, Files)
        return

    img = data[0]
    if len(past_data) == 8:
        Speed, Steering_Angle = past_data[2], past_data[3]
    elif len(past_data) == 10:
        Speed, Steering_Angle = past_data[8], past_data[9]
    else:
        Speed, Steering_Angle = past_data[1], past_data[2]

    # add image shifting here
    offset = 0
    mean = 80
    std_dev = 30

    while True:
        offset = np.random.normal(loc=mean, scale=std_dev)
        if 0 <= offset <= 160:
            offset = int(round(offset))
            break
    
    # offset = np.random.randint(0, 160)

    #check for model type training
    
    img = img[80:, 0 + offset:320 + offset]
    Steering_Angle = Steering_Angle - ((80-offset)/80)*0.2
    img = Image.fromarray(np.uint8(img), mode='L')
    try:
        if Speed == None:
            # print(join(Search_Folder, Files))
            return
        if Steering_Angle == None:
            # print(join(Search_Folder, Files))
            return
        if img == None:
            # print(join(Search_Folder, Files))
            # print(Files)
            return
    except:
        print(Speed)
        print(Steering_Angle)
        print(img)
        print(Search_Folder)

    Images_All.append(img)
    Speeds_All.append(Speed)
    Steering_Angles_All.append(Steering_Angle)

    

if __name__ == "__main__":
    lr = 1e-4 #1e-4
    weight_decay = 0.05
    gamma = 0.01
    batch_size = 150
    best_loss = 1000000
    Speed_scale = 1.0
    Steering_Angle_scale = 1.0

    model_path_name = "resnet34.ckpt"
    checkpoint = None

    
    with open('config.yaml', 'r') as file:
        config_yaml = yaml.safe_load(file)

    parser = argparse.ArgumentParser("Load model from checkpoint")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--fine_tune_model", action="store_true")
    parser.add_argument("model_type", default="lane_following", type=str, nargs='?')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #K adds this in model class definition so I will just add here
    model = ResNet34PilotNet(use_rgb=False).to(device) #manually set rgb to false for now since currently dont think ill ever need it


    args = parser.parse_args()

    if args.load_model:
        print("🟢 Using PPGeo pretrained ResNet-34 encoder")
        ppgeo_ckpt = torch.load('resnet34.ckpt', map_location='cpu')
        state_dict = ppgeo_ckpt['state_dict']
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}

        # ——— GRAYSCALE ADAPTATION & WEIGHT LOADING ———
        # if doing true-grayscale fine-tuning, average the pretrained RGB conv1 → 1-channel
        # checkpoint has "conv1.weight": torch.Size([64,3,7,7])
        w3 = state_dict['conv1.weight']                # [64,3,7,7]
        state_dict['conv1.weight'] = w3.mean(1, keepdim=True)  # → [64,1,7,7] 
        # now load everything (conv1 will match or be ignored)  
        model.backbone.load_state_dict(state_dict, strict=False)

    elif args.fine_tune_model:
        print("fine tuning model")
        checkpoint = torch.load(model_path_name, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

    
    #summary(model, (1, 160, 320)) commenting out for now as it may cause errors from dimension mismatches with kierans

    Images_All_Straight = []
    Speeds_All_Straight = []
    Steering_Angles_All_Straight = []

    Images_All_Turn = []
    Speeds_All_Turn = []
    Steering_Angles_All_Turn = []

    Images_All = []
    Speeds_All = []
    Steering_Angles_All = []

    if args.fine_tune_model:
        model_name = args.model_type + '_finetune'
    else:
        model_name = args.model_type
    print(model_name)

    if args.fine_tune_model:
        lane_follow_files = [
            "lane_bay_pass",
            "roundabout_straight",
            # "lane_following",
            "intersection_lane_following",
            "startpoint_out",
            "startpoint_in",
            "carpark_pass",
            "roundabout_right_turn",
            "lane_empty_bay",
            "lane_empty_bay_first_half",
            "lane_empty_bay_second_half",
            "pullout",
        ]
    else:
        lane_follow_files = [
            "lane_bay_pass",
            "roundabout_straight",
            "lane_following",
            "intersection_lane_following",
            "startpoint_out",
            "startpoint_in",
            "carpark_pass",
            "roundabout_right_turn",
            "lane_empty_bay",
            "lane_empty_bay_first_half",
            "lane_empty_bay_second_half",
            "pullout",
        ]

    pullin_files = [
        # "lane_following", #reduce the amount
        "pullin",
        # "roundabout_turn_around_to_office",
        # "intersection_turn_around_to_office",
        # "startpoint_out",
        # "startpoint_in",
        # "carpark_entry",
        # "roundabout_right_turn",
        "pullin_stops",
        # "carpark_left_turn_in",
        # "carpark_left_turn_out"
    ]

    reverse_files = [
        # "lane_following",
        "reverse",
        # "roundabout_turn_around_to_beach",
        # "roundabout_turn_around_to_office",
        # "startpoint_out",
        # "startpoint_in",
        # "carpark_entry",
        # "roundabout_right_turn",
        "pullout_stops",
        "reverse_manual",
        # "carpark_left_turn_in",
        # "carpark_left_turn_out"
    ]

    Image_Paths = []
    All_Searchable_Folders = []
    past_data = []
    offset = -7 #offset is 6 but need to use 7 because indexing

    if args.model_type == "lane_following":
        model_files = lane_follow_files
    elif args.model_type == "pullin":
        model_files = pullin_files
    else:
        model_files = reverse_files

    Base_Path = dirname("/home/quirky/Documents/eglinton_datasorting_dual/sorted_eglinton_data/")

    for Folders in os.listdir(Base_Path):
        Sorted_Folder_Path = join(Base_Path, Folders)
        for Folder in os.listdir(Sorted_Folder_Path):
            for name in model_files:
                if name == Folder:
                    Image_Paths.append(join(Sorted_Folder_Path, Folder))
    
    for Image_Path in Image_Paths:
        if exists(Image_Path):
            for Folders in os.listdir(Image_Path):
                Search_Folder = join(Image_Path, Folders)
                i = 0
                past_data.clear()
                for file in sorted(os.listdir(Search_Folder), key=lambda x: int(x.split("_")[1])):
                    past_data.append(file)
                    i += 1
                    if len(past_data) < 6:
                        continue
                    else:
                        All_Searchable_Folders.append([join(Search_Folder, past_data[i + offset]), \
                                                   join(Search_Folder, file)])
        else:
            print('[Error!] Check Image Directory is Correct!')
            sys.exit()

    random.shuffle(All_Searchable_Folders)


    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if args.load_model:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler = StepLR(optimizer, step_size=10, gamma=gamma)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=gamma, patience=5)

    epochs = 50
    use_amp = True

    if args.load_model:
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(len(All_Searchable_Folders))

    current_time = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime())

    print(f'training started at: {current_time}')
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        total_epoch_loss = 0
        total_epoch_val_loss = 0
        total_epoch_accuracy1 = 0
        total_epoch_accuracy2 = 0
        total_epoch_val_accuracy1 = 0
        total_epoch_val_accuracy2 = 0
        temp_search = All_Searchable_Folders.copy()
        train_len = 0
        val_len = 0
        batch = 0
        straight_count = 0
        turning_count = 0

        while len(temp_search) != 0:
            epoch_loss = 0
            epoch_accuracy1 = 0
            epoch_accuracy2 = 0

            while (len(Images_All) < 15000) and len(temp_search) != 0:
                current_folder = temp_search.pop()

                load_latest_data(current_folder)
            
                # print(len(Images_All))
            # print(len(temp_search))
            # print(len(All_Searchable_Folders))
            # print(len(Images_All))
            print((len(temp_search)/len(All_Searchable_Folders))*100)
            

            Split_a = train_test_split(Images_All, Speeds_All, Steering_Angles_All, test_size=0.1, shuffle=True)
            (Images, Image_Test, Speeds, Speed_Test, Steering_Angles, Steering_Angle_Test) = Split_a   
            Split_b = train_test_split(Images, Speeds, Steering_Angles, test_size=0.2, shuffle=True)
            (Image_Train, Image_Valid, Speed_Train, Speed_Valid, Steering_Angle_Train, Steering_Angle_Valid) = Split_b

            train_data = dataset(Image_Train, Speed_Train, Steering_Angle_Train, transform=transform)
            test_data = dataset(Image_Test, Speed_Test, Steering_Angle_Test, transform=transform)
            val_data = dataset(Image_Valid, Speed_Valid, Steering_Angle_Valid, transform=transform)

            train_loader = torch.utils.data.DataLoader(
                dataset=train_data, batch_size=batch_size, shuffle=False
            )
            test_loader = torch.utils.data.DataLoader(
                dataset=test_data, batch_size=batch_size, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                dataset=val_data, batch_size=batch_size, shuffle=True
            )

            train_len += len(train_loader.dataset)
            val_len += len(val_loader.dataset)


            for data, label1, label2 in train_loader:
                if data == None or label1 == None or label2 == None:
                    continue
               
                data = data.to(device)
                # print("----------------label -- output -- loss-------------------")
                # print(label1)
                # print(label2)

                for item in label2:
                    if abs(item) > 0.1:
                        turning_count += 1
                    else:
                        straight_count += 1

                label1 = label1.float().to(device)
                label1 = label1 / Speed_scale
                label2 = label2.float().to(device)
                label2 = label2 / Steering_Angle_scale


                output1, output2 = model(data)
                loss1 = criterion(output1, label1)
                loss2 = criterion(output2, label2)

                
                # print(output1)
                # print(output2)
                # print(loss1)
                # print(loss2)
                # print("----------------------------------------------------------")

                loss = loss1 + loss2

                total_epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc1 = (abs(output1 - label1) < (0.27 / 5.4)).float().sum()
                acc2 = (abs(output2 - label2) < (0.015 / 0.3)).float().sum()
                epoch_accuracy1 += acc1
                epoch_accuracy2 += acc2
                epoch_loss += loss.item()

            total_epoch_accuracy1 += epoch_accuracy1
            total_epoch_accuracy2 += epoch_accuracy2
            epoch_accuracy1 = epoch_accuracy1 / len(train_loader.dataset)
            epoch_accuracy2 = epoch_accuracy2 / len(train_loader.dataset)
            epoch_loss = epoch_loss / len(train_loader.dataset)

            # print(
            #     "Batch : {}, train accuracy1 : {}, train accuracy2 : {}, train loss : {}".format(
            #         batch + 1, epoch_accuracy1, epoch_accuracy2, epoch_loss
            #     )
            # )

            with torch.no_grad():
                epoch_val_accuracy1 = 0
                epoch_val_accuracy2 = 0
                epoch_val_loss = 0
                # for i in range(0, len(Image_Valid)): 
                for data, label1, label2 in val_loader:
                    if data == None or label1 == None or label2 == None:
                        continue
                    # data = Image_Valid[i]
                    # speed = Speed_Valid[i]
                    # angle = Steering_Angle_Valid[i]
                    data = data.to(device)
                    label1 = label1.float().to(device)
                    label2 = label2.float().to(device)

                    val_output1, val_output2 = model(data)

                    
                    val_loss1 = criterion(val_output1, label1)
                    val_loss2 = criterion(val_output2, label2)

                    val_loss = val_loss1 + val_loss2
                    total_epoch_val_loss += val_loss.item()

                    acc1 = (abs(val_output1 - label1) < (0.27 / 5.4)).float().sum()
                    acc2 = (abs(val_output2 - label2) < (0.015 / 0.3)).float().sum()
                    epoch_val_accuracy1 += acc1
                    epoch_val_accuracy2 += acc2
                    epoch_val_loss += val_loss.item()

                total_epoch_val_accuracy1 += epoch_val_accuracy1
                total_epoch_val_accuracy2 += epoch_val_accuracy2
                epoch_val_accuracy1 = epoch_val_accuracy1 / len(val_loader.dataset)
                epoch_val_accuracy2 = epoch_val_accuracy2 / len(val_loader.dataset)
                epoch_val_loss = epoch_val_loss / len(val_loader.dataset)

                # print(
                #     "Batch : {}, val_accuracy : {}, val_accuracy : {}, val_loss : {}".format(
                #         batch + 1, epoch_val_accuracy1, epoch_val_accuracy2, epoch_val_loss
                #     )
                # )

            batch += 1
            Images_All = []
            Speeds_All = []
            Steering_Angles_All = []
            del train_data, train_loader, val_data, val_loader, test_data, test_loader
         

        total_epoch_accuracy1 = total_epoch_accuracy1 / train_len
        total_epoch_accuracy2 = total_epoch_accuracy2 / train_len
        total_epoch_loss = total_epoch_loss / train_len
        total_epoch_val_accuracy1 = total_epoch_val_accuracy1 / val_len
        total_epoch_val_accuracy2 = total_epoch_val_accuracy2 / val_len
        total_epoch_val_loss = total_epoch_val_loss / val_len
        writer.add_scalar("Loss/Train", total_epoch_loss, epoch)
        writer.add_scalar("Loss/Validation", total_epoch_val_loss, epoch)
        writer.add_scalar("accuracy1/Train", total_epoch_accuracy1, epoch)
        writer.add_scalar("accuracy2/Train", total_epoch_accuracy2, epoch)
        writer.add_scalar("accuracy1/Validation", total_epoch_val_accuracy1, epoch)
        writer.add_scalar("accuracy2/Validation", total_epoch_val_accuracy2, epoch)

        print(
            "Total Epoch : {} values, accuracy1 : {}, accuracy2 : {}, loss : {}".format(
                epoch + 1, total_epoch_accuracy1, total_epoch_accuracy2, total_epoch_loss
            )
        )
        print(
            "Total Epoch : {} values, val_accuracy1 : {}, val_accuracy2 : {}, val_loss : {}".format(
                epoch + 1, total_epoch_val_accuracy1, total_epoch_val_accuracy2, total_epoch_val_loss
            )
        )

        if best_loss > total_epoch_loss:
            best_loss = total_epoch_loss

            save_name = f"VMamba_shuttle_{model_name}_{epoch+1}_{total_epoch_loss:.4f}_{total_epoch_accuracy1:.4f}_{total_epoch_accuracy2:.4f}.pth"

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_epoch_loss
            }, save_name)

        scheduler.step(total_epoch_val_loss)

        elapsed_time = time.time() - start_time
        start_time = time.time()

        print(f'Time elapsed: {elapsed_time}')

        print(f"straight count: {straight_count}, turning count: {turning_count}")

    end_time = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime())

    print(f'training finished at: {end_time}')

    torch.save(model.state_dict(), f'VMamba_shuttlebus_{model_name}.pth')
    writer.flush()
    writer.close()