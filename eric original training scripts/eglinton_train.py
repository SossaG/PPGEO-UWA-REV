#!/usr/bin/env python
import hydra
import wandb
from wandb.integration.keras  import WandbMetricsLogger, WandbModelCheckpoint
from omegaconf import DictConfig, OmegaConf, ListConfig
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Layer, Input, Lambda, Cropping2D, Conv2D, BatchNormalization,
                                     Activation, Add, Flatten, Dense, Dropout, Concatenate)
from tensorflow.keras import backend as BK
from tensorflow.keras.utils import get_custom_objects
import tf2onnx
import onnx
import numpy as np
import datetime
import cv2
import sys
import os
from os.path import join, exists, dirname, abspath, getmtime
from models import SGrayBNPilotNet, SGrayResPilotNet,SGrayMobileNetV3Small, SGrayMobileNetV3Large, SGrayMobileNetV3LargeD1, SGrayEfficientNetV2S, SGrayEfficientNetV2SD1
    
#source ~/tf_env/bin/activate
# Custom Laplacian-based negative log-likelihood loss
def laplacian_nll_loss(y_true, y_pred):
    # y_pred is assumed to have two components:
    # y_pred[:, 0]: predicted value (mu)
    # y_pred[:, 1]: log-scale (s), such that b = exp(s)
    mu = y_pred[:, 0]
    s = y_pred[:, 1]
    #tf.print("s (log-scale) min:", tf.reduce_min(s), "max:", tf.reduce_max(s))
    s = tf.clip_by_value(s, 0, 1)  # Keep s in a reasonable range
    # Ensure numerical stability (prevent b from being too small)
    b = tf.exp(s)
    # Calculate the loss: |y - mu| / b + s
    loss = tf.abs(y_true - mu) / b +  s
    return tf.reduce_mean(loss)
get_custom_objects().update({"laplacian_nll_loss": laplacian_nll_loss})

def predict_with_uncertainty(model, image):
    # image should have shape (240, 400, 1)
    pred = model.predict(np.expand_dims(image, axis=0))
    mu = pred[0, 0]
    s = pred[0, 1]
    b = np.exp(s)
    # Optionally, define confidence as inverse of b or use 1/(1+b)
    confidence = 1.0 / (1.0 + b)
    return mu, confidence

#Data Augmentation
def add_blur(image, max_kernel_size=5):
    random_kernel_size = np.random.randint(2, max_kernel_size+1)
    #image = cv2.GaussianBlur(image, (random_kernel_size, random_kernel_size), 0)
    image = cv2.blur(image, (random_kernel_size, random_kernel_size))
    return image
def adjust_brightness(image, range=0.05):
    factor = np.random.uniform(1 - range, 1 + range)
    image = np.clip(image * factor, 0, 255).astype(np.uint8)
    return image
def add_noise(image, noise_range=0.02):
    noise = np.random.randn(*image.shape) * 255 * noise_range
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return image
def add_shadow(image,range=0.1):
    h, w = image.shape
    x1, x2 = np.random.randint(0, w, 2)
    shadow_mask = np.zeros_like(image)
    cv2.fillPoly(shadow_mask, np.array([[[x1, 0], [x2, h], [w, h], [0, h]]]), 255)
    shadow_factor = np.random.uniform(1 - range, 1)
    image = np.where(shadow_mask == 255, image * shadow_factor, image)
    return image.astype(np.uint8)

def horizontal_flip(image, steering_angle):
    flipped_image = cv2.flip(image, 1)
    flipped_steering_angle = -steering_angle
    return flipped_image, flipped_steering_angle


def horizontal_shift(image, steering_angle, steering_shift_factor=0.2):
    #steering joy range [-1,1] positive steering left, negtive steering right
    i=np.random.randint(0,80)
    shifted_image=np.array(image[:, i:(400+i)])
    shifted_steering_angle=steering_angle-(40-i)/40*steering_shift_factor
    return shifted_image, shifted_steering_angle

def horizontal_rotate(image, steering_angle, steering_rotate_factor=0.4):
    #steering joy range [-1,1] positive steering left, negtive steering right
    i=np.random.randint(0,80)
    reduc_left=20*(i-40)/40 if i>40 else 0
    reduc_right=20*(40-i)/40 if i<40 else 0
    image=image[:, i:(400+i)]
    h, w = image.shape
    pt_A = np.array([0, reduc_left])
    pt_B = np.array([0, h-1-reduc_left])
    pt_C = np.array([w-1, h-1-reduc_right])
    pt_D = np.array([w-1, reduc_right])
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                            [0, h-1],
                            [w-1, h-1],
                            [w-1, 0]])
    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    rotated_image = cv2.warpPerspective(image,M,(w, h),flags=cv2.INTER_LINEAR)
    rotated_steering_angle=steering_angle-(40-i)/40*steering_rotate_factor

    return rotated_image, rotated_steering_angle

class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model = None
        self.model_folder = None
        self.script_folder = dirname(abspath(__file__))
        self.sorted_data_path = join(self.script_folder, self.cfg.dataset.sorted_data_path)
        #self.sorted_data_path = self.cfg.dataset.sorted_data_path
        self.dataset_idx_list = self.cfg.dataset.dataset_idx_list
        self.behavior_lists = self.cfg.dataset.behavior_lists
        self.dataset_mapping = self.cfg.dataset.dataset_mapping
        self.path_dict = self.get_path_dict(self.sorted_data_path) #save path loading time
        #self.wandb_run_id = wandb.util.generate_id()
        self.save_model_dir = join(self.script_folder, self.cfg.training.save_model_dir)
        self.latest_model_folder = self.get_latest_model_folder()
        np.random.seed(self.cfg.training.random_seed)
        tf.random.set_seed(self.cfg.training.random_seed)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H.%M.%S')
        self.model_folder = join(self.save_model_dir, f"{self.cfg.model.name}_{timestamp}")
        os.makedirs(self.model_folder, exist_ok=True)
        self.logs_folder = join(self.model_folder, self.cfg.wandb.log_dir)
        os.makedirs(self.logs_folder, exist_ok=True)
    # Initialize WandB
    def init_wandb(self):
        """
        Initialize WandB for experiment tracking.
        """
        wandb.init(
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            #id=self.wandb_run_id,
            #resume="must" if self.wandb_run_id else "allow",
            settings=wandb.Settings(console="wrap"),  # Logs all stdout messages
            config=OmegaConf.to_container(self.cfg, resolve=True),
            #sync_tensorboard=self.cfg.wandb.sync_tensorboard,
            dir=self.logs_folder,
            mode=self.cfg.wandb.mode
        )
    def build_model_from_config(self):
        if self.cfg.model.name == "SGrayBNPilotNet":
            self.model = SGrayBNPilotNet()
        elif self.cfg.model.name == "SGrayResPilotNet":
            self.model = SGrayResPilotNet()
        elif self.cfg.model.name == "SGrayMobileNetV3Small":
            self.model = SGrayMobileNetV3Small(pretrained=self.cfg.model.pretrained,trainable=self.cfg.model.trainable, training=self.cfg.model.training)
        elif self.cfg.model.name == "SGrayMobileNetV3Large":
            self.model = SGrayMobileNetV3Large(pretrained=self.cfg.model.pretrained,trainable=self.cfg.model.trainable, training=self.cfg.model.training)
        elif self.cfg.model.name == "SGrayMobileNetV3LargeD1":
            self.model = SGrayMobileNetV3LargeD1(pretrained=self.cfg.model.pretrained,trainable=self.cfg.model.trainable, training=self.cfg.model.training)
            #self.model.load_weights("/home/valkyrie/Documents/eglinton_datasorting_dual/saved_models_logs/SGrayMobileNetV3Large_2025-03-31-16.41.26/SGrayMobileNetV3Large.keras", skip_mismatch=True)
        elif self.cfg.model.name == "SGrayEfficientNetV2S":
            self.model = SGrayEfficientNetV2S(pretrained=self.cfg.model.pretrained,trainable=self.cfg.model.trainable, training=self.cfg.model.training)
        elif self.cfg.model.name == "SGrayEfficientNetV2SD1":
            self.model = SGrayEfficientNetV2SD1(pretrained=self.cfg.model.pretrained,trainable=self.cfg.model.trainable, training=self.cfg.model.training)
            self.model.load_weights("/home/valkyrie/Documents/eglinton_datasorting_dual/saved_models_logs/SGrayEfficientNetV2S_weights.h5", skip_mismatch=True, by_name=True)
        else:
            raise ValueError(f"Unsupported model name: {self.cfg.model.name}")
        optimizer_cfg = self.cfg.model.compile.optimizer
        optimizer_type = optimizer_cfg.get('type')
        optimizer_cfg = {k: v for k, v in optimizer_cfg.items() if k != "type"} # Remove the 'type' key
        optimizer = Adam(**optimizer_cfg) if optimizer_type == 'Adam' else None
        
        compile_cfg = self.cfg.model.compile
        #Convert ListConfig to a Regular Python List
        compile_cfg = {k: v if not isinstance(v, ListConfig) else list(v) for k, v in compile_cfg.items()}
        # self.model.compile(
        #     optimizer=optimizer,
        #     loss=[laplacian_nll_loss, laplacian_nll_loss],
        #     loss_weights=self.cfg.model.compile.loss_weights,
        #     metrics=[laplacian_nll_loss]
        # )

        self.model.compile(
            optimizer=optimizer,
            loss=compile_cfg['loss'],
            loss_weights=compile_cfg['loss_weights'],
            metrics=compile_cfg['metrics']
        )
        self.model.summary()

    def build_callbacks_from_config(self):
        """Builds callbacks dynamically from configuration."""

        callbacks = []
        for cb_cfg in self.cfg.training.callbacks:
            cb_type = cb_cfg.get('type')
            cb_cfg = {k: v for k, v in cb_cfg.items() if k != "type"} # Remove the 'type' key
            if cb_type == 'ModelCheckpoint':
                cb_cfg['filepath'] = join(self.model_folder, cb_cfg['filepath'])
                callbacks.append(ModelCheckpoint(**cb_cfg))
            elif cb_type == 'EarlyStopping':
                callbacks.append(EarlyStopping(**cb_cfg)) 
            elif cb_type == 'ReduceLROnPlateau':
                callbacks.append(ReduceLROnPlateau(**cb_cfg))
            elif cb_type == 'TensorBoard':
                cb_cfg['log_dir'] = self.logs_folder
                callbacks.append(TensorBoard(**cb_cfg))
            else:
                raise ValueError(f"Unsupported callback type: {cb_type}")

        return callbacks

    def process_dataset_from_config(self):
        """Processes dataset from configuration."""


        main_list = self.get_all_lsdirs(self.behavior_lists.main.list)
        dataset_lists = {}
        for cmd_key, cmd_data in self.behavior_lists.items():
            if cmd_key == "main":
                continue
            dataset_lists[cmd_key] = main_list[::cmd_data.main_div] + cmd_data.branch_multi * self.get_all_lsdirs(cmd_data.list)
            if self.cfg.dataset.shuffle:
                np.random.shuffle(dataset_lists[cmd_key])
        return dataset_lists

    def train_model_from_config(self, cmd_key="cmd_0"):
        """Builds and trains a model based on a Hydra configuration."""

        

        self.init_wandb()

        if self.latest_model_folder and self.cfg.training.load_checkpoint:
            checkpoint_path = join(self.latest_model_folder, f"{self.cfg.model.name}.keras")
            if exists(checkpoint_path):
                print(f"Loading model from checkpoint: {checkpoint_path}")
                self.model = load_model(checkpoint_path, safe_mode=False)
                if self.cfg.training.checkpoint_lr:
                    # Set the learning rate from the checkpoint
                    print(f"Setting learning rate from {BK.get_value(self.model.optimizer.learning_rate)} to {self.cfg.training.checkpoint_lr}")
                    self.model.optimizer.learning_rate.assign(self.cfg.training.checkpoint_lr)
            else:
                print(f"Checkpoint not found: {checkpoint_path}, building model from config")
                self.build_model_from_config()
        else:
            print("Building model from config")
            self.build_model_from_config()

        callbacks = self.build_callbacks_from_config()
        dataset = self.process_dataset_from_config()
        selected_dataset = dataset[cmd_key]

        batch_size = self.cfg.training.batch_size
        test_ratio = 1 - self.cfg.training.train_ratio - self.cfg.training.valid_ratio
        train_steps = round(self.cfg.training.train_coeff * len(selected_dataset) * self.cfg.training.train_ratio / batch_size)
        valid_steps = round(self.cfg.training.valid_coeff * len(selected_dataset) * self.cfg.training.valid_ratio / batch_size)
        test_steps = round(self.cfg.training.test_coeff * len(selected_dataset) * test_ratio / batch_size)
        train_batch = self.batch_generator(1, selected_dataset)
        valid_batch = self.batch_generator(2, selected_dataset)
        test_batch = self.batch_generator(3, selected_dataset) if test_ratio > 0 else None

        # Train the model
        history = self.model.fit(
            train_batch,
            epochs=self.cfg.training.epochs,
            validation_data=valid_batch,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            callbacks=[WandbMetricsLogger("epoch"), *callbacks],
            verbose=1,
            shuffle=self.cfg.training.shuffle
        )
        print(f"{self.cfg.model.name} {cmd_key} training completed")
        # # Save the model
        self.model.trainable = False
        model_stamp = f"{self.cfg.model.name}_{cmd_key}_{datetime.datetime.now().strftime('%Y-%m-%d-%H.%M.%S')}"
        
        run_model = tf.function(lambda x: self.model(x))
        input_signature = [tf.TensorSpec([1, 240, 400], self.model.inputs[0].dtype)]
        concrete_func = run_model.get_concrete_function(input_signature)
        #self.model.save(join(self.model_folder, model_stamp), save_format="tf", signatures=concrete_func)

        # Convert to ONNX
        onnx_model, _ = tf2onnx.convert.from_keras(self.model, input_signature, opset=13)
        onnx.save(onnx_model, join(self.model_folder, f"{model_stamp}.onnx"))

        # Convert to TFLite
        #converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func],self.model)
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        with open(join(self.model_folder, f"{model_stamp}.tflite"), 'wb') as f:
            f.write(tflite_model)

        # Evaluate
        if test_ratio > 0:
            test_loss = self.model.evaluate(test_batch, steps=test_steps, verbose=2)[0]
            wandb.log({"Test_loss": test_loss})

        wandb.finish()

    def batch_generator(self, subset_type: int, selected_data: list):
        """Batch generator for training, validation, and testing."""
        batch_size = self.cfg.training.batch_size
        train_ratio = self.cfg.training.train_ratio
        valid_ratio = self.cfg.training.valid_ratio

        if subset_type == 1:  # Train
            i, j = 0, train_ratio
        elif subset_type == 2:  # Validation
            i, j = train_ratio, train_ratio + valid_ratio
        elif subset_type == 3:  # Test
            i, j = train_ratio + valid_ratio, 1
        else:
            print('[Error!] Check subset type is correct!')
            sys.exit()
        
        indices = np.arange(round((j - i) * len(selected_data)))
        x, y1, y2 = [], [], []

        while True:
            np.random.shuffle(indices)
            for ind in indices:
                curr_ind = round(i * len(selected_data)) + (ind % round((j - i) * len(selected_data)))
                current_data = selected_data[curr_ind]
                data_array = np.load(join(self.sorted_data_path, current_data), allow_pickle=True)
                
                if len(data_array) == 8:
                    image, speed, steering = data_array[0], data_array[2], data_array[3]
                elif len(data_array) == 10:
                    image, speed, steering = data_array[0], data_array[8], data_array[9]
                else:
                    image, speed, steering = data_array[0], data_array[1], data_array[2]
                
                if self.cfg.augmentation.augment_data:
                    image, steering = self.data_augmentation(image, steering)
                if self.cfg.augmentation.horizontal_shift or self.cfg.augmentation.horizontal_rotate:
                    if self.cfg.augmentation.horizontal_rotate:
                        image, steering = horizontal_rotate(image, steering, self.cfg.augmentation.steering_rotate_factor)
                    else:
                        image, steering = horizontal_shift(image, steering, self.cfg.augmentation.steering_shift_factor)
                else:
                    image = image[:, 40:440]
                
                x.append(image)
                y1.append(speed) 
                y2.append(steering)
                
                if len(x) == batch_size:
                    yield np.asarray(x), (np.asarray(y1), np.asarray(y2))
                    x, y1, y2 = [], [], []

    def get_all_lsdirs(self, behavior_list):
        """Recursively get all dataset directories."""
        dir_list = []
        for dataset_idx in self.dataset_idx_list:
            for behavior in behavior_list:
                if exists(join(self.sorted_data_path, self.dataset_mapping[dataset_idx], behavior)):
                    dir_list = self.lsdirs(self.path_dict[self.dataset_mapping[dataset_idx]][behavior], dir_list)
        return dir_list
    
    def lsdirs(self, rootdir, img_list):
        """Recursively list directories."""
        for it in os.scandir(rootdir):
            if it.is_dir():
                self.lsdirs(it.path, img_list)
            elif it.is_file():
                img_list.append(it.path)
        return img_list

    def get_latest_model_folder(self):
        """Finds the most recent model folder."""
        model_folders = sorted([f for f in os.listdir(self.save_model_dir) if f.startswith(self.cfg.model.name)], 
                               key=lambda x: getmtime(join(self.save_model_dir, x)), reverse=True)
        return join(self.save_model_dir, model_folders[0]) if model_folders else None

    def get_path_dict(self, base_folder):
        """Get a dictionary of all subfolders and subsubfolders in the base folder."""
        return {subfolder_name: {subsubfolder_name: join(join(base_folder, subfolder_name), subsubfolder_name) 
                                for subsubfolder_name in os.listdir(join(base_folder, subfolder_name))} 
                for subfolder_name in os.listdir(base_folder)}

    def data_augmentation(self, image, steering_angle):
        augment_prob = self.cfg.augmentation.augment_prob
        if(np.random.rand() < augment_prob) and self.cfg.augmentation.add_blur:
            image= add_blur(image,self.cfg.augmentation.blur_range)
        if(np.random.rand() < augment_prob) and self.cfg.augmentation.adjust_brightness:
            image= adjust_brightness(image,self.cfg.augmentation.brightness_range)
        if(np.random.rand() < augment_prob) and self.cfg.augmentation.add_noise:
            image= add_noise(image,self.cfg.augmentation.noise_range)
        if(np.random.rand() < augment_prob) and self.cfg.augmentation.add_shadow:
            image= add_shadow(image,self.cfg.augmentation.shadow_range)
        if(np.random.rand() < augment_prob) and self.cfg.augmentation.horizontal_flip:
            image, steering_angle= horizontal_flip(image, steering_angle)
        return image, steering_angle

@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)  # Instantiate the class
    for cmd_key in cfg.training.cmd_list:
        trainer.train_model_from_config(cmd_key)
    

if __name__ == "__main__":
    main()
