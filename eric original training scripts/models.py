#!/usr/bin/env python
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Layer, Input, Lambda, Cropping2D, Conv2D, BatchNormalization,
                                     Activation, Add, Flatten, Dense, Dropout, Concatenate, GlobalAveragePooling2D)
from tensorflow.keras import backend as BK
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large,EfficientNetV2S
import numpy as np
import datetime
import cv2
import sys
import os
from os.path import join, exists, dirname, abspath, getmtime

class ResidualBlock(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='same', activation='elu', name=None, **kwargs):
        if name is None:
            name = f"residual_block_{filters}"
        super().__init__(name=name, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation

        # Main convolutional path
        self.conv1 = Conv2D(filters, kernel_size, strides=strides, padding=padding, name=f"{name}_conv1")
        self.bn1 = BatchNormalization(name=f"{name}_bn1")
        self.act1 = Activation(activation, name=f"{name}_act1")

        self.conv2 = Conv2D(filters, kernel_size, strides=(1, 1), padding=padding, name=f"{name}_conv2")
        self.bn2 = BatchNormalization(name=f"{name}_bn2")

        # Shortcut connection (identity or projection)
        if strides != (1, 1) or kwargs.get("input_filters", filters) != filters:
            self.shortcut = Conv2D(filters, (1, 1), strides=strides, padding="same", name=f"{name}_shortcut")
            self.shortcut_bn = BatchNormalization(name=f"{name}_shortcut_bn")
        else:
            self.shortcut = None  # Identity mapping

        self.add = Add(name=f"{name}_add")
        self.act2 = Activation(activation, name=f"{name}_act2")

    @tf.function  # Traces the function to prevent untraced warnings
    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # Apply shortcut transformation if necessary
        shortcut = self.shortcut(inputs) if self.shortcut else inputs
        if self.shortcut:
            shortcut = self.shortcut_bn(shortcut)

        x = self.add([x, shortcut])  # Residual connection
        x = self.act2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "activation": self.activation
        })
        return config   
    
    @classmethod
    def from_config(cls, config):
        """Ensures model can be reloaded correctly."""
        return cls(**config)
    
def SGrayBNPilotNet():
    input_shape = Input(shape=(240, 400, 1),dtype='float32', name='Img_Input')
    x = input_shape
    x = (Lambda(lambda x: (x/127.5 - 1.0), name='Normalisation'))(x)
    x = (Conv2D(24, (5, 5), strides=(2, 2), activation='elu', padding='valid', name='Conv_1'))(x)
    x = (BatchNormalization())(x)
    x = (Conv2D(36, (5, 5), strides=(2, 2), activation='elu', padding='valid', name='Conv_2'))(x)
    x = (BatchNormalization())(x)
    x = (Conv2D(48, (5, 5), strides=(2, 2), activation='elu', padding='valid', name='Conv_3'))(x)
    x = (BatchNormalization())(x)
    x = (Conv2D(64, (3, 3), strides=(1, 1), activation='elu', padding='valid', name='Conv_4'))(x)
    x = (BatchNormalization())(x)
    x = (Conv2D(64, (3, 3), strides=(1, 1), activation='elu', padding='valid', name='Conv_5'))(x)
    x = (Flatten())(x)
    x = (Dropout(0.2))(x)
    x = (Dense(100, activation='elu'))(x)
    x = (Dense(50, activation='elu'))(x)
    x = (Dense(10, activation='elu'))(x)
    speed = (Dense(1, name='Speed'))(x)
    steering = (Dense(1, name='Steering'))(x)
    model = Model(inputs = [input_shape], outputs = [speed, steering])
    return model

def SGrayResPilotNet():
    input_shape = Input(shape=(240, 400, 1),dtype='float32', name='Img_Input')
    x = input_shape
    x = (Lambda(lambda x: (x/127.5 - 1.0), name='Normalisation'))(x)
    x = (Conv2D(24, (5, 5), strides=(2, 2), activation='elu', padding='same', name='Conv_1'))(x)
    x = (ResidualBlock(36, (5, 5), strides=(2, 2), activation='elu', padding='same', name='ResBlock_1'))(x)
    x = (ResidualBlock(48, (5, 5), strides=(2, 2), activation='elu', padding='same', name='ResBlock_2'))(x)
    x = (ResidualBlock(64, (3, 3), strides=(1, 1), activation='elu', padding='same', name='ResBlock_3'))(x)
    x = (ResidualBlock(64, (3, 3), strides=(1, 1), activation='elu', padding='same', name='ResBlock_4'))(x)
    x = (Flatten())(x)
    x = (Dropout(0.2))(x)
    x = (Dense(100, activation='elu'))(x)
    x = (Dense(50, activation='elu'))(x)
    x = (Dense(10, activation='elu'))(x)
    speed = (Dense(1, name='Speed'))(x)
    steering = (Dense(1, name='Steering'))(x)
    model = Model(inputs = [input_shape], outputs = [speed, steering])
    return model

def SGrayMobileNetV3Small(pretrained=True, trainable=False, training=False):
    input_shape = Input(shape=(240, 400, 1),dtype='float32', name='Img_Input')
    x = input_shape
    x = (Lambda(lambda x: (x/127.5 - 1.0), name='Lambda_Normalisation'))(x)
    x = (Conv2D(3, (1, 1), strides=(1, 1), activation=None, padding='same', name='GrayConv_1'))(x) # Convert grayscale to 3 channels
    # Load MobileNetV3Small as a feature extractor
    base_model = MobileNetV3Small(
        input_shape=(240, 400, 3),  # MobileNet expects 3 channels
        weights="imagenet" if pretrained else None,
        include_top=False
    )
    # Freeze the base model if using pretrained weights
    base_model.trainable = trainable
    
    x = base_model(x, training=training)
    x = GlobalAveragePooling2D()(x)  # Reduce to a feature vecto
    #x = (Flatten())(x)
    x = (Dropout(0.2))(x)
    x = (Dense(100, activation='relu'))(x)
    x = (Dense(50, activation='relu'))(x)
    x = (Dense(10, activation='relu'))(x)
    speed = (Dense(1, name='Speed'))(x)
    steering = (Dense(1, name='Steering'))(x)
    model = Model(inputs = [input_shape], outputs = [speed, steering])
    return model

def SGrayMobileNetV3Large(pretrained=True, trainable=False, training=False):
    input_shape = Input(shape=(240, 400, 1),dtype='float32', name='Img_Input')
    x = input_shape
    x = (Lambda(lambda x: (x/127.5 - 1.0), name='Lambda_Normalisation'))(x)
    x = (Conv2D(3, (1, 1), strides=(1, 1), activation=None, padding='same', name='GrayConv_1'))(x) # Convert grayscale to 3 channels
    # Load MobileNetV3Small as a feature extractor
    base_model = MobileNetV3Large(
        input_shape=(240, 400, 3),  # MobileNet expects 3 channels
        weights="imagenet" if pretrained else None,
        include_top=False
    )
    # Freeze the base model if using pretrained weights
    base_model.trainable = trainable
    
    x = base_model(x, training=training)
    x = GlobalAveragePooling2D()(x)  # Reduce to a feature vecto
    #x = (Flatten())(x)
    x = (Dropout(0.2))(x)
    x = (Dense(100, activation='relu'))(x)
    x = (Dense(50, activation='relu'))(x)
    x = (Dense(10, activation='relu'))(x)
    speed = (Dense(1, name='Speed'))(x)
    steering = (Dense(1, name='Steering'))(x)
    model = Model(inputs = [input_shape], outputs = [speed, steering])
    return model

def SGrayMobileNetV3LargeD1(pretrained=True, trainable=False, training=False):
    input_shape = Input(shape=(240, 400, 1),dtype='float32', name='Img_Input')
    x = input_shape
    x = (Lambda(lambda x: (x/127.5 - 1.0), name='Lambda_Normalisation'))(x)
    x = (Conv2D(3, (1, 1), strides=(1, 1), activation=None, padding='same', name='GrayConv_1'))(x) # Convert grayscale to 3 channels
    # Load MobileNetV3Small as a feature extractor
    base_model = MobileNetV3Large(
        input_shape=(240, 400, 3),  # MobileNet expects 3 channels
        weights="imagenet" if pretrained else None,
        include_top=False
    )
    # Freeze the base model if using pretrained weights
    base_model.trainable = trainable
    
    x = base_model(x, training=training)
    x = GlobalAveragePooling2D()(x)  # Reduce to a feature vecto
    #x = (Flatten())(x)
    x = (Dropout(0.2))(x)
    x = (Dense(200, activation='relu'))(x)
    x = (Dense(100, activation='relu'))(x)
    x = (Dense(50, activation='relu'))(x)
    x = (Dense(10, activation='relu'))(x)
    speed = (Dense(1, name='Speed'))(x)
    steering = (Dense(1, name='Steering'))(x)
    model = Model(inputs = [input_shape], outputs = [speed, steering])
    return model

def SGrayEfficientNetV2S(pretrained=True, trainable=False, training=False):
    input_shape = Input(shape=(240, 400, 1),dtype='float32', name='Img_Input')
    x = input_shape
    x = (Lambda(lambda x: (x/127.5 - 1.0), name='Lambda_Normalisation'))(x)
    x = (Conv2D(3, (1, 1), strides=(1, 1), activation=None, padding='same', name='GrayConv_1'))(x) # Convert grayscale to 3 channels

    base_model = EfficientNetV2S(
        input_shape=(240, 400, 3),  # MobileNet expects 3 channels
        weights="imagenet" if pretrained else None,
        include_top=False
    )
    # Freeze the base model if using pretrained weights
    base_model.trainable = trainable
    
    x = base_model(x, training=training)
    x = GlobalAveragePooling2D()(x)  # Reduce to a feature vecto
    #x = (Flatten())(x)
    x = (Dropout(0.2))(x)
    x = (Dense(100, activation='relu'))(x)
    x = (Dense(50, activation='relu'))(x)
    x = (Dense(10, activation='relu'))(x)
    speed = (Dense(1, name='Speed'))(x)
    steering = (Dense(1, name='Steering'))(x)
    model = Model(inputs = [input_shape], outputs = [speed, steering])
    return model

def SGrayEfficientNetV2SD1(pretrained=True, trainable=False, training=False):
    input_shape = Input(shape=(240, 400, 1),dtype='float32', name='Img_Input')
    x = input_shape
    x = (Lambda(lambda x: (x/127.5 - 1.0), name='Lambda_Normalisation'))(x)
    x = (Conv2D(3, (1, 1), strides=(1, 1), activation=None, padding='same', name='GrayConv_1'))(x) # Convert grayscale to 3 channels

    base_model = EfficientNetV2S(
        input_shape=(240, 400, 3),  # MobileNet expects 3 channels
        weights="imagenet" if pretrained else None,
        include_top=False
    )
    # Freeze the base model if using pretrained weights
    base_model.trainable = trainable
    
    x = base_model(x, training=training)
    x = GlobalAveragePooling2D()(x)  # Reduce to a feature vecto
    #x = (Flatten())(x)
    x = (Dropout(0.2))(x)
    x = (Dense(200, activation='relu'))(x)
    x = (Dense(100, activation='relu'))(x)
    x = (Dense(50, activation='relu'))(x)
    x = (Dense(10, activation='relu'))(x)
    speed = (Dense(1, name='Speed'))(x)
    steering = (Dense(1, name='Steering'))(x)
    model = Model(inputs = [input_shape], outputs = [speed, steering])
    return model