import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
import os

def ResNet101V2(input_shape, pretrained='imagenet'):

    base_model = tf.keras.applications.ResNet101V2(
        include_top=False,
        weights=pretrained,
        input_shape=input_shape,
        pooling='avg',
    )

    return base_model

def ResNet152V2(input_shape, pretrained='imagenet'):

    base_model = tf.keras.applications.ResNet152V2(
        include_top=False,
        weights=pretrained,
        input_shape=input_shape,
        pooling='avg',
    )

    return base_model

def ResNet50V2(input_shape, pretrained='imagenet'):

    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights=pretrained,
        input_shape=input_shape,
        pooling='avg',
    )

    return base_model

def InceptionV3(input_shape, pretrained='imagenet'):

    base_model = tf.keras.applications.InceptionV3(
        include_top=False,
        weights=pretrained,
        input_shape=input_shape,
        pooling='avg',
    )

    return base_model

def MobileNetV2(input_shape, pretrained='imagenet'):

    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=pretrained,
        input_shape=input_shape,
        pooling='avg',
    )

    return base_model

def DenseNet121(input_shape, pretrained='imagenet'):

    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights=pretrained,
        input_shape=input_shape,
        pooling='avg',
    )

    return base_model

def DenseNet169(input_shape, pretrained='imagenet'):

    base_model = tf.keras.applications.DenseNet169(
        include_top=False,
        weights=pretrained,
        input_shape=input_shape,
        pooling='avg',
    )

    return base_model

def NASNetMobile(input_shape, pretrained='imagenet'):

    base_model = tf.keras.applications.NASNetMobile(
        include_top=False,
        weights=pretrained,
        input_shape=input_shape,
        pooling='avg',
    )

    return base_model

def EfficientNetB0(input_shape, pretrained='imagenet'):

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=pretrained,
        input_shape=input_shape,
        pooling='avg',
    )

    return base_model

def EfficientNetB4(input_shape, pretrained='imagenet'):

    base_model = tf.keras.applications.EfficientNetB4(
        include_top=False,
        weights=pretrained,
        input_shape=input_shape,
        pooling='avg',
    )

    return base_model

def EfficientNetB3(input_shape, pretrained='imagenet'):

    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights=pretrained,
        input_shape=input_shape,
        pooling='avg',
    )

    return base_model

def build_keypoint_model(base_model, n_outputs):

    inputs = base_model.input
    flatten = Flatten()(base_model.output)

    outputs = Dense(flatten.shape[1] // 2, kernel_initializer='he_uniform', activation='relu')(flatten)
    outputs = Dense(n_outputs)(outputs)

    model = Model(inputs, outputs)

    return model

def save_model(model, save_path):
    model.save(save_path)

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    print(model.summary())
    return model