from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import math
import cv2
import copy
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import os

from sklearn.metrics import average_precision_score

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.objectives import categorical_crossentropy
from keras.applications import inception_resnet_v2

from keras.models import Model
from keras.utils import generic_utils
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers

from .config import Config
from .functions import get_anchor_gt, get_data, rpn_to_roi, calc_iou, get_new_img_size, apply_regr, non_max_suppression_fast
from .layers import classifier_layer, RoiPoolingConv, rpn_layer
from .vgg16 import get_img_output_length, nn_base

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

def rpn_loss_regr(num_anchors):
    """Loss function for rpn regression
    Args:
        num_anchors: number of anchors (9 in here)
    Returns:
        Smooth L1 loss function 
                           0.5*x*x (if x_abs < 1)
                           x_abx - 0.5 (otherwise)
    """
    def rpn_loss_regr_fixed_num(y_true, y_pred):

        # x is the difference between true value and predicted vaue
        x = y_true[:, :, :, 4 * num_anchors:] - y_pred

        # absolute value of x
        x_abs = K.abs(x)

        # If x_abs <= 1.0, x_bool = 1
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

        return lambda_rpn_regr * K.sum(
            y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])

    return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
    """Loss function for rpn classification
    Args:
        num_anchors: number of anchors (9 in here)
        y_true[:, :, :, :9]: [0,1,0,0,0,0,0,1,0] means only the second and the eighth box is valid which contains pos or neg anchor => isValid
        y_true[:, :, :, 9:]: [0,1,0,0,0,0,0,0,0] means the second box is pos and eighth box is negative
    Returns:
        lambda * sum((binary_crossentropy(isValid*y_pred,y_true))) / N
    """
    def rpn_loss_cls_fixed_num(y_true, y_pred):

            return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])

    return rpn_loss_cls_fixed_num


def class_loss_regr(num_classes):
    """Loss function for rpn regression
    Args:
        num_anchors: number of anchors (9 in here)
    Returns:
        Smooth L1 loss function 
                           0.5*x*x (if x_abs < 1)
                           x_abx - 0.5 (otherwise)
    """
    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4*num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs
        , 1.0), 'float32')
        return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
    return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))


def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

def draw_rectangle(img, rect):
    (x1, y1, x2, y2) = rect
    
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 1)
    dst = img[y1:y2, x1:x2]
    dst = cv2.medianBlur(dst, 19)
    img[y1:y2, x1:x2]= dst


class FasterRCNN(object):

    def __init__(self, config):
        self.name = 'faster_r_cnn'
        self.C = config

    def feature_extractor(self, img_input, name='vgg16'):
        if name == 'vgg16':
            return nn_base(img_input, trainable=False)
        elif name == 'Inception_ResnetV2':
            return inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=img_input, input_shape=(None, None, 3)).output
        else:
            print('Not supported model')
            return None

    
    def build_model(self, classes, num_features=512):
        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, num_features)

        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(self.C.num_rois, 4))
        feature_map_input = Input(shape=input_shape_features)

        # define the base network (VGG here, can be Resnet50, Inception, etc)
        shared_layers = self.feature_extractor(img_input)
    
        # define the RPN, built on the base layers
        self.num_anchors = len(self.C.anchor_box_scales) * len(self.C.anchor_box_ratios) # 9
        rpn_layers = rpn_layer(shared_layers, self.num_anchors)

        classifier = classifier_layer(feature_map_input, roi_input, self.C.num_rois, nb_classes=len(classes))

        self.model_rpn = Model(img_input, rpn_layers)
        self.model_classifier_only = Model([feature_map_input, roi_input], classifier)
        
        print('Loading weights from {}'.format(self.C.model_path))
        self.model_rpn.load_weights(self.C.model_path, by_name=True)
        self.model_classifier_only.load_weights(self.C.model_path, by_name=True)
        print('Complete loading model')


    def predict(self, img, video_stream, face_threshold=0.78, plate_threshold=0.72):
        st = time.time()
        class_to_color = { self.C.class_mapping[v]: np.random.randint(0, 255, 3) for v in self.C.class_mapping }

        X, ratio = format_img(img, self.C)
        X = np.transpose(X, (0, 2, 3, 1))
        
        # get output layer Y1, Y2 from the RPN and the feature maps F
        # Y1: y_rpn_cls
        # Y2: y_rpn_regr
        [Y1, Y2, F] = self.model_rpn.predict(X)

        # Get bboxes by applying NMS 
        # R.shape = (300, 4)
        R = rpn_to_roi(Y1, Y2, self.C, K.image_dim_ordering(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0]//self.C.num_rois + 1):
            ROIs = np.expand_dims(R[self.C.num_rois*jk:self.C.num_rois*(jk+1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0]//C.num_rois:
                #pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = self.model_classifier_only.predict([F, ROIs])

            # Calculate bboxes coordinates on resized image
            for ii in range(P_cls.shape[1]):
                # Ignore 'bg' class

                cls_name = self.C.class_mapping[np.argmax(P_cls[0, ii, :])]
                if cls_name == 'Vehicle registration plate':
                    threshold = plate_threshold

                if np.max(P_cls[0, ii, :]) < threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                    tx /= self.C.classifier_regr_std[0]
                    ty /= self.C.classifier_regr_std[1]
                    tw /= self.C.classifier_regr_std[2]
                    th /= self.C.classifier_regr_std[3]
                    x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([self.C.rpn_stride*x, self.C.rpn_stride*y, self.C.rpn_stride*(x+w), self.C.rpn_stride*(y+h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        all_dets = []

        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk,:]

                # Calculate real coordinates on original image
                #(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
                draw_rectangle(img, get_real_coordinates(ratio, x1, y1, x2, y2))

                #cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),4)

                textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
                all_dets.append((key,100*new_probs[jk]))
                

        print('Elapsed time = {}'.format(time.time() - st))
        print(all_dets)
        if not video_stream == None:
            video_stream.write(img)
        

