from django.core.management.base import BaseCommand
from django.core.cache import cache

from .faster_rcnn.model import FasterRCNN
from .faster_rcnn.config import Config

import os
import optparse
import numpy as np
import json
import pandas as pd
import requests

class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('--input', dest='input_path', action='store',
                             help='Input video path')
        parser.add_argument('--output', dest='output_path', action='store',
                             help='Output video path')
    
    def load_model(self):
        # Create the config
        C = Config()
        C.use_horizontal_flips = False
        C.use_vertical_flips = False
        C.rot_90 = False
        #C.model_path = options['model_path']
        C.model_path = os.path.join(os.getcwd(), 'predict_app\\management\\commands\\faster_rcnn\\weights\\model_frcnn_vgg_1.278.hdf5')
        C.base_net_weights = os.path.join(os.getcwd(), 'predict_app\\management\\commands\\faster_rcnn\\weights\\vgg16_weights_tf_dim_ordering_tf_kernels.h5')
        class_mapping = {0 : 'Human face', 1:'Vehicle registration plate', 2:'bg'}
        C.class_mapping = class_mapping

        rcnn = FasterRCNN(C)
        rcnn.build_model(class_mapping.keys())

        return rcnn

    def handle(self, *args, **options):
        input_path = options['input_path']
        output_path = options['output_path']

        dl_model = self.load_model()
        
        #cached ( 이거 처음 켤 때  딥러닝 모델을 캐시에 저장하는거다)
        cache.clear()
        cache.set('input_path', input_path)
        cache.set('output_path', output_path)
        cache.set('dl_model', dl_model)

