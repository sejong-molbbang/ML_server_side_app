import os
import numpy as np
import cv2
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from .yolo3.model import yolo_eval, yolo_body
from .yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

class Yolo_Ensemble(object):
    def __init__(self, score=0.2, iou=0.45, gpu_num=1):
        #self.sess = K.get_session()
        config = self.keras_resouce()
        self.sess = tf.Session(config=config)
        self.graph = self.sess.graph
        set_session(self.sess)
        
        self.anchors = np.array([10.0,13.0, 16.0,30.0, 33.0,23.0, 30.0,61.0, 62.0,45.0, 59.0,119.0, 116.0,90.0, 156.0,198.0, 373.0,326.0]).reshape(-1, 2)
        self.model_image_size = (480,480)
        self.score = score
        self.iou = iou
        self.input_image_shape = K.placeholder(shape=(2,))
        self.gpu_num = gpu_num

    def keras_resouce(self):
        num_cores=4
        config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                                inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
                                device_count={'CPU': 1, 'GPU': 1})
        config.gpu_options.allow_growth = True
        return config
    
    def generate(self, model_path, anchors, score):
        print(model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        num_anchors = len(anchors)
        num_classes = 1

        model = None
        try:
            model = load_model(model_path, compile=False)
        except:
            model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            model.load_weights(model_path)
        
        
        if self.gpu_num >=2:
            model = multi_gpu_model(model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(model.output, anchors,
                num_classes, self.input_image_shape,
                score_threshold=score, iou_threshold=self.iou)

        return {'model':model, 'boxes':boxes, 'scores': scores, 'classes': classes}


    def load_model(self, yolo_face_path, yolo_plate_path):
        self.gen_face_model = self.generate(yolo_face_path, self.anchors, 0.2)
        self.gen_plate_model = self.generate(yolo_plate_path, self.anchors, 0.2)

    def get_graph(self):
        #self.sess = K.get_session()
        self.graph = tf.get_default_graph()

    def detect(self, frame, only_face=False):
        faces = []
        plates = []
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'

        new_image_size = (frame.width - (frame.width % 32),
                          frame.height - (frame.height % 32))
        boxed_frame = letterbox_image(frame, new_image_size)
        image_data = np.array(boxed_frame, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        
        with self.graph.as_default():
            face_boxes, face_scores, _ = self.sess.run(
                [self.gen_face_model['boxes'], self.gen_face_model['scores'], self.gen_face_model['classes']],
                feed_dict={
                    self.gen_face_model['model'].input: image_data,
                    self.input_image_shape: [frame.size[1], frame.size[0]],
                    #K.learning_phase(): 0
                })

            # face detection
            for i, score in enumerate(face_scores):
                box = face_boxes[i]
                y1, x1, y2, x2 = box
                y1 = max(0, np.floor(y1 + 2.5).astype('int32'))
                x1 = max(0, np.floor(x1 + 2.5).astype('int32'))
                y2 = min(frame.size[1], np.floor(y2 + 2.5).astype('int32'))
                x2 = min(frame.size[0], np.floor(x2 + 2.5).astype('int32'))
                faces.append((x1, y1, x2, y2))

            if only_face == False:
                plate_boxes, plate_scores, _ = self.sess.run(
                    [self.gen_plate_model['boxes'], self.gen_plate_model['scores'], self.gen_plate_model['classes']],
                    feed_dict={
                        self.gen_plate_model['model'].input: image_data,
                        self.input_image_shape: [frame.size[1], frame.size[0]],
                        #K.learning_phase(): 0
                    })
                
                # plate detection
                for i, score in enumerate(plate_scores):
                    box = plate_boxes[i]
                    y1, x1, y2, x2 = box
                    y1 = max(0, np.floor(y1 + 2.5).astype('int32'))
                    x1 = max(0, np.floor(x1 + 2.5).astype('int32'))
                    y2 = min(frame.size[1], np.floor(y2 + 2.5).astype('int32'))
                    x2 = min(frame.size[0], np.floor(x2 + 2.5).astype('int32'))
                    plates.append((x1, y1, x2, y2))
       
        return faces, plates


    def close_session(self):
        self.sess.close()