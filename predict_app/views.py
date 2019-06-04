from django.http import HttpResponse
from django.template import RequestContext,loader
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from PIL import Image
import json
import cv2
import os
from .yolo_ensemble.detection_model import Yolo_Ensemble
from .yolo_ensemble.face_recognize import Face_Recognition
import logging
logger = logging.getLogger(__name__)

# Load deep learning model
base_path = os.getcwd() + '/model_data/'
detect_model = Yolo_Ensemble()
detect_model.load_model(base_path + 'yolo_face_model.h5', base_path + 'yolo_plate_model.h5')
face_recog_model = Face_Recognition()

class PredictModel(object):
    
    def __init__(self):
        self.progress = 0

    def mask_rectangle(self, img, rect):
        (x1, y1, x2, y2) = rect
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 0)
        dst = img[y1:y2, x1:x2]
        dst = cv2.GaussianBlur(dst, (99,99), 25)
        img[y1:y2, x1:x2]= dst

    @csrf_exempt
    def masking(self, request):
        if request.method == 'POST':
            body = json.loads(request.body)
            video_path = 'D:/python_projects/vip_ml_server' + body['video']

            only_face = body['only_face']
            images = body['images']

            recognize = False
            if len(images) != 0:
                recognize = True
                email = body['email']
                face_recog_model.set_email(email)       
            
            vid = cv2.VideoCapture(video_path)

            if not vid.isOpened():
                return JsonResponse({"result": "video upload error"})
            
            video_FourCC    = cv2.VideoWriter_fourcc(*"mp4v")
            video_fps       = vid.get(cv2.CAP_PROP_FPS)
            video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            current_frame = 0
            length = float(int(vid.get(cv2.CAP_PROP_FRAME_COUNT)))
            wait_time = int(1000/video_fps)
            
            output_path = video_path[:-4] + '_result.mp4'

            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

            detect_model.get_graph()

            while (vid.isOpened()):
                return_value, frame = vid.read()
                
                if not return_value:
                    break

                image = Image.fromarray(frame)
                face_boxes, plate_boxes = detect_model.detect(image, only_face)
                
                # recognize faces
                if recognize:
                    face_boxes = face_recog_model.recognize(frame, face_boxes)

                predicted = face_boxes + plate_boxes

                for box in predicted:
                    self.mask_rectangle(frame, box)
                
                out.write(frame)
                
                current_frame += 1
                self.progress = (int)((float(current_frame) / length) * 100)
                
                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    break
        
            #detect_model.close_session()

            return JsonResponse({"result": "complete", "url": body["video"] + '_result.mp4'})


    @csrf_exempt
    def return_progress(self, request):
        return JsonResponse({'progress': self.progress})

model = PredictModel()