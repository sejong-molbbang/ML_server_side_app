import face_recognition
import os
from itertools import compress

class Face_Recognition(object):
    
    def load_image_encodings(self, images):
        self.encodings = []
        for filename in images:
            img  = face_recognition.load_image_file("D:/python_projects/vip_ml_server" + filename)
            encoding = face_recognition.face_encodings(img)[0]
            self.encodings.append(encoding)
    
    def recognize(self, image, boxes):
        new_boxes = []
        mask = [True] * len(boxes)
        for box in boxes:
            (x1, y1, x2, y2) = box
            new_boxes.append((y1, x2, y2, x1))

        face_encodings = face_recognition.face_encodings(image, new_boxes)
        for i, face_encoding in enumerate(face_encodings):
            match = face_recognition.compare_faces(self.encodings, face_encoding, tolerance=0.48)
            if max(match):
                mask[i] = False
        return list(compress(boxes, mask))
            
