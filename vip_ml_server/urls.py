"""vip_ml_server URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
import os
#from .yolo_ensemble.detection_model import Yolo_Ensemble
#from .yolo_ensemble.face_recognize import Face_Recognition

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('member_manager.urls')),
]

# Load deep learning model
global detect_model
base_path = os.getcwd() + '/model_data/'
#detect_model = Yolo_Ensemble()
#detect_model.load_model(base_path + 'yolo_face_model.h5', base_path + 'yolo_plate_model.h5')

