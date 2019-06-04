from django.urls import path

from . import views

urlpatterns = [
    path('masking', views.model.masking, name='masking'),
    path('progress', views.model.return_progress, name='progress')
]