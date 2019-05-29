from django.urls import path

from . import views

urlpatterns = [
    path('signup', views.signup, name='signup'),
    path('signin', views.signin, name='signin'),
    #path('mainscreen', views.mainscreen, name ='mainscreen')
]