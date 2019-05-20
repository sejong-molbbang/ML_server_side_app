from django.urls import path

from . import views

urlpatterns = [
    path('', views.ListUser.as_view()),
    path('signup', views.signup, name='signup'),
    path('signin', views.signin, name='signin'),
    path('<int:pk>/', views.DetailUser.as_view()),
]