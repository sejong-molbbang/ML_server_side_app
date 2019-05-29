from django.urls import path

from . import views
# 이게 프록시를 저장해놔서 /api/signup 이런식으로 된다 그러니까 아래처럼 기능명처럼만 적어두면 url이 된다.

urlpatterns = [
    #path('signout',view.signout(구현한 로그아웃 함수),name='signout'),
    path('signup', views.signup, name='signup'),    # views.signup을 사용한다. request를 보낸다.
    path('signin', views.signin, name='signin'),
    #path('mainscreen', views.mainscreen, name ='mainscreen')
]