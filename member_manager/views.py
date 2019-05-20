from django.shortcuts import render
from rest_framework import generics

from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import login, authenticate
from django.http import JsonResponse
from .models import User
from .managers import UserManager
from .serializers import UserSerializer

class ListUser(generics.ListCreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer

class DetailUser(generics.RetrieveUpdateDestroyAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer

def signup(request):
    try:
        new_user = UserManager.create_user(request.POST.get('email'), request.POST.get('passwd'))
        login(request, new_user)
        return JsonResponse({'result': 'success'})
    except:
        return JsonResponse({'result': 'fail'})

@csrf_exempt
def signin(request):
    try:
        email = request.POST.get('email')
        passwd = request.POST.get('passwd')
        user = authenticate(email=email, password=passwd)
        if user is not None:
            login(request, user)
            return JsonResponse({'result': 'success'})
        else:
            return JsonResponse({'result': 'fail'})
    except:
        return JsonResponse({'result': 'fail'})
        
