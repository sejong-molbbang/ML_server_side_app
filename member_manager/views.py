from django.shortcuts import render
from rest_framework import generics, parsers, permissions
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.models import User
from .models import UserProfile
from django.http import JsonResponse
from .serializers import ProfilePictureSerializer
import json
import logging
from django.core.files.storage import FileSystemStorage
logger = logging.getLogger(__name__)

@csrf_exempt
def signup(request):
    try:
        if request.method == 'POST':
            body = json.loads(request.body)
            email = body['id']
            passwd = body['passwd']
            if User.objects.filter(username=email).exists():
                return JsonResponse({'result': 'registed'})
            user = User.objects.create_user(username=email, password=passwd)
            uprofile = UserProfile()
            uprofile.user = user
            uprofile.save(create=True)
                        
            return JsonResponse({'result': 'success'})
        else:
            return JsonResponse({'return': 'fail'})
    except:
        return JsonResponse({'result': 'fail'})

@csrf_exempt
def signin(request):
    try:
        if request.method == 'POST':
            body = json.loads(request.body)
            email = body['id']
            passwd = body['passwd']
            user = authenticate(username=email, password=passwd)
            if user is not None:
                login(request, user)
                return JsonResponse({'result': 'success'})
                
        return JsonResponse({'result': 'fail'})
    except:
        return JsonResponse({'result': 'fail'})

@csrf_exempt
def signout(request):
    logout(request)
    return JsonResponse({'result':  'success'})


@csrf_exempt
def image_upload(request):
    if request.method == 'POST':
        logger.debug(request.FILES)
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        uploaded_file_url = fs.url(filename)
        return JsonResponse({'url': uploaded_file_url})
