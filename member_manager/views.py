from django.shortcuts import render
from rest_framework import generics
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import login, authenticate
from django.contrib.auth.models import User
from .models import UserProfile
from django.http import JsonResponse
import json
import logging
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
            logger.debug('{} {}'.format(email, passwd))
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
        
