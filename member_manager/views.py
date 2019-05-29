from django.shortcuts import render
from rest_framework import generics

from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import login, authenticate
from django.http import JsonResponse
from .models import User
from .managers import UserManager

from .serializers import UserSerializer
import json
import logging
logger = logging.getLogger(__name__)


@csrf_exempt
def signup(request):
    try:
        body = json.loads(request.body)
        email = body['id']
        passwd = body['passwd']
        logger.debug('{} {}'.format(email, passwd))
        
        new_user = UserManager.create_user(email, passwd)
        login(request, new_user, backend='django.contrib.auth.backends.ModelBackend')
        return JsonResponse({'result': 'success'})
    except Exception as e:
        logger.debug(e)
        return JsonResponse({'result': 'fail'})

'''
def signup(request):
    userName = request.REQUEST.get('username', None)
    userPass = request.REQUEST.get('password', None)
    userMail = request.REQUEST.get('email', None)

    # TODO: check if already existed

    **user = User.objects.create_user(userName, userMail, userPass)**
    user.save()

    return render_to_response('home.html', context_instance=RequestContext(request))
'''

@csrf_exempt
# 이게 req가 오면 res만들어서 보내는건데 success, fail 을 보낸다
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
        
