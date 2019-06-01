from __future__ import unicode_literals

from django.db import models  
from django.core.mail import send_mail  
from django.utils.translation import ugettext_lazy as _
from django.contrib.auth.models import User
import jsonfield
import json

class ImageModel(models.Model):
    image = models.ImageField(upload_to='img', null=True)
    name = models.TextField('name')

class UserProfile(models.Model):
    user = models.OneToOneField(User, unique=True, on_delete=models.CASCADE)
    #user_images = models.ManyToManyField(ImageModel)
    pic = models.ImageField(null=True, upload_to='media')
    
    def save(self, *args, **kwargs):
        create = kwargs.pop('create', None)
        print('create:', create)
        if create == True:
            super(UserProfile, self).save(*args, **kwargs)
