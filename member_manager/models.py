from __future__ import unicode_literals

from django.db import models  
from django.core.mail import send_mail  
from django.contrib.auth.models import PermissionsMixin  
from django.contrib.auth.base_user import AbstractBaseUser  
from django.utils.translation import ugettext_lazy as _

from .managers import UserManager


class User(AbstractBaseUser, PermissionsMixin):  
    email = models.EmailField(_('email address'), unique=True)
    password = models.CharField(max_length=28)
    date_joined = models.DateTimeField(_('date joined'), auto_now_add=True)
    is_active = models.BooleanField(_('active'), default=True)
    is_superuser = models.BooleanField(default=False)

    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    def is_staff(self):
        return self.is_superuser

    def email_user(self, subject, message, from_email=None, **kwargs):
        '''
        Sends an email to this User.
        '''
        send_mail(subject, message, from_email, [self.email], **kwargs)