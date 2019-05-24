from django.contrib.auth.base_user import BaseUserManager
from django.contrib.auth import get_user_model
import logging
logger = logging.getLogger(__name__)

class UserManager(BaseUserManager):
    use_in_migrations = True
    def __init__(self):
        super().__init__()

    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('You must input email')
        
        User = get_user_model()
        if not (User.objects.filter(email=email).exists()):
            user = User.objects.create(email=email)
            user.set_password(password)
            user.save()
        else:
            return None
        return user

    def create_superuser(self, email, password, **extra_fields):
        user = self.create_user(
            email = email,
            password = password,
        )
        user.is_superuser = True
        user.save(using=self._db)
        return user