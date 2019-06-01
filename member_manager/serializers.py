from rest_framework import serializers
from .models import UserProfile

class ProfilePictureSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = ['user','pic']
        read_only_fields = ('user',)