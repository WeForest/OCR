from rest_framework import serializers
from client_certification.models import ImageUpload

class ImageUploadSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = ImageUpload
        fields = ('url', 'pk', 'title', 'imagefile')