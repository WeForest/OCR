from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response

from client_certification.models import ImageUpload
from client_certification.serializers import ImageUploadSerializer
from rest_framework import viewsets

'''
class UserView(APIView):
    def get(self, request):
        return Response("ok", status=200)  # 테스트용 Response
'''
class ImageUploadViewSet(viewsets.ModelViewSet):
    queryset = ImageUpload.objects.all()
    serializer_class = ImageUploadSerializer