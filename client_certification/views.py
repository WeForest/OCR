from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response

from client_certification.models import ImageUpload
from client_certification.serializers import ImageUploadSerializer
from rest_framework import viewsets

@api_view(['POST'])
def UserView(request):
    return Response("ok", status=200)  # 테스트용 Response
