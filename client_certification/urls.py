from django.urls import path
from . import views
from rest_framework import routers
from django.urls import include
from django.conf.urls import url


app_name = 'api_user'
urlpatterns = [
    path('', views.UserView),  # User에 관한 API를 처리하는 view로 Request를 넘김
]
