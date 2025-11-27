from django.urls import path
from . import views

app_name = "detector"

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload, name='upload'),
    path('process/', views.process_image_upload, name='process'),
    path('live/', views.live_page, name='live'),
    path('live_feed/', views.live_feed, name='live_feed'),
]
