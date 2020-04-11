from django.urls import path, include
from . import views

app_name = "camerafeed"

urlpatterns = [
    path('',views.home, name="home"),  # Home page url
    path('update/',views.update, name="update"),  # Ajax request url
]