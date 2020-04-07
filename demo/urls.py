from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),     # Admin panel url
    path('',include('camerafeed.urls'))  # App url
]
