from django.db import models

# Create your models here.
class data(models.Model):
    license_number = models.CharField(max_length=10)
    image = models.ImageField()
    date_created = models.DateTimeField(auto_now_add=True)