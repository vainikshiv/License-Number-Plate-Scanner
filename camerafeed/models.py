from django.db import models

# Create your models here.
# DB table to store value of number plate
class data(models.Model):
    license_number = models.CharField(max_length=10)
    image = models.ImageField()
    date_created = models.DateTimeField(auto_now_add=True)
    vehicle_type = models.CharField(max_length=10, default="Car")

    def __str__(self):
        return self.license_number