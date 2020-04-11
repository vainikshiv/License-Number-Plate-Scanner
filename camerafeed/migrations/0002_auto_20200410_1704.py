# Generated by Django 3.0.5 on 2020-04-10 11:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('camerafeed', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='data',
            name='vehicle_type',
            field=models.CharField(default='Car', max_length=10),
        ),
        migrations.AlterField(
            model_name='data',
            name='license_number',
            field=models.CharField(max_length=10, unique=True),
        ),
    ]