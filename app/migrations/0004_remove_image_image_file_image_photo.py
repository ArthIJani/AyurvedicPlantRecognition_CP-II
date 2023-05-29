# Generated by Django 4.0.8 on 2023-03-26 08:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0003_image'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='image',
            name='image_file',
        ),
        migrations.AddField(
            model_name='image',
            name='photo',
            field=models.ImageField(default='default.jpg', upload_to='myimage'),
        ),
    ]