from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    # add additional fields here, such as bio, profile picture, etc.


class Image(models.Model):
    photo = models.ImageField(upload_to="myimage", default='default.jpg')
    date = models.DateTimeField(auto_now_add=True)
     
class ContactMessage(models.Model):
    name = models.CharField(max_length=255)
    email = models.EmailField()
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.name} ({self.email})'

# class Plant(models.Model):
#     name = models.CharField(max_length=255)
#     diseases = models.CharField(max_length=255)

class Data(models.Model):
    Plant_Name = models.CharField(("Plant_Name"),max_length=255)
    Features = models.CharField(("Features"),max_length=255)
    Species = models.CharField(("Species"),max_length=255)  
    Description = models.CharField(("Description"),max_length=255)
    Diseases = models.CharField(("Diseases"),max_length=255)
    id = models.IntegerField(primary_key=True)
    
