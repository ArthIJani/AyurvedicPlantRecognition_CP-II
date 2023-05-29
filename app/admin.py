from django.contrib import admin


# Register your models here.
from .models import ContactMessage,Image,Data


admin.site.register(Data)


@admin.register(ContactMessage)
class ContactMessageAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'created_at')


@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = ['id','photo','date']
