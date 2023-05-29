from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import User,Image


class SignUpForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=False, help_text='Optional.')
    last_name = forms.CharField(max_length=30, required=False, help_text='Optional.')
    email = forms.EmailField(max_length=254, help_text='Required. Inform a valid email address.')

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2', )

class LoginForm(AuthenticationForm):
    class Meta:
        model = User
        fields = ('username', 'password',)
        


class ImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = '__all__'
        labels={'photo':''}
        
class DiseaseForm(forms.Form):
    disease_name = forms.CharField(label='Enter the name of a disease')

class PlantForm(forms.Form):
    plant_name = forms.CharField(label='Enter the name of a plant')
