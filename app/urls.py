from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('prediction', views.index, name='index'),
    path('plant_suggestion', views.plant_suggestion, name='plant_suggestion'),
    path('suggested_plant_info',views.suggested_plant_info,name='suggested_plant_info'),
    path('plant_info',views.plant_info,name='plant_info'),

    path('explore',views.explore,name='explore'),
    path('plant_comparision',views.plant_comparision,name='plant_comparision'),
    #path('explore', views.explore, name='explore'),
    #path('blog_list', views.blog_list, name='blog_list'),
    #path('<int:pk>', views.blog_detail, name='blog_detail'),
    #path('create_blog_post', views.create_blog_post, name='create_blog_post'),
    path('contact_us', views.contact_us, name='contact_us'),
    path('signup', views.signup, name='signup'),
    path('login', views.login_view, name='login'),
]