from django.shortcuts import render, redirect
#from django.http import HttpResponse
from django.contrib.auth import authenticate, login
from .forms import LoginForm,SignUpForm
from django.shortcuts import render
from django.core.mail import send_mail
from django.conf import settings
from .models import Image
from .forms import ImageForm
import pandas as pd
import numpy as np
import os 
from django.http import HttpResponse, FileResponse
from django.core.files.storage import default_storage
import tensorflow as tf


from tensorflow.keras.models import Model, load_model, model_from_json
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import *
import os
from time import time
from datetime import datetime

import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model, model_from_json
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.preprocessing import image as IMAGE
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras import backend as K



#Import the required libaries
import matplotlib.pyplot as plt
#from PIL import Image
import os
import gc
import pickle
import numpy as np
from tqdm.notebook import tqdm
from skimage import io
# from matplotlib import cm
# from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from pylab import rcParams
# rcParams['figure.figsize'] = 20, 10

import pandas as pd
import math
import random

#from PIL import Image
import glob
#--------------------                           Main Page                      -----------------------------------
def index(request):
    fixed_footer=True
    pred = 'Not Found'
    def capitalize_first_character(input_str): #function to capitalize the first word
            if input_str!=None:
                words = input_str.split()
                capitalized_words = []
                for word in words: 
                    capitalized_word = word[0].upper() + word[1:].lower()
                    capitalized_words.append(capitalized_word)
                capitalized_str = ' '.join(capitalized_words)
                return capitalized_str
            else:
                return input_str
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
        form = ImageForm()
        img1 =  Image.objects.order_by('date').last()
        img = default_storage.path(img1.photo.name)
        #img = os.path.normpath(img)
        #img = 'static\dataset_3\Avaram\img.jpg'

        ###To Predict##########
        # Load the trained model
        model = tf.keras.models.load_model('static\models\plant_recognition_model.h5')

        # Load an image to make a prediction on
         # Replace with your own image path
        from tensorflow.keras.preprocessing import image as IMAGE
        img = IMAGE.load_img(img, target_size=(224, 224))
        x = IMAGE.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.

        # Make a prediction on the image using the loaded model
        preds = model.predict(x)
        class_idx = np.argmax(preds[0])
        class_indices = pickle.load(open('static\models\class_indices','rb'))
        class_name = class_indices
        for k, v in class_name.items():
            if v == class_idx:
                pred = k
        message = None
        hindi_names=None
        english_hindi_name=None
        english_names=None
        common_names=None
        scientific_names=None
        features=None
        discription=None
        diseases_can_cure=None
        img_path1 = None
        location = None
        plant = pred
        plant_lower = plant.lower()
        dt_info = pd.read_csv('plant_info.csv')
        dt_name = pd.read_csv('Common Name.csv')

        #storing the different different columns which contains the various names of the plant
        data_name = list(dt_info.iloc[:,0].values)
        scientific_name = list(dt_name.iloc[:,-1].values)
        scientific_name_lower = [i.lower() for i in scientific_name]

        hindi_name1 = list(dt_name.iloc[:,1].values)
        hindi_name2 = list(dt_name.iloc[:,3].values)
        hindi_name3 = list(dt_name.iloc[:,4].values)
        
        hindi_name1_lower = [i.lower() for i in hindi_name1 ]
        hindi_name2_lower = [i.lower() for i in hindi_name2 ]
        hindi_name3_lower = [i.lower() for i in hindi_name3 ]


        english_name = list(dt_name.iloc[:,2].values)
        english_name_lower = [i.lower() for i in english_name]
        common_name = list(dt_name.iloc[:,0].values)
        common_name_lower = [i.lower() for i in common_name]

        #Getting the index of the input_data 
        location =None
        if(plant_lower in hindi_name1_lower):
            location = hindi_name1_lower.index(plant_lower)
        elif(plant_lower in hindi_name2_lower):
            location = hindi_name2_lower.index(plant_lower)
        elif(plant_lower in hindi_name3_lower):
            location = hindi_name3_lower.index(plant_lower)
        elif(plant_lower in english_name_lower):
            location = english_name_lower.index(plant_lower)
        elif(plant_lower in common_name_lower):
            location = common_name_lower.index(plant_lower)
        elif(plant_lower in scientific_name_lower):
            location = scientific_name_lower.index(plant_lower)
        elif(plant in hindi_name1):
            location = hindi_name1.index(plant)
        elif(plant in hindi_name2):
            location = hindi_name2.index(plant)
        elif(plant in hindi_name3):
            location = hindi_name3.index(plant)
        elif(plant in english_name):
            location = english_name.index(plant)
        elif(plant in common_name):
            location = common_name.index(plant)
        elif(plant_lower in scientific_name):
            location = scientific_name.index(plant_lower)
        
        #Extracting the information from the various info of plant by using its location
        if(location != None):
            hindi_names = hindi_name1[location]
            english_hindi_name = hindi_name3[location]
            english_names = english_name[location]
            common_names = common_name[location]
            scientific_names1 = scientific_name[location]
            scientific_names = capitalize_first_character(scientific_names1)
            features = str(dt_info.iloc[location,1])
            discription = str(dt_info.iloc[location,3])
            diseases_can_cure = str(dt_info.iloc[location,4])
            message=None

            data_names = str(data_name[location])
            folder_path = 'static\dataset_3' 
            for subdir in os.listdir(folder_path):
                if(subdir == data_names):
                    img_path1 = os.path.join(folder_path,subdir)

        else:
            message = "We don't have information of the plant mentioned by you."
            hindi_names=None
            english_hindi_name=None
            english_names=None
            common_names=None
            scientific_names=None
            features=None
            discription=None
            diseases_can_cure=None
            img_path1 = None
            location = None
        fixed_footer=False
        return render(request,'plant_prediction.html',{'l':location,'pred':pred,'hindi_names':hindi_names,'english_hindi_name':english_hindi_name,'message':message,'english_names':english_names,
                                                    'common_names':common_names, 'scientific_names':scientific_names,'features':features,'discription':discription
                                                    ,'diseases_can_cure':diseases_can_cure,'img':img,'img_path1':img_path1,'fixed_footer':fixed_footer})

        
        #return render(request, 'plant_prediction.html' , {'img':img,'pred':pred,'form':form,'fixed_footer':fixed_footer} )
    else:
        return render(request,'index.html', {'fixed_footer':fixed_footer})
   

def explore(request):
    fixed_footer=True
    return render(request,'explore.html',{'fixed_footer':fixed_footer})

def plant_info(request): #to showing the information of the plant entered by the user
    hindi_names=None
    english_hindi_name=None
    english_names=None
    common_names=None
    scientific_names=None
    features=None
    discription=None
    diseases_can_cure=None
    message=None
    img_path1=None
    from_suggetion = None
    location = None
    def capitalize_first_character(input_str): #function to capitalize the first word
            if input_str!=None:
                words = input_str.split()
                capitalized_words = []
                for word in words: 
                    capitalized_word = word[0].upper() + word[1:].lower()
                    capitalized_words.append(capitalized_word)
                capitalized_str = ' '.join(capitalized_words)
                return capitalized_str
            else:
                return input_str
        

    if request.method == 'POST':
        input_data = request.POST['input']  #getting input from the user
        input_data_lower = input_data.lower() #converting the input into lowercase and storing it

        #Loading required datasets
        dt_info = pd.read_csv('plant_info.csv')
        dt_name = pd.read_csv('Common Name.csv')

        #storing the different different columns which contains the various names of the plant
        data_name = list(dt_info.iloc[:,0].values)
        scientific_name = list(dt_name.iloc[:,-1].values)
        scientific_name_lower = [i.lower() for i in scientific_name]

        hindi_name1 = list(dt_name.iloc[:,1].values)
        hindi_name2 = list(dt_name.iloc[:,3].values)
        hindi_name3 = list(dt_name.iloc[:,4].values)

        hindi_name1_lower = [i.lower() for i in hindi_name1 ]
        hindi_name2_lower = [i.lower() for i in hindi_name2 ]
        hindi_name3_lower = [i.lower() for i in hindi_name3 ]


        english_name = list(dt_name.iloc[:,2].values)
        english_name_lower = [i.lower() for i in english_name]
        common_name = list(dt_name.iloc[:,0].values)
        common_name_lower = [i.lower() for i in common_name]

        #Getting the index of the input_data 
        location =None
        if(input_data_lower in hindi_name1_lower):
            location = hindi_name1_lower.index(input_data_lower)
        elif(input_data_lower in hindi_name2_lower):
            location = hindi_name2_lower.index(input_data_lower)
        elif(input_data_lower in hindi_name3_lower):
            location = hindi_name3_lower.index(input_data_lower)
        elif(input_data_lower in english_name_lower):
            location = english_name_lower.index(input_data_lower)
        elif(input_data_lower in common_name_lower):
            location = common_name_lower.index(input_data_lower)
        elif(input_data_lower in scientific_name_lower):
            location = scientific_name_lower.index(input_data_lower)
        elif(input_data in hindi_name1):
            location = hindi_name1.index(input_data)
        elif(input_data in hindi_name2):
            location = hindi_name2.index(input_data)
        elif(input_data in hindi_name3):
            location = hindi_name3.index(input_data)
        elif(input_data in english_name):
            location = english_name.index(input_data)
        elif(input_data in common_name):
            location = common_name.index(input_data)
        elif(input_data_lower in scientific_name):
            location = scientific_name.index(input_data_lower)
        
        #Extracting the information from the various info of plant by using its location
        if(location != None):
            hindi_names = hindi_name1[location]
            english_hindi_name = hindi_name3[location]
            english_names = english_name[location]
            common_names = common_name[location]
            scientific_names1 = scientific_name[location]
            scientific_names = capitalize_first_character(scientific_names1)
            features = str(dt_info.iloc[location,1])
            discription = str(dt_info.iloc[location,3])
            diseases_can_cure = str(dt_info.iloc[location,4])
            message=None

            data_names = str(data_name[location])
            folder_path = 'static\dataset_3' 
            for subdir in os.listdir(folder_path):
                if(subdir == data_names):
                    img_path1 = os.path.join(folder_path,subdir)

        else:
            message = "We don't have information of the plant mentioned by you."
            hindi_names=None
            english_hindi_name=None
            english_names=None
            common_names=None
            scientific_names=None
            features=None
            discription=None
            diseases_can_cure=None
            img_path1 = None
            location = None
    fixed_footer=False
    return render(request,'plant_info.html',{'l':location,'hindi_names':hindi_names,'english_hindi_name':english_hindi_name,'message':message,'english_names':english_names,
                                                'common_names':common_names, 'scientific_names':scientific_names,'features':features,'discription':discription
                                                ,'diseases_can_cure':diseases_can_cure,'img_path1':img_path1,'fixed_footer':fixed_footer})
    
        

def suggested_plant_info(request):  #to showing the information of the plant that is clicked by an user
    hindi_names=None
    english_hindi_name=None
    english_names=None
    common_names=None
    scientific_names=None
    features=None
    discription=None
    diseases_can_cure=None
    message=None
    img_path1=None
    from_suggetion = None
    image_path = request.GET.get('data')
    if request.method == 'GET':

        
        def capitalize_first_character(input_str):
                if input_str!=None:
                    words = input_str.split()
                    capitalized_words = []
                    for word in words: 
                        capitalized_word = word[0].upper() + word[1:].lower()
                        capitalized_words.append(capitalized_word)
                    capitalized_str = ' '.join(capitalized_words)
                    return capitalized_str
                else:
                    return input_str

    
        input_path = image_path
        count = 0
        input_data = ''
        for i in input_path:
            if count==2:
                input_data = input_data + i
            elif i == "\\" :
                count=count+1
    
        input_data_lower = input_data.lower()
        dt_info = pd.read_csv('plant_info.csv')
        dt_name = pd.read_csv('Common Name.csv')


        data_name = list(dt_info.iloc[:,0].values)
        scientific_name = list(dt_name.iloc[:,-1].values)
        scientific_name = [i.lower() for i in scientific_name]

        hindi_name1 = list(dt_name.iloc[:,1].values)
        hindi_name2 = list(dt_name.iloc[:,3].values)
        hindi_name3 = list(dt_name.iloc[:,4].values)

        hindi_name1_lower = [i.lower() for i in hindi_name1 ]
        hindi_name2_lower = [i.lower() for i in hindi_name2 ]
        hindi_name3_lower = [i.lower() for i in hindi_name3 ]
        english_name = list(dt_name.iloc[:,2].values)
        english_name_lower = [i.lower() for i in english_name]
        common_name = list(dt_name.iloc[:,0].values)
        common_name_lower = [i.lower() for i in common_name]
        location =None
        if(input_data_lower in hindi_name1_lower):
            location = hindi_name1_lower.index(input_data_lower)
        elif(input_data_lower in hindi_name2_lower):
            location = hindi_name2_lower.index(input_data_lower)
        elif(input_data_lower in hindi_name3_lower):
            location = hindi_name3_lower.index(input_data_lower)
        elif(input_data_lower in english_name_lower):
            location = english_name_lower.index(input_data_lower)
        elif(input_data_lower in common_name_lower):
            location = common_name_lower.index(input_data_lower)
        elif(input_data in hindi_name1):
            location = hindi_name1.index(input_data)
        elif(input_data in hindi_name2):
            location = hindi_name2.index(input_data)
        elif(input_data in hindi_name3):
            location = hindi_name3.index(input_data)
        elif(input_data in english_name):
            location = english_name.index(input_data)
        elif(input_data in common_name):
            location = common_name.index(input_data)

        if(location != None):
            hindi_names = hindi_name1[location]
            english_hindi_name = hindi_name3[location]
            english_names = english_name[location]
            common_names = common_name[location]
            scientific_names1 = scientific_name[location]
            scientific_names = capitalize_first_character(scientific_names1)
            features = str(dt_info.iloc[location,1])
            discription = str(dt_info.iloc[location,3])
            diseases_can_cure = str(dt_info.iloc[location,4])
            message=None

            data_names = str(data_name[location])
            folder_path = 'static\dataset_3' 
            for subdir in os.listdir(folder_path):
                if(subdir == data_names):
                    img_path1 = os.path.join(folder_path,subdir)

        else:
            message = 'we dont have information of the plant mentioned by you.'
            hindi_names=None
            english_hindi_name=None
            english_names=None
            common_names=None
            scientific_names=None
            features=None
            discription=None
            diseases_can_cure=None
            img_path1 = None
            
        fixed_footer=False

    return render(request,'suggested_plant_info.html',{'path':image_path,'input_data':input_data,'hindi_names':hindi_names,'english_hindi_name':english_hindi_name,'message':message,'english_names':english_names,
                                            'common_names':common_names, 'scientific_names':scientific_names,'features':features,'discription':discription
                                        ,'diseases_can_cure':diseases_can_cure,'img_path1':img_path1,'fixed_footer':fixed_footer})
    

def plant_suggestion(request):  #to show the output of various 
    output_data = ''
    fixed_footer=False

    if request.method == 'POST':
        input_data = request.POST['input'].lower()
        # importing the datasets
        dt = pd.read_csv('plant_suggestion_for_diseases.csv')
        x = list(dt.iloc[:,0])
        common_name = pd.read_csv("Common Name.csv")
        ranges = 0
        if input_data in x:
            location = x.index(input_data)      #getting locatioon of entered disease in dataset
            data = str(dt.iloc[location,-1]).lower()    #getting data from that perticular index.
            scientific_name = list(common_name.iloc[:,0].values)    #storing main name from the 2nd dataset
            scientific_name = [item.lower() for item in scientific_name]
            english_common_name1 = list(common_name.iloc[:,2])  #storing english common names from the 2nd dataset
            english_common_name1 = [item.lower() for item in english_common_name1]
            hindi_common_name1 = list(common_name.iloc[:,1])    #storing hindi names from the 2nd dataset
            hindi_common_name1= [item.lower() for item in hindi_common_name1]
            hindi_common_name = []
            english_common_name = []
            temp = ''
            output_data1 = []
            output_data = []
            for i in data:
                if(i==','):     #seprating strings with comma
                    if(temp[-1]==' '):      #removing the space before coma
                        temp = temp[:-1]
                        output_data1.append(temp) #apeending the plant name in output_data1
                    temp = ''
                else:
                    temp = temp + i     #storing all strings until coma came
            
            if(temp[-1]==' '):      #removing last space
                temp = temp[:-1] 
                output_data1.append(temp)
            else:
                output_data1.append(temp)
            
            toremove=[]
            for i in output_data1:
                if i not in scientific_name:
                    toremove.append(i)
            for i in toremove:
                output_data1.remove(i)
            
            for i in output_data1:
                if i in scientific_name:
                    j = scientific_name.index(i)
                    hindi_common_name.append(hindi_common_name1[j])
                    english_common_name.append(english_common_name1[j])

            if(len(output_data1)>0):
                for i,j in enumerate(output_data1):
                    tempo=''
                    tempo = str(output_data1[i])+'  OR  '+str(english_common_name[i])+'  OR  '+str(hindi_common_name[i])
                    output_data.append(tempo)

            
                #images path
                img_path = []
                folder_path = 'static\dataset_3'
                for subdir in os.listdir(folder_path):
            # Get a list of all image files in the folder
                    subdir1 = subdir
                    subdir1 = subdir1.lower()
                    if subdir1 in output_data1:
                        subdir_path = os.path.join(folder_path, subdir)
                        img_path.append(subdir_path)
                output_data = zip(output_data,img_path)
                temp = 'found'
                return render(request, 'plant_suggestion.html',  {'output_data':output_data,'temp':temp,'output_data1':output_data1,'fixed_footer':fixed_footer })
            else:
                temp=''
                output_data = 'no data found'
                return render(request, 'plant_suggestion.html',{'temp':temp,'output_data':output_data})
        else:
            temp=''
            output_data = 'no data found'
            return render(request, 'plant_suggestion.html',{'temp':temp,'output_data':output_data})       
    else:
        temp='initial'
        output_data = 'Enter the correct name of the disease in English.'
        return render(request, 'plant_suggestion.html',{'temp':temp,'output_data':output_data})



def plant_comparision(request):
    hindi_names1=None
    english_hindi_name1=None
    english_names1=None
    common_names1=None
    scientific_names1=None
    features1=None
    discription1=None
    diseases_can_cure1=None
    img_path1=None
    real_scientific_name1 = None

    hindi_names2=None
    english_hindi_name2=None
    english_names2=None
    common_names2=None
    scientific_names2=None
    features2=None
    discription2=None
    diseases_can_cure2=None
    img_path2=None
    real_scientific_name2 = None
    if request.method =='POST':

        def capitalize_first_character(input_str):
            if input_str!=None:
                words = input_str.split()
                capitalized_words = []
                for word in words: 
                    capitalized_word = word[0].upper() + word[1:].lower()
                    capitalized_words.append(capitalized_word)
                capitalized_str = ' '.join(capitalized_words)
                return capitalized_str
            else:
                return input_str
        
        #storing inputs
        input_data1 = str(request.POST['input1'].lower())
        input_data2 = str(request.POST['input2'].lower())
        input_data1_lower = input_data1.lower()
        input_data2_lower = input_data2.lower()
        #importing required datasets
        dt_info = pd.read_csv('plant_info.csv')
        dt_name = pd.read_csv('Common Name.csv')


        #dividing the datafield in required format
        data_name = list(dt_info.iloc[:,0].values)
        scientific_name = list(dt_name.iloc[:,-1].values)
        scientific_name_lower = [i.lower() for i in scientific_name]

        hindi_name1 = list(dt_name.iloc[:,1].values)
        hindi_name2 = list(dt_name.iloc[:,3].values)
        hindi_name3 = list(dt_name.iloc[:,4].values)

        hindi_name1_lower = [i.lower() for i in hindi_name1 ]
        hindi_name2_lower = [i.lower() for i in hindi_name2 ]
        hindi_name3_lower = [i.lower() for i in hindi_name3 ]
        
        
        english_name = list(dt_name.iloc[:,2].values)
        english_name_lower = [i.lower() for i in english_name]

        common_name = list(dt_name.iloc[:,0].values)
        common_name_lower = [i.lower() for i in common_name]


        #getting required info for first input
        location1 = None
        if(input_data1 in hindi_name1):
            location1 = hindi_name1.index(input_data1)
        elif(input_data1 in hindi_name2):
            location1 = hindi_name2.index(input_data1)
        elif(input_data1 in hindi_name3):
            location1 = hindi_name3.index(input_data1)
        elif(input_data1 in english_name):
            location1 = english_name.index(input_data1)
        elif(input_data1 in common_name):
            location1 = common_name.index(input_data1)
        elif(input_data1 in scientific_name):
            location1 = scientific_name.index(input_data1)
        elif(input_data1_lower in hindi_name1_lower):
            location1 = hindi_name1_lower.index(input_data1_lower)
        elif(input_data1_lower in hindi_name2_lower):
            location1 = hindi_name2_lower.index(input_data1_lower)
        elif(input_data1_lower in hindi_name3_lower):
            location1 = hindi_name3_lower.index(input_data1_lower)
        elif(input_data1_lower in english_name_lower):
            location1 = english_name_lower.index(input_data1_lower)
        elif(input_data1_lower in common_name_lower):
            location1 = common_name_lower.index(input_data1_lower)
        elif(input_data1_lower in scientific_name_lower):
            location1 = scientific_name_lower.index(input_data1_lower)
        #getting required information for second input
        location2 = None
        if(input_data2 in hindi_name1):
            location2 = hindi_name1.index(input_data2)
        elif(input_data2 in hindi_name2):
            location2 = hindi_name2.index(input_data2)
        elif(input_data2 in hindi_name3):
            location2 = hindi_name3.index(input_data2)
        elif(input_data2 in english_name):
            location2 = english_name.index(input_data2)
        elif(input_data2 in common_name):
            location2 = common_name.index(input_data2)
        elif(input_data2 in scientific_name):
            location2 = scientific_name.index(input_data2)
        elif(input_data2_lower in hindi_name1_lower):
            location2 = hindi_name1_lower.index(input_data2_lower)
        elif(input_data2_lower in hindi_name2_lower):
            location2 = hindi_name2_lower.index(input_data2_lower)
        elif(input_data2_lower in hindi_name3_lower):
            location2 = hindi_name3_lower.index(input_data2_lower)
        elif(input_data2_lower in english_name_lower):
            location2 = english_name_lower.index(input_data2_lower)
        elif(input_data2_lower in common_name_lower):
            location2 = common_name_lower.index(input_data2_lower)
        elif(input_data2_lower in scientific_name_lower):
            location2 = scientific_name_lower.index(input_data2_lower)

        if(location1==None)and(location2!=None):
            message1 = 'We dont have the information of '+input_data1
            message1 = message1 + ' but we have the information of '+input_data2
            hindi_names2 = hindi_name1[location2]
            english_hindi_name2 = hindi_name3[location2]
            english_names2 = english_name[location2]
            common_names2 = common_name[location2]
            scientific_name2 = scientific_name[location2]
            scientific_names2 = capitalize_first_character(scientific_name2)
            features2 = str(dt_info.iloc[location2,1])
            discription2 = str(dt_info.iloc[location2,3])
            diseases_can_cure2 = str(dt_info.iloc[location2,4])
            
            data_names = str(data_name[location2])
            folder_path = 'static\dataset_3' 
            for subdir in os.listdir(folder_path):
                if(subdir == data_names):
                    img_path2 = os.path.join(folder_path,subdir)
                
            fixed_footer=False

            return render(request,'plant_comparision.html',{'message1':message1,'input_data2':input_data2,
                                                            'hindi_names2':hindi_names2,'english_hindi_name2':english_hindi_name2,'english_names2':english_names2,
                                                            'common_names2':common_names2, 'scientific_names2':scientific_names2,'features2':features2,'discription2':discription2
                                                            ,'diseases_can_cure2':diseases_can_cure2,'img_path2':img_path2,'fixed_footer':fixed_footer})
        elif(location1!=None) and (location2==None):
            message2 = 'We dont have the information of '+input_data2
            message2 = message2 + ' but we have the information of '+input_data1
            hindi_names1 = hindi_name1[location1]
            english_hindi_name1 = hindi_name3[location1]
            english_names1 = english_name[location1]
            common_names1 = common_name[location1]
            scientific_name1 = scientific_name[location1]
            scientific_names1 = capitalize_first_character(scientific_name1)
            features1 = str(dt_info.iloc[location1,1])
            discription1 = str(dt_info.iloc[location1,3])
            diseases_can_cure1 = str(dt_info.iloc[location1,4])
            
            data_names = str(data_name[location1])
            folder_path = 'static\dataset_3' 
            for subdir in os.listdir(folder_path):
                if(subdir == data_names):
                    img_path1 = os.path.join(folder_path,subdir)
                
            fixed_footer=False
            return render(request,'plant_comparision.html',{'message2':message2,'input_data1':input_data1,
                                                            'hindi_names1':hindi_names1,'english_hindi_name1':english_hindi_name1,'english_names1':english_names1,
                                                            'common_names1':common_names1, 'scientific_names1':scientific_names1,'features1':features1,'discription1':discription1
                                                            ,'diseases_can_cure1':diseases_can_cure1,'img_path1':img_path1,'fixed_footer':fixed_footer})
        elif(location1==None)and(location2==None):
            message12 = 'We dont have the information of '+input_data1
            message12 = message12 + ' and '+input_data2

            fixed_footer=False

            return render(request,'plant_comparision.html',{'message12':message12,'fixed_footer':fixed_footer})
        else:
            message0 = ''
            #for first plant
            hindi_names1 = hindi_name1[location1]
            english_hindi_name1 = hindi_name3[location1]
            english_names1 = english_name[location1]
            common_names1 = common_name[location1]
            scientific_name1 = scientific_name[location1]
            scientific_names1 = capitalize_first_character(scientific_name1)
            features1 = str(dt_info.iloc[location1,1])
            discription1 = str(dt_info.iloc[location1,3])
            diseases_can_cure1 = str(dt_info.iloc[location1,4])
            
            data_names = str(data_name[location1])
            folder_path = 'static\dataset_3' 
            for subdir in os.listdir(folder_path):
                if(subdir == data_names):
                    img_path1 = os.path.join(folder_path,subdir)


            ##for second plant
            hindi_names2 = hindi_name1[location2]
            english_hindi_name2 = hindi_name3[location2]
            english_names2 = english_name[location2]
            common_names2 = common_name[location2]
            scientific_name2 = scientific_name[location2]
            scientific_names2 = capitalize_first_character(scientific_name2)
            features2 = str(dt_info.iloc[location2,1])
            discription2 = str(dt_info.iloc[location2,3])
            diseases_can_cure2 = str(dt_info.iloc[location2,4])
            message0 = True
            data_names = str(data_name[location2])
            folder_path = 'static\dataset_3' 
            for subdir in os.listdir(folder_path):
                if(subdir == data_names):
                    img_path2 = os.path.join(folder_path,subdir)
            fixed_footer=False

            return render(request,'plant_comparision.html',{'message0':message0,
                                                            'input_data1':input_data1,
                                                            'hindi_names1':hindi_names1,'english_hindi_name1':english_hindi_name1,'english_names1':english_names1,
                                                            'common_names1':common_names1, 'scientific_names1':scientific_names1,'features1':features1,'discription1':discription1
                                                            ,'diseases_can_cure1':diseases_can_cure1,'img_path1':img_path1,
                                                            'input_data2':input_data2,
                                                            'hindi_names2':hindi_names2,'english_hindi_name2':english_hindi_name2,'english_names2':english_names2,
                                                            'common_names2':common_names2, 'scientific_names2':scientific_names2,'features2':features2,'discription2':discription2
                                                            ,'diseases_can_cure2':diseases_can_cure2,'img_path2':img_path2,'fixed_footer':fixed_footer})
    else:
        fixed_footer=False

        return render(request,'plant_comparision.html',{'fixed_footer':fixed_footer})


from .models import ContactMessage

def contact_us(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        message = request.POST.get('message')

        # Save to database
        contact_message = ContactMessage(name=name, email=email, message=message)
        contact_message.save()
        fixed_footer=False

        return render(request, 'contact_us.html', {'form': ContactMessage(), 'success': True,'fixed_footer':fixed_footer})
    
    fixed_footer=False

    return render(request, 'contact_us.html',{'fixed_footer':fixed_footer})





#--------------------                           SignUP                     -----------------------------------
def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            fixed_footer=False

            return redirect('index',{'fixed_footer':fixed_footer})
    else:
        fixed_footer=False
        form = SignUpForm()
    fixed_footer=False
    return render(request, 'signup.html', {'form': form,'fixed_footer':fixed_footer})

#--------------------                           Login Page                    -----------------------------------
def login_view(request):
    if request.method == 'POST':
        form = LoginForm(data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request,username=username, password=password)
            if user is not None:
                login(request, user)
                fixed_footer=False
                return redirect('index',{'fixed_footer':fixed_footer})
    else:
        form = LoginForm()
        fixed_footer=False
    return render(request, 'login.html', {'form': form,'fixed_footer':fixed_footer})