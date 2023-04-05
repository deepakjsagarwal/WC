#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sklearn import svm


# In[8]:


from sklearn.model_selection import GridSearchCV


# In[9]:


param_grid={'C':[1000, 10000],'gamma':[0.01, 0.001],'kernel':['rbf']}


# In[10]:


svc=svm.SVC(probability=True)


# In[11]:


model=GridSearchCV(svc,param_grid, refit = True, verbose = 3)


# In[12]:


import os, timeit
import numpy as np


# In[15]:


from sys import platform
DATA_DIRECTORY = ""
SLASH = ""
if platform == "linux" or platform == "linux2":
    DATA_DIRECTORY = "/home/ishu_g.iitr/wheat/data/BULK/"
    SLASH = "/"
elif platform == "win32":
    DATA_DIRECTORY = "D:\mvl\wheat\data\BULK\\"
    SLASH="\\"


# In[16]:


#Constants
BAND_NUMBER = 60
FILLED_AREA_RATIO = 0.9
TOTAL_IMAGE_COUNT = 2400
IMAGE_COUNT = int(TOTAL_IMAGE_COUNT/4)
NUM_VARIETIES = 4

IMAGE_WIDTH = 30
IMAGE_HEIGHT = 30

TRAIN_IMAGES = 1200
TEST_IMAGES = 300


# In[17]:


from enum import Enum

class filter_method(Enum):
    none = 0
    snv = 1
    msc = 2
    savgol = 3

FILT = 1
FILTER = filter_method(FILT).name

# to be set if filter chosen is savgol
WINDOW = 7
ORDER = 2
DERIVATIVE = "none"


# In[18]:


from enum import Enum
 
class feature_extraction_method(Enum):
    none = 0
    pca_loading = 1
    lda = 2
    ipca = 3

FEAT_EXT = 0
FEATURE_EXTRACTION = feature_extraction_method(FEAT_EXT).name

NUM_OF_BANDS = 3
if FEATURE_EXTRACTION == "pca_loading" or FEATURE_EXTRACTION == "ipca":
    NUM_OF_BANDS = 8
elif FEATURE_EXTRACTION == "lda":
    NUM_OF_BANDS = 3
    assert NUM_OF_BANDS <= min(NUM_VARIETIES-1,168),"NUM_OF_BANDS is greater."


REMOVE_NOISY_BANDS = False
FIRST_BAND = 15
LAST_BAND = 161


# In[19]:


def start_timer():
    print("Testing started")
    return timeit.default_timer()

def end_timer():
    return timeit.default_timer()

def show_time(tic,toc): 
    test_time = toc - tic
    print('Testing time (s) = ' + str(test_time) + '\n')


# In[20]:


# List for All varieties
VARIETIES = []
VARIETIES_CODE = {}

for name in os.listdir(DATA_DIRECTORY):
    if (name.endswith(".hdr") or name.endswith(".bil")):
        continue
    VARIETIES_CODE[name] = len(VARIETIES)
    VARIETIES.append(name)
    if len(VARIETIES)==NUM_VARIETIES:
        break


# In[21]:


VARIETIES = ['DBW 187', 'DBW222', 'HD 3086', 'PBW 291']


# In[22]:


def dataset_file_name(variety):
    name = "./dataset/"+str(variety).zfill(3)+"_IC_"+str(TOTAL_IMAGE_COUNT).zfill(5)+"_FilledArea_"+str(FILLED_AREA_RATIO)+"_BandNo_"+str(BAND_NUMBER)+"_ImageHeight_"+str(IMAGE_HEIGHT)+"_ImageWidth_"+str(IMAGE_WIDTH)
    if FILT != 0:
        name+="_FILTER_"+str(FILTER)
    if FEAT_EXT !=0:
        name+="_FeatureExtraction_"+str(FEATURE_EXTRACTION)+"_NumOfBands_"+str(NUM_OF_BANDS)
    if REMOVE_NOISY_BANDS:
        name+="_REMOVE_NOISY_BANDS_"+str(REMOVE_NOISY_BANDS)+"_FB_"+str(FIRST_BAND)+"_LB_"+str(LAST_BAND)
    if FILTER == "savgol":
        name+="_WINDOW_"+str(WINDOW)+"_ORDER_"+str(ORDER)
    return name


# In[31]:


train_dataset = np.empty([0, IMAGE_HEIGHT, IMAGE_WIDTH, 168])
train_dataset_label = np.empty([0,], dtype=int)
test_dataset= np.empty([0, IMAGE_HEIGHT, IMAGE_WIDTH, 168])
test_dataset_label = np.empty([0,], dtype=int)

for idx, v in enumerate(VARIETIES):
    print("idx: ",idx)
    if idx >= NUM_VARIETIES:
        break
    train_dataset= np.concatenate((train_dataset, np.load(dataset_file_name(v)+"_train_dataset.npy")[:TRAIN_IMAGES]), axis =0)
    train_dataset_label = np.concatenate((train_dataset_label, np.load(dataset_file_name(v)+"_train_dataset_label.npy")[:TRAIN_IMAGES]), axis =0)
    test_dataset = np.concatenate((test_dataset, np.load(dataset_file_name(v)+"_test_dataset.npy")[:TEST_IMAGES]), axis =0)
    test_dataset_label = np.concatenate((test_dataset_label, np.load(dataset_file_name(v)+"_test_dataset_label.npy")[:TEST_IMAGES]), axis =0)


# In[93]:


x_train = train_dataset.reshape(train_dataset.shape[0], -1)
y_train = train_dataset_label.reshape(train_dataset_label.shape[0])


# In[94]:


model.fit(x_train, y_train)


# In[101]:


x_test = test_dataset.reshape(test_dataset.shape[0], -1)
y_test = test_dataset_label.reshape(test_dataset_label.shape[0])


# In[102]:


y_pred = model.predict(x_test)


# In[103]:


from sklearn.metrics import classification_report


# In[105]:


print(classification_report(y_test, y_pred))


# In[ ]:




