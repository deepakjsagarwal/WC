#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


from tensorflow import keras as keras
from keras import layers as layers


# In[3]:


import os, timeit
from skimage.filters import threshold_otsu
import numpy as np
from math import inf as inf


# In[4]:


from spectral.io import envi as envi
from spectral import imshow


# In[5]:


from sklearn.decomposition import IncrementalPCA


# In[6]:


import sys


# In[61]:


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[62]:


from sys import platform
DATA_DIRECTORY = ""
SLASH = ""
if platform == "linux" or platform == "linux2":
    DATA_DIRECTORY = "/home/ishu_g.iitr/wheat/data/BULK/"
    SLASH = "/"
elif platform == "win32":
    DATA_DIRECTORY = "D:\mvl\wheat\data\BULK\\"
    SLASH="\\"


# In[63]:


#Constants
BAND_NUMBER = 60
FILLED_AREA_RATIO = 0.9
TOTAL_IMAGE_COUNT = 2400
IMAGE_COUNT = int(TOTAL_IMAGE_COUNT/4)
NUM_VARIETIES = 17

IMAGE_WIDTH = 30
IMAGE_HEIGHT = 30


# In[64]:


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


# In[65]:


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


# In[66]:


def start_timer():
    print("Testing started")
    return timeit.default_timer()

def end_timer():
    return timeit.default_timer()

def show_time(tic,toc): 
    test_time = toc - tic
    print('Testing time (s) = ' + str(test_time) + '\n')


# In[67]:


def exactPathHDR(variety,file):
    return DATA_DIRECTORY+variety+SLASH+file+".bil.hdr"

def exactPathBIL(variety,file):
    return DATA_DIRECTORY+variety+SLASH+file+".bil"


# In[68]:


def getROI(img, band_number):
    img_band = img.read_band(band_number)
    threshold = threshold_otsu(img_band)
    roi=[]
    for x in range(img_band.shape[0]):
        a=[]
        for y in range(img_band.shape[1]):
            if img_band[x][y]>threshold:
                a.append(1)
            else:
                a.append(0)
        roi.append(a)
    return roi


# In[69]:


#Returns range for x and y from where we have to crop images
def getRangeXandY(img,band_number):
    img_band = img.read_band(band_number)
    roi = getROI(img,band_number)
    xmin = inf
    xmax = 0
    ymin = inf
    ymax = 0
    for x in range(img_band.shape[0]):
        for y in range(img_band.shape[1]):
            if roi[x][y]==1:
                if x<xmin:
                    xmin=x
                if x>xmax:
                    xmax=x
                if y<ymin:
                    ymin=y
                if y>ymax:
                    ymax=y
    return xmin, xmax, ymin, ymax


# In[70]:


def getCroppedImage(img,band_number):
    xmin, xmax, ymin, ymax = getRangeXandY(img,band_number)
    new_img = img[xmin:xmax, ymin:ymax, :]
    return new_img


# In[71]:


def getCroppedROI(img,band_number):
    xmin, xmax, ymin, ymax = getRangeXandY(img,band_number)
    roi = np.array(getROI(img,band_number))
    roi = roi[xmin:xmax, ymin:ymax]
    return roi   


# In[72]:


def getUsefulImage(img,band_number):
    crop_img = getCroppedImage(img,band_number)
    crop_roi = getCroppedROI(img,band_number)
    for x in range(crop_img.shape[2]):
        band = crop_img[:,:,x]
        crop_img[:,:,x] = band*crop_roi
    return crop_img


# In[73]:


def preprocessHSI(img, band_number):
    img = getUsefulImage(img, band_number)
    return img


# In[74]:


data_augmentation = keras.Sequential([
    layers.RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    layers.RandomRotation(factor=(-0.1, 0.1)),
    layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1,0.1)),
    layers.RandomFlip(mode="horizontal_and_vertical", seed=None)
])

def getAugumentedImage(img):
    augmented_image = data_augmentation(img) 
    return augmented_image

def checkAugumentedImage(img):
    aug_band = img[:,:,0]
    filled_area_ratio = (np.count_nonzero(aug_band))/(aug_band.shape[0]*aug_band.shape[1])
    if filled_area_ratio > FILLED_AREA_RATIO :
        return True
    else:
        return False


# In[75]:


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


# In[76]:


#List for all file names in varities
FILES = []
MAX_FILE_NUM = 4
for x in range(1,MAX_FILE_NUM+1):
    FILES.append("B_"+str(x))


# In[77]:


def extractRawImages(v):
    #List of all images
    images = []
    for f in FILES:
        try:
            img = envi.open(exactPathHDR(v,f),exactPathBIL(v,f))
            img = preprocessHSI(img, BAND_NUMBER)
            images.append(img)
        except:
            pass
    return images


# In[78]:


from IPython.display import clear_output

def createDataset(images, label):
    train_dataset = []
    train_dataset_label = []
    test_dataset = []
    test_dataset_label = []
    tic = start_timer()
    for index, img in enumerate(images):
        count = 0
        while count<IMAGE_COUNT:
            aug_img = getAugumentedImage(img)
            if checkAugumentedImage(aug_img):
                if count%5 == 0:
                    test_dataset.append(aug_img)
                    test_dataset_label.append(label)
                else:
                    train_dataset.append(aug_img)
                    train_dataset_label.append(label)
                count+=1 

            clear_output(wait=True)
            print("Label: ",label," Index: ",index," Count: ",count)
    toc = end_timer()
    show_time(tic,toc)
    
    train_dataset = np.array(train_dataset)
    train_dataset_label = np.array([VARIETIES_CODE[label] for label in train_dataset_label])
    test_dataset = np.array(test_dataset)
    test_dataset_label = np.array([VARIETIES_CODE[label] for label in test_dataset_label])
    
    return train_dataset,train_dataset_label,test_dataset,test_dataset_label


# In[79]:


results_dir = os.path.join('./dataset')
    
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# In[80]:


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


# In[81]:


def save_dataset(variety, train_dataset,train_dataset_label,test_dataset,test_dataset_label):
    DATASET_FILE_NAME = dataset_file_name(variety)
    np.save(DATASET_FILE_NAME+"_train_dataset.npy",train_dataset)
    np.save(DATASET_FILE_NAME+"_train_dataset_label.npy",train_dataset_label)
    np.save(DATASET_FILE_NAME+"_test_dataset.npy",test_dataset)
    np.save(DATASET_FILE_NAME+"_test_dataset_label.npy",test_dataset_label)


# In[82]:


def remove_noisy_bands(remove_noisy_bands,train_dataset,test_dataset):
    if remove_noisy_bands:
        train_dataset = train_dataset[:,:,:,FIRST_BAND:LAST_BAND+1]
        test_dataset = test_dataset[:,:,:,FIRST_BAND:LAST_BAND+1]
    return train_dataset,test_dataset


# In[83]:


def snv(input_data):
    """
        :snv: A correction technique which is done on each
        individual spectrum, a reference spectrum is not
        required        :param input_data: Array of spectral data
        :type input_data: DataFrame

        :returns: data_snv (ndarray): Scatter corrected spectra
    """

    input_data = np.asarray(input_data)

    # Define a new array and populate it with the corrected data  
    data_snv = np.zeros_like(input_data)
    for i in range(data_snv.shape[0]):    # Apply correction
        data_snv[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
    return (data_snv)


# In[84]:


def msc(input_data, reference=None):
    ''' Perform Multiplicative scatter correction'''

    # mean centre correction
    for i in range(input_data.shape[0]):
        input_data[i,:] -= input_data[i,:].mean()

    # Get the reference spectrum. If not given, estimate it from the mean    
    if reference is None:
        # Calculate mean
        ref = np.mean(input_data, axis=0)
    else:
        ref = reference
#     print(ref.shape)
#     print(input.shape)

    ref = np.reshape(ref,-1)
    
    # Define a new array and populate it with the corrected data    
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Run regression
        in_data = np.reshape(input_data[i,:], -1)
        fit = np.polyfit(ref, in_data, 1, full=True)
        # Apply correction
        data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0] 

    return (data_msc, ref)


# In[85]:


import numpy as np
import scipy

def sgolay2d (input_data, window_size, order, derivative="none"):
    """
    """
    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0

    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial. 
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ... 
    # this line gives a list of two item tuple. Each tuple contains 
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])
    
    filtered_data = np.empty(shape = input_data.shape)
    
    for num, input_ in enumerate(input_data):
        filtered_image = np.empty(shape = input_.shape)
        for i in range(input_.shape[2]):
            z = input_[:,:,i]
            # pad input array with appropriate values at the four borders
            new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
            Z = np.zeros( (new_shape) )
            # top band
            band = z[0, :]
            Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
            # bottom band
            band = z[-1, :]
            Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band )
            # left band
            band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
            Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
            # right band
            band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
            Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
            # central band
            Z[half_size:-half_size, half_size:-half_size] = z

            # top left corner
            band = z[0,0]
            Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
            # bottom right corner
            band = z[-1,-1]
            Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band )

            # top right corner
            band = Z[half_size,-half_size:]
            Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band )
            # bottom left corner
            band = Z[-half_size:,half_size].reshape(-1,1)
            Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band )

            # solve system and convolve
            if derivative == "none":
                m = np.linalg.pinv(A)[0].reshape((window_size, -1))
                filtered_image[:,:,i] = scipy.signal.fftconvolve(Z, m, mode='valid')
            elif derivative == 'col':
                c = np.linalg.pinv(A)[1].reshape((window_size, -1))
                filtered_image[:,:,i] = scipy.signal.fftconvolve(Z, -c, mode='valid')
            elif derivative == 'row':
                r = np.linalg.pinv(A)[2].reshape((window_size, -1))
                filtered_image[:,:,i] = scipy.signal.fftconvolve(Z, -r, mode='valid')
            elif derivative == 'both':
                c = np.linalg.pinv(A)[1].reshape((window_size, -1))
                r = np.linalg.pinv(A)[2].reshape((window_size, -1))
                filtered_image[:,:,i] = scipy.signal.fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')
        filtered_data[num, :, :, :] = filtered_image
    return filtered_data


# In[86]:


def apply_filters(input_data, reference=None):
    if FILTER == "snv":
        return snv(input_data)
    elif FILTER == "msc":
        return msc(input_data, reference)[0]
    elif FILTER == "savgol":
        return sgolay2d(input_data, window_size = WINDOW, order = ORDER, derivative= DERIVATIVE)
    else:
        return input_data


# In[87]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def lda(X_train,Y_train,X_test, numComponents = NUM_OF_BANDS):
    
    assert numComponents <= min(NUM_VARIETIES-1,X_train.shape[3]),"NUM_OF_BANDS is greater."
    
    RX_train = np.reshape(X_train, (-1, X_train.shape[3]))
    RX_test = np.reshape(X_test, (-1, X_test.shape[3]))
    RY_train = []
    for i in range(Y_train.shape[0]):
        for x in range(X_train.shape[1]*X_train.shape[2]):
            RY_train.append(Y_train[i])
    RY_train = np.array(RY_train)
    
    lda = LinearDiscriminantAnalysis(n_components=numComponents)
    RX_train = lda.fit_transform(RX_train, RY_train)
    RX_test = lda.transform(RX_test)
    
    X_train = np.reshape(RX_train, (-1,X_train.shape[1],X_train.shape[2], numComponents))
    X_test = np.reshape(RX_test, (-1,X_test.shape[1],X_test.shape[2], numComponents))
    
    return X_train,X_test


# In[88]:


from sklearn.decomposition import PCA

def pca_loading(inp,numComponents = NUM_OF_BANDS):
    t = inp.reshape(-1, inp.shape[2])
    pca = PCA(n_components = numComponents)
    dt = pca.fit_transform(t)
    dt = dt.reshape(inp.shape[0],inp.shape[1],-1)
    return dt


# In[89]:


import matplotlib.pyplot as plt
#just for checking the number of bands to take into account. 99.97% is good enough to consider.
def check_pca_bands(inp):
    t = inp.reshape(-1, inp.shape[2])
    pca = PCA(n_components = 75)
    principalComponents = pca.fit_transform(t)
    ev=pca.explained_variance_ratio_
    plt.figure(figsize=(12, 6))
    plt.plot(np.cumsum(ev))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()
    
    return np.cumsum(ev)        


# In[90]:


## Dimensional Reduction Method
def ipca(HSI, numComponents = NUM_OF_BANDS):
    print(HSI.shape)
    RHSI = np.reshape(HSI, (-1, HSI.shape[2]))
    print(RHSI.shape)
    n_batches = 10
    inc_pca = IncrementalPCA(n_components=numComponents)
    for X_batch in np.array_split(RHSI, n_batches):
        inc_pca.partial_fit(X_batch)
    X_ipca = inc_pca.transform(RHSI)
    print(X_ipca.shape)
    RHSI = np.reshape(X_ipca, (HSI.shape[0],HSI.shape[1], -1))
    print(RHSI.shape)
    return RHSI


# In[91]:


def feature_extraction(X_train,Y_train,X_test,Y_test,method="none"):
    if method=="none":
        pass
    elif method == "pca_loading":
        X_train = np.array([pca_loading(inp) for inp in X_train])
        X_test = np.array([pca_loading(inp) for inp in X_test])
    elif method == "lda":
        X_train,X_test = lda(X_train,Y_train,X_test)
    elif method == "ipca":
        X_train = np.array([ipca(inp) for inp in X_train])
        X_test = np.array([ipca(inp) for inp in X_test])
    
    return X_train,Y_train,X_test,Y_test


# In[92]:


# v = 'PBW 343'
# images = extractRawImages(v)
# train_dataset,train_dataset_label,test_dataset,test_dataset_label = createDataset(images, v)
# train_dataset,test_dataset = remove_noisy_bands(REMOVE_NOISY_BANDS,train_dataset,test_dataset)
# train_dataset = apply_filters(train_dataset)
# test_dataset = apply_filters(test_dataset)
# train_dataset,train_dataset_label,test_dataset,test_dataset_label = feature_extraction(train_dataset,train_dataset_label,test_dataset,test_dataset_label,FEATURE_EXTRACTION)
# save_dataset(v, train_dataset,train_dataset_label,test_dataset,test_dataset_label)


# In[93]:


for v in VARIETIES:
    images = extractRawImages(v)
    train_dataset,train_dataset_label,test_dataset,test_dataset_label = createDataset(images, v)
    train_dataset,test_dataset = remove_noisy_bands(REMOVE_NOISY_BANDS,train_dataset,test_dataset)
    train_dataset = apply_filters(train_dataset)
    test_dataset = apply_filters(test_dataset)
    train_dataset,train_dataset_label,test_dataset,test_dataset_label = feature_extraction(train_dataset,train_dataset_label,test_dataset,test_dataset_label,FEATURE_EXTRACTION)
    save_dataset(v, train_dataset,train_dataset_label,test_dataset,test_dataset_label)


# In[ ]:





# In[ ]:




