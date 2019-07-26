# -*- coding: utf-8 -*-
"""


@author: Abenezer
"""


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot')
import os
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

'Read the file'
df = pd.read_csv('C:/Users/Abenezer/Documents/1_Summer_2019/Paper/Driver/Data/vehicle2.csv')
'Checking the data distribution per class'
df['Class'].value_counts().plot(kind='bar', title='Number of data point per class',color='C1')
plt.ylabel('Data Points')
plt.xlabel('Classes')



def pre_process_encoder(df):

    'Features and label'
    X = df.drop('Class',1)
    y = df.Class
   
    if "Car_Id" in X.columns:
        X.drop('Car_Id', axis=1, inplace=True)
    if 'Trip'in X.columns:
        X.drop('Trip', axis=1, inplace=True)
        
    return X,y
    

X, y = pre_process_encoder(df)


'Split the data set into window samples'

from sklearn.preprocessing import LabelEncoder

def window(X1, y1):
    X_samples = []
    y_samples = []
    
     
    encoder = LabelEncoder()
    encoder.fit(y1)
    y1 = encoder.transform(y1)
    
    length = 16
    overlapsize = length//2
    n = y1.size    
 
    
    Xt = np.array(X1)
    yt= np.array(y1).reshape(-1,1)

    # for over the 263920  310495in jumps of 64
    for i in range(0, n , length-overlapsize):
        # grab from i to i+length
        sample_x = Xt[i:i+length,:]
        if (np.array(sample_x).shape[0]) == length: 
            X_samples.append(sample_x)

        sample_y = yt[i:i+length]
        if (np.array(sample_y).shape[0]) == length:
            y_samples.append(sample_y)

    return np.array(X_samples),  np.array(y_samples)


'for the label Select the maximum occuring value in the given array'
def max_occuring_label(sample):
    values, counts = np.unique(sample, return_counts=True)
    ind = np.argmax(counts)
    
    return values[ind] 


'Creating y_sample label by taking only the maximum'

def label_y(y_value):
    y_samples_1 = []
    for i in range(len(y_value)):
        y_samples_1.append(max_occuring_label(y_value[i]))
        
    return np.array( y_samples_1 ).reshape(-1,1)


from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def rnn_dimension(X,y):
    X_samples, y_samples = window(X, y)
    y_samples = label_y(y_samples)

    #Shuffling 
    from sklearn.utils import shuffle
    X_samples,  y_samples = shuffle(X_samples, y_samples)

    # to catagory
    y_samples_cat = to_categorical(y_samples)


    X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn =train_test_split(X_samples, y_samples_cat, train_size=0.85)
    X_train,  y_train = shuffle(X_train_rnn, y_train_rnn)
    
    return X_train, y_train, X_test_rnn, y_test_rnn




X_train_5,y_train_5, X_test_5,y_test_5 = rnn_dimension(X,y)


device_lib.list_local_devices()


def normalizing(X_test):
            
            dim1=X_test.shape[1]
            dim2=X_test.shape[2]

            X_test_2d = X_test.reshape(-1,dim2)
            scale = StandardScaler()
            scale.fit(X_test_2d)

            X_test_scaled = scale.transform(X_test_2d)
            X_test_scaled = X_test_scaled.reshape(-1,dim1,dim2)

            return X_test_scaled
        
        

clean_model = load_model('Model_clean_binary_cross_ICTAI_vehicle2_1.h5')

X_test_normalized = normalizing(X_test_5)
score = clean_model.evaluate(X_test_normalized, y_test_5, batch_size=50)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



anomality_level = [0,0.05,0.1,0.2,0.4,0.6,0.8]


# anomality_level = [0.05,0.1,0.2,0.4,0.6]

def LSTM_anomality(X_test_rnn,y_test_rnn ):
    acc_noise_test = []
    acc_noise_test_rf_box = []
    for anomaly in anomality_level:
        print("="*5)
        print("for anomaly percentage = ",anomaly)

        def anomality(X, ): 
            orgi_data = np.copy(X_test_5.reshape(-1,21))
            mask = np.random.choice( orgi_data.shape[0], int(len(orgi_data)* .5), replace=False)
            # orgi_data[mask].shape

            orgi_data[mask] = orgi_data[mask]+orgi_data[mask]*anomaly
            
            return orgi_data
        
        def normalizing(X_test):
            
            dim1=X_test.shape[1]
            dim2=X_test.shape[2]

            X_test_2d = X_test.reshape(-1,dim2)
            scale = StandardScaler()
            scale.fit(X_test_2d)

            X_test_scaled = scale.transform(X_test_2d)
            X_test_scaled = X_test_scaled.reshape(-1,dim1,dim2)

            return X_test_scaled

           
        iter_score = []    
        for i in range(5):
            
            X_test_rnn_anomal = np.copy(anomality(X_test_rnn).reshape(-1,X_test_5.shape[1],X_test_5.shape[2]))
            
            X_test_rnn_noise_scaled = normalizing(X_test_rnn_anomal)
           
            #pd.DataFrame(noising2(X_train.reshape(-1,49)))[1].head(1000).plot(kind='line')

            score_1 = clean_model.evaluate(X_test_rnn_noise_scaled, y_test_rnn, batch_size=50,verbose=0)
            iter_score.append(score_1[1])
#             print(score_1[1])

        dif = max(iter_score) - min(iter_score)
        score_2 = sum(iter_score)/len(iter_score)
        acc_noise_test.append(score_2)
        print('Avg Test loss:', score_2)
        print('Avg Test accuracy:', score_2)
        acc_noise_test_rf_box.append(dif)
        
    return acc_noise_test,acc_noise_test_rf_box
        
        
LSTM_acc_noise_test, LSTM_noise_acc_box = LSTM_anomality(X_test_5, y_test_5)
acc = []
fig1 = plt.figure()
for n in range(len(LSTM_acc_noise_test)):
    acc.append(LSTM_acc_noise_test[n])
    
plt.plot(anomality_level,acc)
plt.errorbar(anomality_level,LSTM_acc_noise_test, LSTM_noise_acc_box, fmt='.k', color='black', ecolor='red', elinewidth=3, capsize=0)


def normalizing_2d(X):              
           
            scale = StandardScaler()
            scale.fit(X)

            X = scale.transform(X)
            
            return X
        
def anomality_2d(X, anomaly): 

    X = np.array(X).reshape(-1,21)
    mask = np.random.choice( X.shape[0], int(len(X)* .4), replace=False)
    # orgi_data[mask].shape

    X[mask] = X[mask]+X[mask]*anomaly

    return X

X_train, X_test, y_train, y_test =train_test_split(X, y, train_size=0.85,shuffle=False)


from keras.utils import np_utils 
from sklearn.preprocessing import StandardScaler
y_dummy = np_utils.to_categorical(y)

from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import shuffle
from keras import layers
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(X, y_dummy, train_size=0.85)


X_train_scaled = np.copy(normalizing_2d(X_train))

X_train, X_test_, y_train, y_test_ =train_test_split(X_train_scaled, y_train, train_size=0.99)
X_train,  y_train = shuffle(X_train, y_train)

mlp = Sequential()
mlp.add(Dense(160, input_dim=X_train.shape[1], activation='relu'))
mlp.add(layers.BatchNormalization())
mlp.add(layers.Dropout(0.5))
mlp.add(Dense(120, activation='relu'))
mlp.add(layers.BatchNormalization())
mlp.add(Dense(y_test.shape[1], activation='softmax'))
#mlp.add(Dense(1,activation='sigmoid'))
# mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
mlp.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
with tf.device('/GPU:0'):
    mlp_history = mlp.fit(X_train,y_train, epochs=10,batch_size=100, shuffle=True)



X_test_normalized = normalizing_2d(X_test)


score = mlp.evaluate(X_test_normalized, y_test, batch_size=50)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


def mlp_acc_test(X_test, y_test):
    acc_noise_test = []
    acc_noise_test_rf_box = []
    
#     anomality_level = [0,0.2,0.4,0.6,0.8,1]
        
    for anomal in anomality_level:      

        i = 0
        iter_score = []
        while i < 5:
            X_test_anomal = np.copy(anomality_2d(X_test, anomal))
            X_test_normalized = normalizing_2d(X_test_anomal)


            score_1 = mlp.evaluate(X_test_normalized, y_test, batch_size=50)
            iter_score.append(score_1[1])
            i += 1
#             print(i)
  
        dif = max(iter_score) - min(iter_score) 
        score_2 = sum(iter_score)/len(iter_score)
        acc_noise_test.append(score_2)
        print('Avg Test loss:', score_2)
        print('Avg Test accuracy:', score_2)
        acc_noise_test_rf_box.append(dif)

    return acc_noise_test, acc_noise_test_rf_box


mlp_noise_acc, mlp_noise_acc_box  = mlp_acc_test(X_test,y_test)
acc_mlp = []


for n in range(len(mlp_noise_acc)):
    acc_mlp.append(mlp_noise_acc[n])
    
# plt.plot(noise_sig,acc_mlp)
plt.plot(anomality_level,acc_mlp)
plt.errorbar(anomality_level,acc_mlp, mlp_noise_acc_box, fmt='.k', color='black',
             ecolor='red', elinewidth=3, capsize=0)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(X, y, train_size=0.85)

X_train_scaled = np.copy(normalizing_2d(X_train))


from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics



def acc_noise_test_dt(X_train, y_train ,X_test , y_test):
    
    dt = DecisionTreeClassifier()
    dt.fit(X_train,y_train)

    acc_noise_test_dt = []
    acc_noise_test_rf_box = []



    
    
    for anomal in anomality_level:
       
        iter_score=[]
        for i in range(10):
            
            X_test_anomal = np.copy(anomality_2d(X_test, anomal))
            X_test_normalized = normalizing_2d(X_test_anomal)
           

            'Decision Tree'
            y_pred_dt = dt.predict(X_test_normalized)   
            acc_n = metrics.accuracy_score(y_test, y_pred_dt)
            
            iter_score.append(acc_n)
            
        dif = max(iter_score) - min(iter_score)    
        score_2 = sum(iter_score)/len(iter_score)
        acc_noise_test_dt.append(score_2)
        print('Avg Test loss:', score_2)
        print('Avg Test accuracy:', score_2)
        acc_noise_test_rf_box.append(dif)
            

        
    return  acc_noise_test_dt, acc_noise_test_rf_box


dt_noise_acc,dt_noise_acc_box = acc_noise_test_dt(X_train_scaled, y_train, X_test, y_test)
acc_dt = []
# anomality_level = [0,0.2,0.4,0.6,0.8,1]
for n in range(len(dt_noise_acc)):
    acc_dt.append(dt_noise_acc[n])
    
plt.plot(anomality_level,acc_dt)
plt.errorbar(anomality_level,acc_dt, dt_noise_acc_box, fmt='.k', color='black',
             ecolor='red', elinewidth=3, capsize=0)




from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics



def acc_noise_test_rf(X_train, y_train ,X_test , y_test):
    
    rf = RandomForestClassifier(n_estimators=20)
    rf.fit(X_train, y_train)

    acc_noise_test_rf = []
    acc_noise_test_rf_box = []
    
    for anomal in anomality_level:
       
        iter_score=[]
        for i in range(10):
            
            X_test_anomal = np.copy(anomality_2d(X_test, anomal))
            X_test_normalized = normalizing_2d(X_test_anomal)           

        
            'Random Forest'
            y_pred_rf =rf.predict(X_test_normalized) 
            acc_n = metrics.accuracy_score(y_test, y_pred_rf)
            iter_score.append(acc_n)
#             print(acc_n)
        
        dif = max(iter_score) - min(iter_score)
        acc_noise_test_rf_box.append(dif)
        score_2 = sum(iter_score)/len(iter_score)
        acc_noise_test_rf.append(score_2)
        
        print("=")
        print(score_2)
        
        
    return (acc_noise_test_rf,acc_noise_test_rf_box)


    
rf_noise_acc, rf_noise_acc_box = acc_noise_test_rf(X_train_scaled, y_train, X_test, y_test)
acc_rf = []

# anomality_level = [0,0.2,0.4,0.6,0.8,1]
for n in range(len(rf_noise_acc)):
    acc_rf.append(rf_noise_acc[n])
    
# plt.plot(noise_sig,acc_rf,'or')
# plt.plot()
plt.plot(anomality_level,acc_rf)
plt.errorbar(anomality_level,acc_rf, rf_noise_acc_box, fmt='.k', color='black',
             ecolor='red', elinewidth=3, capsize=0)
# plt.boxplot(noise_sig,rf_noise_acc_box)




#mpl.style.use('seaborn-poster')
fig2 = plt.figure()
plt.axis([-0.07,.82,0,1.08])
# anomality_level = [0,0.2,0.4,0.6,0.8,1]
# noise_sig = anomality_level 

plt.plot(anomality_level[:10],acc[:10], marker='^' ,label="LSTM", linewidth=3.5)
plt.plot(anomality_level[:10], acc_mlp[:10], marker='o', label="FCNN", linewidth=3.5)
plt.plot(anomality_level[:10],acc_dt[:10], marker='*', label="Decision Tree", linewidth=3.5)
plt.plot(anomality_level[:10],acc_rf[:10], marker='x', label="Random Forest", linewidth=3.5)

plt.xlabel("Percentage of sensor anomalities induced in the data (*100)" , fontsize=16)
plt.ylabel("accuracy", fontsize=20)
# plt.title("Accuracy on noisy data")
plt.legend(loc=3, fontsize=16)
#plt.grid()
