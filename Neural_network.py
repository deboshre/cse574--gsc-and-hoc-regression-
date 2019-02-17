
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import math
from sklearn.utils import shuffle
from keras.utils import np_utils
TrainingTarget= []
RawData = []


# In[20]:


def prepare_concat_data(path,dataType):
    df_same= pd.read_csv(path+"/same_pairs.csv")
    df_diff= pd.read_csv(path+"/diffn_pairs.csv")
    df_diff= df_diff.sample(n=800)
    df= pd.read_csv(path+"/features.csv")
    df.rename(columns={'img_id':'img_id_A'}, inplace=True)
    df_merged_same= pd.merge(df_same,df, on="img_id_A")
    df.rename(columns={'img_id_A':'img_id_B'}, inplace=True)
    df_merged_same= pd.merge(df_merged_same,df, on="img_id_B")

    df_merged_diff= pd.merge(df_diff,df, on="img_id_B")
    df.rename(columns={'img_id_B':'img_id_A'}, inplace=True)
    df_merged_diff= pd.merge(df_merged_diff,df, on="img_id_A")
    df_total= df_merged_same.append(df_merged_diff)   
    if dataType == 'GSC':
        df_total = df_total.sample(n=10000)
    else:
        df_total.sample(n=1)
    print (df_total.shape)
    df_total = shuffle(df_total)
    df_target = df_total['target']
    return df_total, df_target


# In[21]:


def prepare_subtract_data(path):
    df_same= pd.read_csv(path+"/same_pairs.csv")
    df_diff= pd.read_csv(path+"/diffn_pairs.csv")
    df_diff= df_diff.sample(n=800)
    df= pd.read_csv(path+"/features.csv")
    df_total= df_same.append(df_diff)
    df_total = shuffle(df_total)
    df_target = df_total['target']
    df.rename(columns={'img_id':'img_id_A'}, inplace=True)
    df_merged_1= pd.merge(df_total,df, on="img_id_A")
    df.rename(columns={'img_id_A':'img_id_B'}, inplace=True)
    df_merged_2= pd.merge(df_total,df, on="img_id_B")
    df_merged_1.drop(['img_id_A','img_id_B','target' ], axis=1, inplace=True)
    df_merged_2.drop(['img_id_A','img_id_B','target' ], axis=1, inplace=True)
    df_total = df_merged_2.sub(df_merged_1)
    return df_total, df_target


# In[22]:


def GenerateRawData(path, feature_operation,dataType):
    if feature_operation == 'Concat':
        df_final, df_target = prepare_concat_data(path,dataType)
        df_final.drop(['img_id_A','img_id_B','target' ], axis=1, inplace=True)
    elif feature_operation == 'Subtract':
        df_final, df_target = prepare_subtract_data(path)
    return df_final, df_target


# In[23]:


def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return np_utils.to_categorical(np.array(t),2)

def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
    d2 = rawData[:,0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

def GenerateValData(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix

def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t


# ## Model Definition

# In[24]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras import optimizers

import numpy as np

input_size = 20
drop_out = 0.2
first_dense_layer_nodes  = 2048
hidden_layer_nodes = 1024
second_dense_layer_nodes = 2

def get_model():
    
    model = Sequential()
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))
   
    model.add(Dropout(drop_out))
    
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('sigmoid'))
    
    model.summary()
    
    sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


# # <font color='blue'>Creating Model</font>

# In[25]:


model = get_model()


# # <font color = blue>Run Model</font>

# In[26]:


validation_data_split = 0.2
num_epochs = 5000
model_batch_size = 128
tb_batch_size = 32
early_patience = 100

TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10

tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')

print ("-----------HUMAN OBSERVED FEATURE CONCAT-------------")
path = "HumanObserved-Features-Data"
df_final, df_target = GenerateRawData(path,"Concat", "HOC")
RawData = np.transpose(df_final.as_matrix())
RawTarget = df_target.as_matrix()
processedData = np.transpose(np.array(GenerateTrainingDataMatrix(RawData,TrainingPercent)))

processedLabel   = GenerateTrainingTarget(RawTarget,TrainingPercent)
print(processedData)
print(processedLabel)
history = model.fit(processedData
                    , processedLabel
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                   )


# # <font color = blue>Training and Validation Graphs</font>

# In[27]:


get_ipython().magic('matplotlib inline')
df = pd.DataFrame(history.history)
df.plot(subplots=True, grid=True, figsize=(10,15))


# In[28]:


wrong   = 0
right   = 0

TrainingCount = len(TrainingTarget)
valSize = int(math.ceil(len(RawData[0])*20*0.01))
V_End = TrainingCount + valSize
TestData = RawData[:,TrainingCount+1:V_End]
processedData1 = np.transpose(TestData)

valSize = int(math.ceil(len(RawTarget)*20*0.01))
V_End = TrainingCount + valSize
t =RawTarget[TrainingCount+1:V_End]
TestDataAct = np.array(t)
#print (str(ValPercent) + "% Val Target Data Generated..")

for i,j in zip(processedData1,TestDataAct):
    y = model.predict(np.array(i).reshape(-1,20))
#     predictedTestLabel.append(decodeLabel(y.argmax()))
    
    if j.argmax() == y.argmax():
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy: " + str(right/(right+wrong)*100))

