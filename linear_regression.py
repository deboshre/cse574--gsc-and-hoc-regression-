#preparing the data
import pandas as pd
from pandas import DataFrame as df
from sklearn.cluster import KMeans #using Kmeans for clustering
from sklearn.utils import shuffle
import numpy as np 
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt

maxAcc = 0.0
maxIter = 0
C_Lambda = 0.03
TrainingPercent = 80
ValidationPercent = 10
ValPercent = 10
TestPercent = 10
M = 10
PHI = []
IsSynthetic = False

#preparing the datasets
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
        df_total = df_total.sample(n=3000)
    else:
        df_total.sample(n=1)
    
    df_total = shuffle(df_total)
    df_target = df_total['target']
    return df_total, df_target


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

def GenerateRawData(path, feature_operation,dataType):
    if feature_operation == 'Concat':
        df_final, df_target = prepare_concat_data(path,dataType)
    elif feature_operation == 'Subtract':
        df_final, df_target = prepare_subtract_data(path)
    return df_final, df_target

# Now that the data has been prepared, we create the training, validation and test matrices
def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t

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

def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):
    BigSigma    = np.zeros((len(Data),len(Data)))
    DataT       = np.transpose(Data) 
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])  
        varVect.append(np.var(vct)) 
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j] 
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(200,BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma

def GetScalar(DataRow,MuRow, BigSigInv):
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))  
    L = np.dot(R,T)
    return L

def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))
    PHI = np.zeros((int(TrainingLen),len(MuMatrix)))
    BigSigInv = np.linalg.pinv(BigSigma) 
    
   
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI

def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    return Y

def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))

def generate_ERMS(TrainingTarget, TRAINING_PHI, VAL_PHI, TEST_PHI, ValDataAct, TestDataAct, dataType):
    if dataType == "GSC":
        range_loop = 200
    else:
        range_loop = TrainingTarget.shape[0]
    W_Now = np.ones(M)
    La           = 2
    learningRate = 0.01
    L_Erms_Val   = []
    L_Erms_TR    = []
    L_Erms_Test  = []
    W_Mat        = []

    for i in range(0,range_loop):
        Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
        La_Delta_E_W  = np.dot(La,W_Now)
        Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
        Delta_W       = -np.dot(learningRate,Delta_E)
        W_T_Next      = W_Now + Delta_W
        W_Now         = W_T_Next
        
        #-----------------TrainingData Accuracy---------------------#
        TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
        Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
        L_Erms_TR.append(float(Erms_TR.split(',')[1]))

        #-----------------ValidationData Accuracy---------------------#
        VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
        Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
        L_Erms_Val.append(float(Erms_Val.split(',')[1]))

        #-----------------TestingData Accuracy---------------------#
        TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
        Erms_Test = GetErms(TEST_OUT,TestDataAct)
        L_Erms_Test.append(float(Erms_Test.split(',')[1]))
        
    
    print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
    print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
    print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))

def train_model(RawData, RawTarget,dataType):
    TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
    TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
    
    ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
    ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
    
    TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
    TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
    
    ErmsArr = []
    AccuracyArr = []
    kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
    Mu = kmeans.cluster_centers_
    BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)
    TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
    TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) 
    print(TestData)
    VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)
    generate_ERMS(TrainingTarget, TRAINING_PHI, VAL_PHI, TEST_PHI, ValDataAct, TestDataAct, dataType)


print ("-----------HUMAN OBSERVED FEATURE CONCAT-------------")
path = "HumanObserved-Features-Data"
df_final, df_target = GenerateRawData(path,"Concat", "HOC")
df_final.drop(['img_id_A','img_id_B','target' ], axis=1, inplace=True)
RawData = np.transpose(df_final.as_matrix())
RawTarget = df_target.as_matrix()
train_model(RawData, RawTarget,"HOC")

print ("-----------HUMAN OBSERVED FEATURE SUBTRACT-------------")
path = "HumanObserved-Features-Data"
df_final, df_target = GenerateRawData(path,"Subtract", "HOC")
RawData = np.transpose(df_final.as_matrix())
RawTarget = df_target.as_matrix()
train_model(RawData, RawTarget,"HOC")

print ("-----------GSC FEATURE CONCAT-------------")
path = "GSC-Features-Data"
df_final, df_target = GenerateRawData(path,"Concat", "GSC")
df_final.drop(['img_id_A','img_id_B','target' ], axis=1, inplace=True)
uniques = df_final.apply(lambda x: x.nunique())
df_final = df_final.drop(uniques[uniques==1].index, axis=1)

RawData = np.transpose(df_final.as_matrix())
RawTarget = df_target.as_matrix()
train_model(RawData, RawTarget,"GSC")

print ("-----------GSC FEATURE SUBTRACT-------------")
path = "GSC-Features-Data"
df_final, df_target = GenerateRawData(path,"Subtract", "GSC")
uniques = df_final.apply(lambda x: x.nunique())
df_final = df_final.drop(uniques[uniques==1].index, axis=1)
RawData = np.transpose(df_final.as_matrix())
RawTarget = df_target.as_matrix()
train_model(RawData, RawTarget,"GSC")