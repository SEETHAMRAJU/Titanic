import numpy as np
import sklearn as sk
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
from sklearn.model_selection import *
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import *
import warnings

warnings.filterwarnings("ignore",category=FutureWarning)


addpoly = True
plot_lc = True
train_data = pd.read_csv("train.csv") 
test_data = pd.read_csv("test.csv")
print "Train : " + str(train_data.shape) +" Test: "+str(test_data.shape)
#This counts the numebr of unique values in the given parameter list and is used to check if there are multiple people with same ID
if(train_data.PassengerId.nunique() == train_data.shape[0]):
    print "Unique!!!"

#This is for checking if there are any similar values in test set and training set. so that we have completely new elements in train and test sets
if(len(np.intersect1d(test_data.PassengerId.values,train_data.PassengerId.values)) == 0):
    print "No intersection"

#Trying to check wether we have Nan values or not by using the count method which gives the number of non - Nan values
dataHasNan = False
if(test_data.count().min() == test_data.shape[0] and train_data.count().min() == train_data.shape[0]):
    print "No NaN values"
else:
    dataHasNan = True
    print "Has Nan values"




#this is for storing the data types of each of the data columns of the data
dtype_df = train_data.dtypes.reset_index()
dtype_df.columns = ["Count","Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()
print "Taining data types information"
#print dtype_df

#ANALYSIS OF THE DATA

#to checek for Nan and also trying to get rid of them
if(dataHasNan):
    nas = pd.concat([train_data.isnull().sum(),test_data.isnull().sum()],axis=1,keys=['TrainData','TestData'])
print nas[nas.sum(axis=1)>0]


# survived vs class
print train_data[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
#groupby just puts the index specified on the left and mean gets the mean of all the displayed values and the values that are displayed are sorted on the basis of the survival values 


#gender vs survival
print train_data[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)

#SibSp vs survival
print train_data[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)

#Parch vs survival 
#this is same as above




#STARTING DATA CLEANING

#fileing  the Nan values

#Mean and standard deviation of the training set for generating randim values inthe range of 
#   [mean-std , mean+std]
mean = train_data["Age"].mean()
std = train_data["Age"].std()
train_random_ages = np.random.randint(mean-std,mean+std,size = train_data["Age"].isnull().sum())

mean = test_data["Age"].mean()
std = test_data["Age"].std()
test_random_ages = np.random.randint(mean-std,mean+std,size=test_data["Age"].isnull().sum())
#changing the Nan values to the randm generated values
train_data["Age"][np.isnan(train_data["Age"])] = train_random_ages
test_data["Age"][np.isnan(test_data["Age"])] = test_random_ages
#changing the type of the numbers to int type
train_data["Age"] = train_data["Age"].astype(int)
test_data["Age"] = test_data["Age"].astype(int)

#changing values for the Embarked putting S in place of na values
train_data["Embarked"].fillna('S',inplace = True)
test_data["Embarked"].fillna('S',inplace=True)


#changing the values of port to the given values from embarked
train_data["Port"] = train_data["Embarked"].map({'S':0,'C':1,'Q':2}).astype(int)
test_data["Port"] = test_data["Embarked"].map({'S':0,'C':1,'Q':2}).astype(int)

#deleting the embarked in the dataset
del train_data["Embarked"]
del test_data["Embarked"]

#replacing all the Nan in fares to the  median
test_data["Fare"].fillna(test_data["Fare"].median(),inplace=True)

#adding more features for ease 

#has_cabin is a feature which tells wether a person has a cabin or not


train_data["Has_Cabin"] = train_data["Cabin"].apply(lambda x:0 if type(x)==float else 1)
test_data["Has_Cabin"] = test_data["Cabin"].apply(lambda x: 0 if type(x)==float else 1)



#together data grouping
full_data = [train_data,test_data]

#generating new features liek family size and also whether a passenger is alone
for data in full_data:
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
for data in full_data:
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1,'IsAlone'] = 1

#-------------------------------------------------------------------------------------
#getting title from the names
train_data["Title"] = train_data.Name.str.extract("([A-Za-z]+)\.",expand=False)
test_data["Title"] = test_data.Name.str.extract("([A-Za-z]+)\.",expand=False)

for data in full_data:
    data["Title"] = data["Title"].replace(["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"Rare")
    data["Title"] = data["Title"].replace("Mme","Mrs")
    data["Title"] = data["Title"].replace("Ms","Miss")
    data["Title"] = data["Title"].replace("Mlle","Miss")

#creating new column or feature called family size 
for data in full_data:
    data["FamilySizeGroup"] = "Small"
#the below statement means that we choose all those exmaples where family size is 1 and then set hteir familly gourp size to alonne
    data.loc[data["FamilySize"]==1,"FamilySizeGroup"] = "Alone"

#the same as above
    data.loc[data["FamilySize"]>=5,"FamilySizeGroup"] = "Big"

#average survival rate for different family size
train_data[["FamilySize","Survived"]].groupby(["FamilySize"],as_index=False).mean()

#mapping the string values to numeric values for the math
for data in full_data:
    data["Sex"] = data["Sex"].map({"male":0,"female":1}).astype(int)
#changing the ages to a set of digital values
for data in full_data:
    data.loc[data["Age"]<=14,"Age"] = 0
    data.loc[data["Age"]>64,"Age"] = 4
    data.loc[(data["Age"]>14) & (data["Age"]<=32),"Age"] = 1
    data.loc[(data["Age"]>32) & (data["Age"]<=48),"Age"] = 2
    data.loc[(data["Age"]>48) & (data["Age"]<=64),"Age"] = 3

#changing the discrete fare values to digital values
for data in full_data:
    data.loc[data["Fare"]<=7.91,"Fare"] = 0
    data.loc[data["Fare"] > 31,"Fare"] = 3
    data.loc[(data["Fare"]>7.91) & (data["Fare"] <= 14.454),"Fare"] = 1
    data.loc[(data["Fare"]>14.454) & (data["Fare"] <= 31),"Fare"] = 2
    data["Fare"] = data["Fare"].astype(int)

#mapping the title to numeric values
title_map = {"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Rare":5}
family_map = {"Small":0,"Alone":1,"Big":2}
for data in full_data:
    data["Title"] = data["Title"].map(title_map)
    data["FamilySizeGroup"] = data["FamilySizeGroup"].map(family_map)


#adding new features
for data in full_data:
    data["IsChildandRich"] = 0
    data.loc[(data["Age"]<=0)&(data["Pclass"]==1),"IsChildandRich"] = 1
    data.loc[(data["Age"]<=0) & (data["Age"] == 2),"IsChildandRich"] = 1

#changing the cabins value to high low medium or x
for data in full_data:
    data["Cabin"] = data["Cabin"].fillna("X")
    data["Cabin"] = data["Cabin"].apply(lambda a:str(a)[0])
    data["Cabin"] = data["Cabin"].replace(["A","D","E","T"],"M")
    data["Cabin"] = data["Cabin"].replace(["B","C"],"H")
    data["Cabin"] = data["Cabin"].replace(["F","G"],"L")
    data["Cabin"] = data["Cabin"].map({"X":0,"L":1,"M":2,"H":3}).astype(int)

#Deleting some extra features
del train_data["Name"]
del test_data["Name"]

del train_data["SibSp"]
del test_data["SibSp"]

del train_data["Parch"]
del test_data["Parch"]

del train_data["FamilySize"]
del test_data["FamilySize"]

del train_data["Cabin"]
del test_data["Cabin"]

del train_data["Ticket"]
del test_data["Ticket"]

del train_data["Port"]
del test_data["Port"]

print "------------------Data cleaning finished----------------------------"
#ENDING DATA CLEANING
print "Dimensions : test : "+str(test_data.shape) + " train : "+str(train_data.shape)




del train_data["PassengerId"]


train_X = train_data.drop("Survived",axis=1)
train_Y = train_data["Survived"]
test_X = test_data.drop("PassengerId",axis = 1).copy()
print train_X.shape,train_Y.shape,test_X.shape


#here we are trying to make a polynomial hypothesis and use it as a funciton for our prediction
if addpoly:
    all_data = pd.concat((train_X,test_X),ignore_index=True)
    scaler = MinMaxScaler()
    scaler.fit(all_data)
    all_data = scaler.transform(all_data)
    poly = PolynomialFeatures(3) #determines the highest degree of the features
    all_data = poly.fit_transform(all_data)
    train_X = all_data[:train_data.shape[0]]
    test_X = all_data[train_data.shape[0]:]

    print train_X.shape
    print train_Y.shape
    print test_X.shape
'''
#Can be written in another file
train_X = pd.read_csv("trainx.csv")
train_Y = pd.read_csv("trainy.csv")
test_X = pd.read_csv("testx.csv")
train_data = pd.read_csv("traind.csv") 
test_data = pd.read_csv("testd.csv")
'''








#for random shuffling
cv = ShuffleSplit(n_splits=100,test_size = 0.2,random_state =0)
                    #number of re shufflings  portion of data into test   seed for generation of random number
logreg = LogisticRegression()
def Learning_Curve(X,Y,model,cv,train_sizes):
    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes , train_scores , test_scores = learning_curve(model,X,Y,cv=cv,n_jobs=4,train_sizes = train_sizes)
    
    #taking the means and standard deviations for scores
    train_scores_mean = np.mean(train_scores,axis=1)
    train_scores_std = np.std(train_scores,axis=1)
    test_scores_mean = np.mean(test_scores,axis=1)
    test_scores_std = np.std(test_scores,axis=1)
    plt.grid()

    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean + train_scores_std,alpha=0.1,color="r")
    plt.fill_between(train_sizes,test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,alpha=0.1,color="g")

    plt.plot(train_sizes,train_scores_mean,"o-",color="r",label="Training")
    plt.plot(train_sizes,test_scores_mean,"o-",color="g",label="Cross validation")

    plt.legend(loc="best")

    return plt

if plot_lc:
    train_size = np.linspace(.1,1.0,15)
    Learning_Curve(train_X,train_Y,logreg,cv,train_size)

logreg = LogisticRegression()
logreg.fit(train_X,train_Y)
pred_Y = logreg.predict(test_X)

train_res = logreg.score(train_X,train_Y)
val_res = cross_val_score(logreg,train_X,train_Y,cv=5).mean()
print train_res,val_res


#Support vector machines that is using classification 

svm = SVC(C=0.15,gamma=0.1)
svm.fit(train_X,train_Y)
Y_pred2 = svm.predict(test_X)

train_res = svm.score(train_X,train_Y)
val_res = cross_val_score(svm,train_X,train_Y,cv=5).mean()
print train_res,val_res

print pred_Y.shape,Y_pred2.shape

#Random Forest Classifier

random_forest = RandomForestClassifier(criterion="gini",n_estimators=1000,min_samples_split=10,min_samples_leaf=1,oob_score=True,max_features="auto",random_state=1,n_jobs=-1)
seed = 42
random_forest = RandomForestClassifier(n_estimators=1000, criterion="entropy",max_depth=5,min_samples_split=2,min_samples_leaf=1,max_features="auto",bootstrap=False,oob_score=False,n_jobs=1,random_state=seed,verbose=0)
random_forest.fit(train_X,train_Y)
Y_pred3 = random_forest.predict(test_X)

result_train = random_forest.score(train_X,train_Y)
result_val = cross_val_score(random_forest,train_X,train_Y,cv=5).mean()

print result_train,result_val

print Y_pred3.shape
ans = []
for i in range(0,Y_pred3.shape[0]):
	a = pred_Y[i]
	b = Y_pred2[i]
	c = Y_pred3[i]
	if(((a&b) | (c&a) | (b&c)) == 0):
		ans.append(0)
	else:
		ans.append(1)
answ = pd.DataFrame({"PassengerId":test_data["PassengerId"],"Survived":ans})
answ.to_csv("answer.csv",index=False)
