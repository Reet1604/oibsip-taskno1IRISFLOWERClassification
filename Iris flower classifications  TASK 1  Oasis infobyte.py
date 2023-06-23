#!/usr/bin/env python
# coding: utf-8
AUTHOR- > REETU KHATRI
# #  IRIS FLOWER   data description

# In[1]:


# Iris flower has three species; setosa, versicolor, and virginica, which differs according to their
# measurements. Now assume that we have the measurements of the iris flowers according to
# their species, and  task is to train a machine learning model that can learn from the
# measurements of the iris species and classify them.


# we will able to identify several interesting patterns and relationships between the different features of the dataset. Our analysis revealed clear distinctions between the three types of Iris flowers based on their sepal length, sepal width, petal length, and petal width. We also noticed some outliers in the dataset that may warrant further investigation

# In[2]:


# Import the requierd labrary
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns


# In[3]:


# Download the dataset
iris = pd.read_csv('Iris.csv')


# In[4]:


iris


# In[5]:


#  find the top 5 rows
iris.head()


# In[6]:


# find the bottom 5 rows
iris.tail()


# In[7]:


# Basic information of dataset dtypes: flIn this Dataset 5 columns and 150 rows ,also we can check Datatype of of columns ,float64(4), object(1)

iris.info()


# In[8]:


# Drop unwanted columns ID  , here axis =1 for columns and 0 for rows
iris= iris.drop(['Id'], axis=1)


# In[9]:


iris


# In[10]:


iris.head()


# In[11]:


# Change the names of the columns , as our requirements .

iris.columns


# In[12]:


iris.columns= ['sepal_length','sepal_width','petal_length','petal_width','species']


# In[13]:


iris.columns


# In[14]:


# Number of rows and columns in the dataset
iris.shape


# In[100]:


iris.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# In[15]:


# Basis statistical measures of the dataset
# BY describe function we can find MIN , MAX , MEAN , STANDERED DEVIATION , counts of the dataset
iris.describe()


# In[109]:


iris.hist()


# In[114]:


iris.plot(kind='hist',subplots=True,layout=(2,2), figsize =(20,10) ,sharex= False, sharey= False , title='Histogram')
plt.show()


# In[16]:


# find the duplicate value of the dataset Return boolean Series denoting duplicate rows. Considering certain columns is optional.


iris.duplicated()


# In[17]:


iris.duplicated().value_counts()


# In[18]:


# Null values in the dataset
iris.isnull()


# In[19]:


iris.isnull().sum()


# In[20]:


# Count of species column of the dataset
iris_species =iris["species"].value_counts()


# In[21]:


print("Count of the species column of the dataset : ")
iris_species


# In[22]:


iris.head()


# In[23]:


# Calculate the mean of sepal length for each species
mean_sepal_length= iris["sepal_length"].mean()


# In[24]:


mean_sepal_length


# In[25]:


# we found average sepal lenght is different , but Iris-setosa and Iris-virginica has quite dieffrent from each other .
iris_mean_sepal_lenght= iris.groupby('species')['sepal_length'].mean()
print( "mean of sepal lenght of each species")
iris_mean_sepal_lenght


# In[26]:


# Find the median of each species of petal width 
median_petal_width= iris.groupby('species')['petal_width'].median()
print('median of petal lenght of each species')
median_petal_width


# In[27]:


#  plotting the species of iris , vizualization

plt.figure(figsize=(5,5))
sns.pairplot(iris,hue='species')
plt.show()

EXPLANATIONS
From the above diagrams it clear that -

1* petal Lenght is more dependent on depal
2* Sepal Lenght is more dependent on sepal width and vice versa
3* petal width is more dependent on sepal lenght
# In[115]:


iris.corr()


# In[28]:


sns.violinplot(x ='sepal_width',y='sepal_length',data=iris,)
plt.title('sepal lenght and width',color='Green')
plt.show()


# In[117]:


plt.figure(figsize=(10,8))
sns.heatmap(iris.corr(),annot=True ,cmap="gist_rainbow")
plt.title ('correlation heatmap of iris features', fontsize =18)
plt.show()


# In[29]:


# plotting the count plot 
iris['species'].value_counts()


# In[30]:


# count plot 

sns.countplot(data=iris,x='species',ec='black')
plt.xlabel("Name of the Species")
plt.ylabel("Count")
plt.grid(True)
plt.show()


# In[31]:


# Bar Plot species btw sepal length 
plt.bar(iris['species'],iris['sepal_length'],color="red")
plt.title("species v/s Species")
plt.xlabel("Name of the Species")
plt.ylabel("Sepal length")
plt.grid(True)
plt.show()


# In[32]:


# Bar Plot bw sepal width and species
plt.bar(iris['species'],iris['sepal_width'],color="yellow")
plt.title("species v/s Species")
plt.xlabel("Name of the Species")
plt.ylabel("Sepal length")
plt.grid(True)
plt.show()


# In[33]:


sns.barplot(data=iris,x='sepal_width' ,y= 'species',)
plt.show()


# In[34]:


# Barplot  btw species vs petal_length
plt.bar(iris['species'],iris['petal_length'],color="green")
plt.title("Petal Length VS Species")
plt.xlabel("Name of the Species")
plt.ylabel("Petal Length")
plt.grid(True)
plt.show()


# In[35]:


# Bar plot btw petal width and species

plt.bar(iris['species'],iris['petal_width'],color="pink")
plt.title("Petal width VS Species")
plt.xlabel("Name of the Species")
plt.ylabel("Petal width")
plt.grid(True)
plt.show()


# In[36]:


plt.bar(iris['species'],iris['petal_width'])
plt.title("Petal Width VS Spcies")
plt.xlabel("Name of the Species")
plt.ylabel("Petal Width")
plt.grid(True)
plt.show()


# # model selection

# In[ ]:


Iris Flower dataset has target value so its supervised data ,so classify this dataset we can use sklearn's ml models .

Scikit-Learn, also known as sklearn is a python library to implement machine learning models and statistical modelling. Through scikit-learn, we can implement various machine learning models for regression, classification, clustering, and statistical tools for analyzing these models.

Clssification -->
Identifying which category an object belongs to.
Applications: Spam detection, image recognition.
Algorithms: SVM, nearest neighbors, random forest

Regression  -->
Predicting a continuous-valued attribute associated with an object.
Applications: Drug response, Stock prices.
Algorithms: SVR, nearest neighbors, random forest
# In[37]:


#importing Scikit-learn library for model selection
import sklearn


# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


from sklearn.linear_model import LinearRegression, LogisticRegression


# In[40]:


from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score 


# In[41]:


from sklearn.metrics import accuracy_score, classification_report , mean_absolute_error, mean_squared_error,r2_score


# In[42]:


from sklearn.preprocessing import StandardScaler


# In[43]:


from sklearn.svm import SVC


# In[44]:


# Make the data X and Y


# In[45]:


iris.head()


# In[46]:


X = iris.drop('species',axis = 1)


# In[47]:


X


# In[48]:


type(X)


# In[49]:


Y = iris['species']


# In[50]:


Y


# In[51]:


type(Y)


# In[ ]:


# Split the data into training and testing sets


# In[54]:


X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state =15)


# In[55]:


X_train.head()


# In[56]:


X_train.shape


# In[57]:


type(X_train)


# In[58]:


X_test.head()


# In[59]:


X_test.shape


# In[60]:


type(X_test)


# In[61]:


Y_train.head()


# In[62]:


Y_train.shape


# In[63]:


type(Y_train)


# In[64]:


Y_test.head()


# In[65]:


Y_test.shape


# In[66]:


type(Y_test)


# In[67]:


# Standardized X with the help of StandardScaler

StandardScaler removes the mean and scales each feature/variable to unit variance. This operation is performed feature-wise in an independent way. StandardScaler can be influenced by outliers (if they exist in the dataset) since it involves the estimation of the empirical mean and standard deviation of each feature.
# In[74]:


scaler=StandardScaler()


# In[76]:


X_train =scaler.fit_transform(X_train)


# In[77]:


X_train


# In[78]:


X_train.shape


# In[79]:


type(X_train)


# In[81]:


X_test=scaler.fit_transform(X_test)


# In[82]:


X_test


# In[83]:


X_test.shape


# In[84]:


type(X_train)


# In[85]:


print(X_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(Y_train.shape)


# In[86]:


# Creating a logestic regression model


# In[87]:


model=LogisticRegression()


# In[88]:


model.fit(X_train,Y_train)


# In[ ]:


# Prediction


# In[89]:


y_pred=model.predict(X_test)


# In[90]:


y_pred


# In[91]:


print(X_test.shape)
print(y_pred.shape)


# In[92]:


# Training accuracy


# In[93]:


train_accuracy= model.score(X_train,Y_train)


# In[94]:


print("The training accuracy is",train_accuracy)


# In[95]:


# Test accuracy


# In[96]:


test_accuracy=model.score(X_test,Y_test)


# In[97]:


print("The testing accuracy is",test_accuracy)

Final conclusion
The testing accuracy is 1.0 which is higher than training accuracy.This suggests that model will make accurate prediction on new sample.
The training accuarcy is 0.966 which is slightly less than testing accuracy.This indicates that the model fit the training data quite well.
Overall model exhibits high accuracy on both the training and testing data, which is a positive outcome.
# In[ ]:





# In[ ]:




