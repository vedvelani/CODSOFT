#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('Churn_Modelling.csv')


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.shape


# In[6]:


print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])


# In[7]:


data.info()


# In[8]:


data.isnull().sum()


# In[9]:


data.describe(include='all')


# In[10]:


data.columns


# In[11]:


data = data.drop(['RowNumber', 'CustomerId', 'Surname'],axis=1)


# In[12]:


data.head()


# In[13]:


data['Geography'].unique()


# In[14]:


data = pd.get_dummies(data,drop_first=True)


# In[15]:


data.head()


# In[16]:


data['Exited'].value_counts()


# In[17]:


import seaborn as sns


# In[18]:


sns.countplot(data['Exited'])


# In[19]:


X = data.drop('Exited',axis=1)
y = data['Exited']


# In[36]:


get_ipython().system('pip install imbalanced-learn')


# In[37]:


from imblearn.over_sampling import SMOTE


# In[38]:


X_res,y_res = SMOTE().fit_resample(X,y)


# In[39]:


y_res.value_counts()


# In[ ]:





# In[40]:


from sklearn.model_selection import train_test_split


# In[41]:


X_train,X_test,y_train,y_test=train_test_split(X_res,y_res,test_size=0.20,random_state=42)


# In[42]:


from sklearn.preprocessing import StandardScaler


# In[43]:


sc= StandardScaler()


# In[44]:


X_train=sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[45]:


X_train


# In[47]:


from sklearn.linear_model import LogisticRegression


# In[48]:


log = LogisticRegression()


# In[49]:


log.fit(X_train,y_train)


# In[50]:


y_pred1 = log.predict(X_test)


# In[51]:


from sklearn.metrics import accuracy_score


# In[31]:


accuracy_score(y_test,y_pred1)


# In[52]:


accuracy_score(y_test,y_pred1)


# In[53]:


from sklearn.metrics import precision_score,recall_score,f1_score


# In[33]:


precision_score(y_test,y_pred1)


# In[54]:


precision_score(y_test,y_pred1)


# In[34]:


recall_score(y_test,y_pred1)


# In[55]:


recall_score(y_test,y_pred1)


# In[35]:


f1_score(y_test,y_pred1)


# In[56]:


f1_score(y_test,y_pred1)


# In[ ]:





# In[57]:


from sklearn import svm


# In[58]:


svm = svm.SVC()


# In[59]:


svm.fit(X_train,y_train)


# In[60]:


y_pred2 = svm.predict(X_test)


# In[61]:


accuracy_score(y_test,y_pred2)


# In[62]:


precision_score(y_test,y_pred2)


# In[ ]:





# In[63]:


from sklearn.neighbors import KNeighborsClassifier


# In[64]:


knn = KNeighborsClassifier()


# In[65]:


knn.fit(X_train,y_train)


# In[66]:


y_pred3 = knn.predict(X_test)


# In[67]:


accuracy_score(y_test,y_pred3)


# In[68]:


precision_score(y_test,y_pred3)


# In[ ]:





# In[69]:


from sklearn.tree import DecisionTreeClassifier


# In[70]:


dt = DecisionTreeClassifier()


# In[71]:


dt.fit(X_train,y_train)


# In[72]:


y_pred4 = dt.predict(X_test)


# In[73]:


accuracy_score(y_test,y_pred4)


# In[74]:


precision_score(y_test,y_pred4)


# In[ ]:





# In[75]:


from sklearn.ensemble import RandomForestClassifier


# In[76]:


rf = RandomForestClassifier()


# In[77]:


rf.fit(X_train,y_train)


# In[78]:


y_pred5 = rf.predict(X_test)


# In[79]:


accuracy_score(y_test,y_pred5)


# In[80]:


precision_score(y_test,y_pred5)


# In[ ]:





# In[81]:


from sklearn.ensemble import GradientBoostingClassifier


# In[82]:


gbc = GradientBoostingClassifier()


# In[83]:


gbc.fit(X_train,y_train)


# In[84]:


y_pred6 = gbc.predict(X_test)


# In[85]:


accuracy_score(y_test,y_pred6)


# In[86]:


precision_score(y_test,y_pred6)


# In[ ]:





# In[87]:


final_data=pd.DataFrame({'Models':['LR','SVC','KNN','DT','RF','GBC'],
                        'ACC':[accuracy_score(y_test,y_pred1),
                              accuracy_score(y_test,y_pred2),
                              accuracy_score(y_test,y_pred3),
                              accuracy_score(y_test,y_pred4),
                              accuracy_score(y_test,y_pred5),
                              accuracy_score(y_test,y_pred6)]})


# In[88]:


final_data


# In[89]:


import seaborn as sns


# In[90]:


sns.barplot(final_data['Models'],final_data['ACC'])


# In[ ]:





# In[91]:


final_data=pd.DataFrame({'Models':['LR','SVC','KNN','DT','RF','GBC'],
                        'PRE':[precision_score(y_test,y_pred1),
                              precision_score(y_test,y_pred2),
                              precision_score(y_test,y_pred3),
                              precision_score(y_test,y_pred4),
                              precision_score(y_test,y_pred5),
                              precision_score(y_test,y_pred6)]})


# In[92]:


final_data


# In[93]:


sns.barplot(final_data['Models'],final_data['PRE'])


# In[ ]:





# In[94]:


X_res=sc.fit_transform(X_res)


# In[95]:


rf.fit(X_res,y_res)


# In[96]:


import joblib


# In[97]:


joblib.dump(rf,'churn_predict_model')


# In[98]:


model = joblib.load('churn_predict_model')


# In[ ]:





# In[99]:


data.columns


# In[100]:


model.predict([[619,42,2,0.0,0,0,0,101348.88,0,0,0]])


# In[ ]:





# In[101]:


from tkinter import *
from sklearn.preprocessing import StandardScaler
import joblib


# In[102]:


def show_entry_fields():
    p1=int(e1.get())
    p2=int(e2.get())
    p3=int(e3.get())
    p4=float(e4.get())
    p5=int(e5.get())
    p6=int(e6.get())
    p7=int(e7.get())
    p8=float(e8.get())
    p9=int(e9.get())
    if p9 == 1:
        Geography_Germany=1
        Geography_Spain=0
        Geography_France=0
    elif p9 == 2:
        Geography_Germany=0
        Geography_Spain=1
        Geography_France=0
    elif p9 == 3:
        Geography_Germany=0
        Geography_Spain=0
        Geography_France=1  
    p10=int(e10.get())
    model = joblib.load('churn_model')
    result=model.predict(sc.transform([[p1,p2,p3,p4,
                           p5,p6,
                           p7,p8,Geography_Germany,Geography_Spain,p10]]))
    
    if result == 0:
        Label(master, text="No Exit").grid(row=31)
    else:
        Label(master, text="Exit").grid(row=31)
    
    
master = Tk()
master.title("Bank Customers Churn Prediction Using Machine Learning")


label = Label(master, text = "Customers Churn Prediction Using ML"
                          , bg = "black", fg = "white"). \
                               grid(row=0,columnspan=2)


Label(master, text="CreditScore").grid(row=1)
Label(master, text="Age").grid(row=2)
Label(master, text="Tenure").grid(row=3)
Label(master, text="Balance").grid(row=4)
Label(master, text="NumOfProducts").grid(row=5)
Label(master, text="HasCrCard").grid(row=6)
Label(master, text="IsActiveMember").grid(row=7)
Label(master, text="EstimatedSalary").grid(row=8)
Label(master, text="Geography").grid(row=9)
Label(master,text="Gender").grid(row=10)


e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)


e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10,column=1)

Button(master, text='Predict', command=show_entry_fields).grid()

mainloop()


# In[ ]:




