#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
import seaborn as sns 
data=pd.read_csv('spam.csv',encoding='latin-1')


# In[67]:


data.info()


# In[68]:


data.head()


# In[69]:


data.describe()


# In[70]:


data=data.dropna(axis=1)


# In[71]:


data.head()


# In[ ]:





# In[72]:


data.columns=['label','masseges']


# In[73]:


data.head()


# In[74]:


data['label'].unique()


# In[75]:


data.head()


# In[76]:


data['length']=data['masseges'].apply(len)


# In[77]:


data.head()


# In[78]:


sns.countplot(x='length',data=data)


# In[80]:


data.head()


# In[81]:


data['masseges'][0]


# In[82]:


len(data['masseges'])


# In[83]:


sns.countplot(x='label',data=data)


# In[84]:


data=data[['label','masseges']]


# In[40]:


import nltk


# In[41]:


import string
string.punctuation


# In[42]:


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[43]:


ps=PorterStemmer()
Wn=WordNetLemmatizer()


# In[44]:


string=['A paragraph is a series of related sentences developing a central idea, \n called the topic. Tryto think about paragraphs in terms of thematic unity: a paragraph is a sentence or a group of sentences that supports one central, unified idea. Paragraphs add one idea at a time to your broader argument']


# In[45]:


print(string)


# In[46]:


corpus=[]
for i in range(len(data)):
    review=re.sub("[^a-zA-BZ]"," ",data['masseges'][i])
    review=review.lower()
    review=review.split()
    #review=[Wn.lemmatize(c) for c in review if c not in stopwords.words('english')]
    review=[ps.stem(c) for c in review if c not in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)


# In[47]:


corpus


# In[48]:


data['corpus']=corpus


# In[49]:


data.head()


# In[50]:


y=pd.get_dummies(data['label'])
y.head()
y=y['spam']
y=pd.DataFrame(y)


# In[65]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
cv=CountVectorizer(max_features=None)
#cv=TfidfTransformer()
X=cv.fit_transform(corpus).toarray()


# In[66]:


X.shape


# In[67]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=11)


# In[ ]:





# In[68]:


from sklearn.naive_bayes import MultinomialNB
naive=MultinomialNB()


# In[69]:


naive.fit(x_train,y_train)
y_predict=naive.predict(x_test)


# In[ ]:





# In[70]:


from sklearn.metrics import confusion_matrix,classification_report,r2_score
print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test,y_predict))
print(r2_score(y_test,y_predict))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




