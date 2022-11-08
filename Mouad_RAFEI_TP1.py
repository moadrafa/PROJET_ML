#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


imdb_neg = pd.read_pickle(r'C:\Users\Administrator\Desktop\imdb_raw_neg.pickle')
imdb_pos = pd.read_pickle(r'C:\Users\Administrator\Desktop\imdb_raw_pos.pickle')


# In[8]:


df1 = pd.DataFrame(imdb_neg)
df1['sentiment'] = 0
df1.rename(columns={0 : 'review'}, inplace = True)


# In[9]:


df1


# In[10]:


df2 = pd.DataFrame(imdb_pos)
df2['sentiment'] = 1
df2.rename(columns={0 : 'review'}, inplace = True)


# In[11]:


df2


# In[12]:


sep = {"négatif": df1, "positif": df2}

result = pd.concat(sep)


# In[13]:


result


# In[14]:


def no_of_words(text):
    words= text.split()
    word_count = len(words)
    return word_count


# In[15]:


result['word count'] = result['review'].apply(no_of_words)


# In[16]:


def data_processing(text):
    text= text.lower()
    text = re.sub('', '', text)
    text = re.sub(r"https\S+|www\S+|http\S+", '', text, flags = re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)


# In[19]:


result.review = result['review'].apply(data_processing)


# In[20]:


stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data


# In[21]:


result.review = result['review'].apply(lambda x: stemming(x))


# In[22]:


result['word count'] = result['review'].apply(no_of_words)
result.head()


# In[23]:


result


# In[24]:


x = result['review']
y = result['sentiment']


# In[25]:


vect = TfidfVectorizer()
x = vect.fit_transform(result['review'])


# In[26]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# In[27]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[28]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_pred = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))


# In[30]:


print(confusion_matrix(y_test, logreg_pred))
print("\n")
print(classification_report(y_test, logreg_pred))


# In[ ]:




