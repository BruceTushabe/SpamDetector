import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

spam = pd.read_csv('emails.csv')
spam.head()


z = spam["EmailText"]
y = spam["Label"]
z_train, z_test, y_train, y_test = train_test_split(z,y,test_size=0.25)

