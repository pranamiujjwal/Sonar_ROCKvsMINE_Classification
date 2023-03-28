#!/usr/bin/python3
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# # Load Data
# df=pd.read_csv("sonar_data.csv",header=None)
# # print(df.shape)  # (208, 61)
# # print(df[60].value_counts())  # Labels


# labels=df[60]
# # print(labels)
# sonar_data=df.iloc[:, :60]
# # print(sonar_data)

# # find Best random_state and split for model
# max_accuracy=0
# best_random_state=0
# best_split=0
# for split in range(1,10):
#   local_max_accuracy=0
#   local_best_random_state=0
#   for state in range(60):
#     X_train,X_test,Y_train,Y_test=train_test_split(sonar_data , labels, test_size=split/10, random_state=state)
#     # print(X_train.shape, X_test.shape)

#     classifier=LogisticRegression()
#     classifier.fit(X_train,Y_train)
#     X_pred=classifier.predict(X_test)
#     score=accuracy_score(X_pred,Y_test)

#     if score>local_max_accuracy:
#       local_max_accuracy=score
#       local_best_random_state=state

#     # print(f"random_state: {i}, Accuracy: {score}")
#   print(f"Max accuracy: {local_max_accuracy}, Best random_state: {local_best_random_state}")

#   if local_max_accuracy>max_accuracy:
#     max_accuracy=local_max_accuracy
#     best_random_state=local_best_random_state
#     best_split=split/10

# print(f"Best random_state: {best_random_state}, Best split: {best_split}, Max accuracy: {max_accuracy}")
# # Best random_state: 12, Best split: 0.1, Max accuracy: 0.9523809523809523


# # Train Model
# X_train,X_test,Y_train,Y_test=train_test_split(sonar_data , labels, test_size=0.1, random_state=12)
# classifier=LogisticRegression()
# classifier.fit(X_train,Y_train)

# # Save Model
# import pickle
# with open('sonar_classifier.pkl','wb') as file:
  # pickle.dump(classifier, file)

# # load model
import pickle
with open('sonar_classifier.pkl','rb') as file:
  classifier=pickle.load(file)

# # Predict
# input_data=(0.02,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.066,0.2273,0.31,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.555,0.6711,0.6415,0.7104,0.808,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.051,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.018,0.0084,0.009,0.0032)
input_data=input().split()
input_data=np.asarray(input_data).reshape(1,-1)
if classifier.predict(input_data)=="R":
  print("Rock")
else:
  print("Mine")