import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam


##### PREPROCCESING ######
data = pd.read_csv('admissions_data.csv')
data.head(10)
data.describe().T
data.info()

plt.figure(figsize=(8,8))
sns.pairplot(data.drop(columns=['Serial No.']))
plt.show()

plt.figure(figsize=(8,8))
corr_mat = data.drop(columns=['Serial No.']).corr()
sns.heatmap(corr_mat,annot=True)
plt.show()

labels = data.iloc[:,-1]
features = data.drop(columns=['Serial No.','Chance of Admit '])

training_set,test_set, labels_train,labels_test = train_test_split(features,labels,test_size = 0.25,random_state=1)

col = training_set.select_dtypes(include=np.number).columns
ct = ColumnTransformer([('Only numeric',StandardScaler(),col)],remainder='passthrough',verbose=1)
train_sc = ct.fit_transform(training_set)
test_sc = ct.transform(test_set)


###### MODEL ######

model = Sequential(name='Admissions')
input = InputLayer(input_shape=(train_sc.shape[1],))
model.add(input)
model.add(Dense(64,activation='relu'))
model.add(Dense(8,activation='softmax'))
model.add(Dense(1))
opt = Adam(learning_rate=0.001)
model.compile(loss='mse',metrics=['mae'],optimizer=opt)
e_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=20)
print(model.summary())

##### TRAINING #####

history = model.fit(train_sc,labels_train,epochs=100,batch_size=3,verbose=1,validation_split = 0.12,callbacks=[e_stop])

##### RESULTS #####

r_mse,r_mae = model.evaluate(test_sc,labels_test,verbose=0)
print("MSE, MAE: ", r_mse,r_mae)

y_pred = model.predict(test_sc,verbose=0)
r_score = r2_score(labels_test, y_pred)
print(r_score) # R^2 ~ 0.8012124426327143


fig = plt.figure(figsize=(14,10))
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')
 
  # Plot loss and val_loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')
 
# used to keep plots from overlapping each other  
fig.tight_layout()
fig.savefig('/content/drive/MyDrive/Colab Notebooks/my_plots.png')


