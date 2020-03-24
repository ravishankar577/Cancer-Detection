import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
# from keras import optimizers
import tensorflow as tf 
from sklearn.model_selection import train_test_split
import pickle

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# Importing the dataset
dataset = pd.read_csv('cancer.csv')
y = dataset.iloc[:, 10].values
X = dataset.iloc[:, 1:10].values

print("X0",X[0])

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer = imputer.fit(X[:, 2:10])
X[:, 2:10] = imputer.transform(X[:, 2:10])

y[y < 3] = 0 ## Benign
y[y > 3] =  1 ## Malignant

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from keras import regularizers
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)
# Initialising the ANN
classifier = Sequential()

# Create ANN Here

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'lecun_uniform', activation = 'relu', input_shape = (9,)))

# Adding the second hidden layer
classifier.add(Dense(units = 5, kernel_initializer = 'glorot_uniform', activation = 'tanh'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

# Compiling the ANN


kFold = StratifiedKFold(n_splits=10)
for train, test in kFold.split(X_train, y_train):

    adam = tf.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999)
    classifier.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fitting the ANN to the Training set
    # checkpoint = ModelCheckpoint('output.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose= 1, mode = 'auto')
    history = classifier.fit(x=X_train,y=y_train,batch_size=20, validation_data=( X_test, y_test),epochs=60,callbacks = [reducelr], verbose=2)

    # classifier.load_weights('output.hdf5')
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    scores = cohen_kappa_score(y_test,y_pred)
    print("%s: %.2f%%" % ("cohen_kappa_score", scores))
    # summarize history for accuracy

# pickle.dump(classifier, open('neural_network_model.pkl','wb'))
classifier.save('neural_network_model')
print("history", history.history)
print("history keys", history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
