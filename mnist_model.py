# This uses a CNN to train, same as Keras documentation
#
# This is pretty slow on the VM, each epoch took 10 minutes!

# build the model
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten 
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
import pickle
import mlflow
import mlflow.keras
import sys
import time

start_time = time.time()

batch_size = 128

file_path = sys.argv[1] 
epochs = int(sys.argv[2])

# input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10
(x_train, x_test, y_train, y_test) = pickle.load( open( file_path, "rb" ) )
input_shape = (28, 28, 1)

# build model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# show models, text or layer form.
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
mlflow.log_param("batch_size", batch_size)
mlflow.set_tag("app", "mnist_model")
mlflow.set_tag("framework", "keras")
mlflow.set_tag("method", "tensorflow")
mlflow.log_metric("loss", score[0])
mlflow.log_metric("accuracy", score[1])
mlflow.log_metric("process_time", time.time() - start_time)


mlflow.keras.log_model(model, "model")
