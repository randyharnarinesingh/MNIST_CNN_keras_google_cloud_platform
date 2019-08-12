from keras.models import Model
from keras.layers import Input, Dense, Activation,Flatten, Conv2D, Flatten, Dropout, MaxPooling2D, AveragePooling2D
from keras import optimizers
from keras.initializers import glorot_uniform
from keras.callbacks import TensorBoard
from tensorflow.python.lib.io import file_io

import keras.backend as K
K.set_image_data_format('channels_last')

import numpy as np
import pandas as pd
import argparse
import os

def get_args() :

    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir',type=str,required=True,help='local or GCS location for writing checkpoints and exporting models')
    parser.add_argument('--num-epochs',type=int,default=20,help='number of times to go through the data, default=20')
    parser.add_argument('--batch-size',default=128,type=int,help='number of records to read during each training step, default=128')
    parser.add_argument('--learning-rate',default=.01,type=float,help='learning rate for gradient descent, default=.01')
    parser.add_argument('--verbosity',choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],default='INFO')
    args, _ = parser.parse_known_args()
    return args

def load_data() :

  url = 'https://storage.googleapis.com/mnist-keras-on-cloud/train.csv' # link to GCP storage bucket link
  print('Downloading file from URL...')
  data = pd.read_csv(url).values
  print('File downloaded!')

  X = data[:,1:data.shape[1]]
  y = data[:,0].reshape(42000,1) # produce y as row vector
  X = X.reshape(42000,28,28,1) # put into appropriate dimensions for input to CNN
  X = X / 255 # normalise pixel values between 0-1

  Y = np.zeros((42000,10))
  for i in range(y.size) :
    Y[i,y[i,0]]=1 # build up output matrix Y

  train_x = X[0:36000,:,:,:]
  train_y = Y[0:36000,:]

  eval_x = X[36000:42000,:,:,:]
  eval_y = Y[36000:42000,:]

  return train_x, train_y, eval_x, eval_y

def create_keras_model(input_shape,learning_rate):
    X_input = Input(input_shape)
    X = X_input

    # Stage 1
    X = Conv2D(6, (5, 5), strides=(1, 1), name='conv1', activation='relu', kernel_initializer=glorot_uniform(seed=0))(X)
    X = AveragePooling2D(pool_size=(2, 2), strides=[2, 2], padding='valid', data_format=None)(X)

    # Stage 2
    X = Conv2D(16, (5, 5), strides=(1, 1), name='conv2', activation='relu', kernel_initializer=glorot_uniform(seed=0))(X)
    X = AveragePooling2D(pool_size=(2, 2), strides=[2, 2], padding='valid', data_format=None)(X)

    # Output Stage
    X = Flatten()(X)
    X = Dense(120, activation='relu', name='fc1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(84, activation='relu', name='fc2', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(10, activation='softmax', name='fc3', kernel_initializer=glorot_uniform(seed=0))(X)

    # Create and compile model
    model = Model(inputs=X_input, outputs=X, name='LeNet5')
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam,loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def train_model(args):

    ##Setting up the path for saving logs
    logs_path = args.job_dir + 'logs/tensorboard'
    train_x, train_y, eval_x, eval_y = load_data()

    # dimensions
    num_train_examples = train_x.shape[0]
    num_eval_examples = eval_x.shape[0]

    # Create the Keras Model
    keras_model = create_keras_model(input_shape=(28, 28, 1), learning_rate=args.learning_rate)

    # Setup TensorBoard callback.
    tensorboard_cb = TensorBoard(log_dir=os.path.join(args.job_dir, 'keras_tensorboard'), histogram_freq=1)

    print('Type of train_x =',type(train_x))
    print('Type of train_y =', type(train_y))
    print('Type of eval_x =', type(eval_x))
    print('Type of eval_y =', type(eval_y))

    # Train model
    keras_model.fit(x=train_x, y=train_y, epochs=args.num_epochs, verbose=1, batch_size=args.batch_size, validation_data=(eval_x,eval_y),callbacks=[tensorboard_cb])

    # Save keras model
    keras_model.save('model.h5')

    # Copy model.h5 over to Google Cloud Storage
    with file_io.FileIO('model.h5', mode='rb') as input_f:
        with file_io.FileIO(os.path.join(args.job_dir, 'model.h5'), mode='wb+') as output_f:
            output_f.write(input_f.read())
            print("Saved model.h5 to GCS")

if __name__ == "__main__":
    args = get_args()
    train_model(args)