#importing packages

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


class SudokuNet:
  @staticmethod
  def build(width, height, depth, classes):     # width, height, (depth)channels of MNIST, classes total digits
    # initializing the model
    model = Sequential()
    inputShape = (height, width, depth)

    # First layer,  conv -> relu -> pool layers
    model.add(Conv2D(32, (5, 5), padding = 'same', input_shape = inputShape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Second layer, conv -> relu -> pool layers
    model.add(Conv2D(32, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # First set of FC -> Relu
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Second set of FC -> relu
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    return model