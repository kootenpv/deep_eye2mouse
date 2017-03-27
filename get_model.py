import os
import json
import just
from keras.layers import (Activation, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D)
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization


def get_model(model_name=None, models_path="models/"):
    if model_name:
        model_file = models_path + model_name + ".json"
        weights_file = models_path + model_name + ".h5"
        if os.path.isfile(model_file):
            model = model_from_json(json.dumps(just.read(model_file)))
            # load weights into new model
            model.load_weights(weights_file)
            print("Loaded model from disk")
        else:
            print("Cannot read model, creating fresh one")
            # Create the model
            model = Sequential()
            model.add(BatchNormalization(input_shape=(3, 72, 128)))
            model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
            model.add(Activation('relu'))
            model.add(Dropout(0.15))
            model.add(BatchNormalization())
            model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.5))
            # to prediction
            model.add(Dense(2))
            model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer="adam")
    return model


def save_model(model, model_name, models_path="models/"):
    # serialize model to JSON
    model_file = models_path + model_name + ".json"
    weights_file = models_path + model_name + ".h5"
    just.write(json.loads(model.to_json()), model_file)
    model.save_weights(weights_file)
    print("Saved model to disk")
