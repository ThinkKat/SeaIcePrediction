'''
ModelController

class for controlling the model (train, save, load, etc)

'''

import tensorflow as tf
import matplotlib.pyplot as plt

class ModelController:

    def __init__(self, modelPath, model = None):
        '''
        model: tensorflow model
        '''
        self.modelPath = modelPath
        if model is not None:
            self.model = model
        else:
            self.model = tf.keras.models.load_model(modelPath)

    def compileModel(self, optimizer, loss, learning_rate = None):
        '''
        compile model
        '''
        if optimizer is str:
            if learning_rate is not None:
                try:
                    optimizer = getattr(tf.keras.optimizers, optimizer)(learning_rate = learning_rate)
                except ValueError:
                    print("learning_rate can be only number")
            else:
                optimizer = getattr(tf.keras.optimizers, optimizer)
        
        self.model.compile(optimizer = optimizer, loss = loss)
            
                

    def trainModel(self, dataset, val_dataset, epochs, shuffle, batch_size, callbacks, steps_per_epoch, validation_steps):
        pass
        if dataset is generator or tf.utils.Sequence:
            self.history = model.fit(
                x = dataset, batch_size = batch_size, epochs = epochs, callbacks = callbacks, shuffle = shuffle, validation_data = val_dataset, steps_per_epoch = steps_per_epoch, validation_steps = validation_steps)
        else:
            self.history = model.fit(
                x = dataset[0], y=dataset[1], batch_size = batch_size, epochs = epochs, callbacks = callbacks, shuffle = shuffle, validation_data = val_dataset, steps_per_epoch = steps_per_epoch, validation_steps = validation_steps)

        return self.history

    def predict(self, dataset):
        self.pred = self.model.predict(dataset)

        return self.pred

    def visualizeHistory(self):
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        plt.plot(loss, label="train loss", color = "orange")
        plt.plot(val_loss, label="val loss", color = "royalblue")
        plt.legend(loc = "upper right")
        plt.show()

    def saveModel(self):
        '''
        save trained model to model Path
        '''
        self.model.save(self.modelPath)
        
    