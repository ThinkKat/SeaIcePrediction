import tensorflow as tf
from utils import obs_generator, fit_data_generator

class DataGenerator:

    def __init__(self, path, trainDataset, valDataset, X_dt, y_dt, transpose = False, usePath=False):
        self.path = path
        self.X_train_data = trainDataset[0]
        self.y_train_data = trainDataset[1]
        self.X_val_data = valDataset[0]
        self.y_val_data = valDataset[1]
        self.transpose = transpose
        self.usePath = usePath
        self.X_dt = X_dt
        self.y_dt = y_dt

        fit_data = fit_data_generator(path, self.X_train_data[0], self.y_train_data[0], self.transpose)
        X_fit_data = fit_data[0]
        y_fit_data = fit_data[1]
        self.X_dt.fit(X_fit_data)
        self.y_dt.fit(y_fit_data)

    def getDataset(self):
        if self.transpose:
            ds_train_dataset = tf.data.Dataset.from_generator(self._getTrainTransposeGenerator, output_types = (tf.float32, tf.float32))
            ds_val_dataset = tf.data.Dataset.from_generator(self._getValTransposeGenerator, output_types = (tf.float32, tf.float32))
        else:
            ds_train_dataset = tf.data.Dataset.from_generator(self._getTrainGenerator, output_types = (tf.float32, tf.float32))
            ds_val_dataset = tf.data.Dataset.from_generator(self._getValGenerator, output_types = (tf.float32, tf.float32))

        return ds_train_dataset, ds_val_dataset

    def _getTrainGenerator(self):
        for (X, y) in zip(self.X_train_data, self.y_train_data):
            obs = obs_generator(self.path, X, y, usePath=self.usePath)
            X_data = self.X_dt.transform(obs[0])
            y_data = self.y_dt.transform(obs[1])
            yield X_data, y_data

    def _getValGenerator(self):
        for (X, y) in zip(self.X_val_data, self.y_val_data):
            obs = obs_generator(self.path, X, y,usePath=self.usePath)
            X_data = self.X_dt.transform(obs[0])
            y_data = self.y_dt.transform(obs[1])
            yield X_data, y_data

    def _getTrainTransposeGenerator(self):
        for (X, y) in zip(self.X_train_data, self.y_train_data):
            obs = obs_generator(self.path, X, y, True,usePath=self.usePath)
            X_data = self.X_dt.transform(obs[0])
            y_data = self.y_dt.transform(obs[1])
            yield X_data, y_data

    def _getValTransposeGenerator(self):
        for (X, y) in zip(self.X_val_data, self.y_val_data):
            obs = obs_generator(self.path, X, y, True,usePath=self.usePath)
            X_data = self.X_dt.transform(obs[0])
            y_data = self.y_dt.transform(obs[1])
            yield X_data, y_data

    
            
    
