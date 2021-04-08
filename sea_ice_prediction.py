'''
SeaIcePrediction 

class for loading data, training the models and save the results.

'''

import numpy as np
from data_transformer import DataTransformer

class SeaIcePrediction:

    def __init__(self, dt_args, basePath, dataPath, modelPath, resultPath):
        self.basePath = basePath
        self.dataPath = dataPath
        self.modelPath = modelPath
        self.resultPath = resultPath
        self.make_datatransformer(dt_args)

    def loadData(self):
        '''
        load npz type data
        '''

        self.data = np.load(self.basePath + self.dataPath)

    def make_datatransformer(self,dt_args):
        self.X_dt = DataTransformer(**dt_args)
        self.y_dt = DataTransformer(**dt_args)
    


    
    