'''
Data 변형 class

filter_size: int, variable
strides: int, 
N: int, ceil(136192 / filter_size)


(12, 136192) --> (,12,filter_size)
'''

import numpy as np
import math 

class DataTransformer:

    def __init__(self, strides, filter_size):
        self.strides = strides
        self.filter_size = filter_size

    def fit(self, data):
        '''
            data: numpy array (2d)
        '''
        self.data_shape = data.shape
        tmpN = (self.data_shape[1] - self.filter_size)/self.strides
        if tmpN - int(tmpN) == 0.0:
            self.N = int(tmpN) + 1
        else:
            raise ValueError 
        
        # weight of redundant pixels 
        self.weights = np.ones(self.data_shape)
        weight = 0
        for i in range(int(self.data_shape[1]/self.strides)):
            if (self.filter_size > (i * self.strides)):
                weight += 1
                self.weights[:,i*self.strides:(i+1)*self.strides] = np.array([1/weight]*self.strides)
            elif ((self.filter_size <= (i * self.strides)) and ((i * self.strides) <= self.data_shape[1] - self.filter_size)):
                self.weights[:,i*self.strides:(i+1)*self.strides] = np.array([1/weight]*self.strides)
            else:
                weight -= 1
                self.weights[:,i*self.strides:(i+1)*self.strides] = np.array([1/weight]*self.strides)


    def transform(self, data):
        '''
            data: numpy array (2d)
            return: transformed_data
        '''
        transformed_data = []

        for i in range(self.N):
            filtered_data = data[:, i*self.strides:i*self.strides+self.filter_size]
            transformed_data.append(filtered_data)
        
        return np.array(transformed_data)

    def inverse_transform(self, data):
        '''
            data: (transforemd) numpy array (3d)
        '''
        nparr = np.zeros(self.data_shape)

        for (i, d) in enumerate(data):
            if(self.strides != self.filter_size): # 겹치는 부분 있을 때
                #겹치는 부분
                nparr[:, i*self.strides: i*self.strides+self.filter_size] += d * self.weights[:, i*self.strides: i*self.strides+self.filter_size]

            else: # 겹치는 부분 없을 때
                nparr[:, i*self.strides: i*self.strides+self.filter_size] = d
        
        return nparr