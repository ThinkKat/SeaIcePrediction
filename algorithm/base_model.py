'''
BaseModel

class for making the model

'''

class BaseModel:

    def __init__(self, in_shape, out_shape):
        '''
        in_shape: tuple
        '''
        
        self.in_shape = in_shape

    def makeModel(self, **args):
        pass

    def getModel(self):
        return self.model