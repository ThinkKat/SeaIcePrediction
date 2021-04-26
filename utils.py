import os
import numpy as np

def dataGenerartor(path, X, y = None, X_dt =None, y_dt=None, BATCH_SIZE = 8, transpose = False, usePath = False):
    X_batch_data = []
    y_batch_data = []
    i = 0
    if y is None:
        y = [None] * len(X)

    for (features, label) in zip(X, y):
        obs =  obs_generator(path, features, label, transpose = transpose, usePath=usePath)
        X_data = obs[0]
        y_data = obs[1]
        if X_dt is not None:
            X_data = X_dt.transform(X_data)
        if y_dt is not None:
            y_data = y_dt.transform(y_data)
        i += 1
        X_batch_data.append(X_data)
        y_batch_data.append(y_data)
        if i == BATCH_SIZE:
            i = 0
            yield (np.array(X_batch_data), np.array(y_batch_data))
            X_batch_data = []
            y_batch_data = []

def obs_generator(path, features, label=None, transpose=False, usePath=False):
    '''

    usePath: using os.path.join (if False, faster than using usePath, but you have to check the file path)
    '''
    
    featuresArr = []
    for feature in features:
        if usePath:
            feature_path = os.path.join(path, feature)
        else:
            feature_path = path + feature
        featureArr = np.load(feature_path)
        # transpose
        if transpose:
            featureArr = np.transpose(featureArr).reshape(-1)
        else:
            featureArr = featureArr.reshape(-1)
        featuresArr.append(featureArr)
    featuresArr = np.array(featuresArr)

    if label is not None:
        if usePath:
            label_path = os.path.join(path, label)
        else:
            label_path = path + feature
        labelArr = np.load(label_path)
        if transpose:
            labelArr = np.transpose(labelArr).reshape(1, -1)
        else:
            labelArr = labelArr.reshape(1, -1)

        return featuresArr, labelArr
    else:
        return featuresArr

def fit_data_generator(path, features, label, transpose = False):
    obs = obs_generator(path, features, label, transpose)
    return obs
    

