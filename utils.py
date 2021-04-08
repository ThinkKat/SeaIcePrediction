import os
import numpy as np

def dataGenerartor(X, y, X_dt =None, y_dt=None,BATCH_SIZE = 8):
    X_batch_data = []
    y_batch_data = []
    i = 0
    for (features, label) in zip(X, y):
        obs =  obs_generate(features, label)
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

def obs_generator(path, features, label=None, transpose=False):
    featuresArr = []
    for feature in features:
        feature_path = os.path.join(path, feature)
        featureArr = np.load(feature_path)
        # transpose
        if transpose:
            featureArr = np.transpose(featureArr).reshape(-1)
        else:
            featureArr = featureArr.reshape(-1)
        featuresArr.append(featureArr)
    featuresArr = np.array(featuresArr)

    if label is not None:
        label_path = os.path.join(path, label)
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
    

