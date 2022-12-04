import matplotlib.pyplot as plt
from resource import Observatory
import numpy as np

back_track = 365
k9 = 500
b = np.exp(np.arcsinh(k9 / 2) / 9)
k_range = np.array([0, 5, 10, 20, 40, 70, 120, 200, 330, 500]) * k9 / 500
outliers = []
past_ki_values = dict()


def quiet_curve(obs: Observatory, d_index, slot):
    bt_data = []
    for i in range(1, 31):
        if tuple([d_index - i, slot]) not in outliers:
            bt_data.append(np.expand_dims(obs.data[d_index - i][slot], axis=0))
    bt_data = np.concatenate(bt_data, axis=0)
    return np.mean(bt_data, axis=0)


def ki_variation(obs: Observatory, d_index, slot):
    index = tuple([d_index, slot])
    if index in past_ki_values:
        return past_ki_values[index]

    if d_index >= obs.data.shape[0] or d_index < 0 or slot >= obs.data.shape[1] or slot < 0:
        raise IndexError
    data = obs.data[d_index][slot]

    if tuple([d_index, slot]) in outliers:
        past_ki_values[index] = np.nan
        return np.nan

    q_curve = quiet_curve(obs, d_index, slot)
    data = data - q_curve
    V = np.max(data) - np.min(data)
    past_ki_values[index] = V
    return V


def extract_features_and_label(obs: Observatory, day, slot):
    if day >= obs.data.shape[0]:
        ki = np.nan
    else:
        ki = ki_variation(obs, day, slot)
    features_array = []

    d = day
    for s in range(1, 2):
        slt = slot - s
        if slt < 0:
            slt = 7
            d = d - 1
        features_array.append(ki_variation(obs, d, slt))

    for d in range(1, back_track + 1):
        features_array.append(ki_variation(obs, day - d, slot))

    return np.array(features_array), ki


def prepare_feature_matrix(ob: Observatory):
    X = []
    y = []
    for d_index in range(30 + back_track, ob.data.shape[0]):
        for slot in range(ob.data.shape[1]):
            feature, label = extract_features_and_label(ob, d_index, slot)
            X.append(feature)
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    x_mean = X[np.logical_not(np.isnan(X))].ravel().mean()
    y_mean = y[np.logical_not(np.isnan(y))].ravel().mean()
    X = np.where(np.isnan(X), x_mean, X)
    y = np.where(np.isnan(y), y_mean, y)

    print('Data Preparation Complete for the station: {sta}'.format(sta=ob.metadata['Station Name']))

    return X, y


def k_scale(v):
    K = -1
    for k in range(len(k_range) - 1):
        if k_range[k] <= v < k_range[k + 1]:
            K = k
            break
    if K == -1:
        K = 9
    return K


def check_outliers(obs: Observatory):
    for di in range(obs.data.shape[0]):
        for s in range(obs.data.shape[1]):
            data = obs.data[di][s]
            if np.max(data) - np.min(data) > 1e3:
                outliers.append(tuple([di, s]))
                print('Detected Outliers. Excluding data from calculation')
