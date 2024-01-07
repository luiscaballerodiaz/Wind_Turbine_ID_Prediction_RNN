import numpy as np
import pandas as pd
from keras import layers
from keras import optimizers
from keras import models
import collections
from keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
import random
import display
import os


def config_params(app, loss_weights, id_emb_out, output_seq, units2, threshold):
    """
    Customize parameters which depends on the application under simulation
    :param app: application under simulation (0 --> RNN, 1 --> RNN multilabel and 2 --> Autoencoder)
    :param loss_weights: approach for loss weights balancing (0 --> none, 1 --> ln, 2 --> mean and 3 --> mix ln-mean)
    :param id_emb_out: number of positions for ID input embedding vector
    :param output_seq: output sequence length (to be 1 if app is RNN)
    :param units2: number of neurons in the second hidden layer (to be 0 if app is Autoencoder)
    :param threshold: minimum probability in pu to consider prediction as True (to be -1 if app is RNN or Autoencoder)
    :return: updated parameters to be used during simulation
    """
    if app == 0:
        act_layer = 'softmax'
        name = 'RNN '
        autoenc = False
        output_seq = 1
        threshold = -1
    elif app == 1:
        act_layer = 'sigmoid'
        name = 'RNN MULTILABEL '
        autoenc = False
    else:
        act_layer = None
        name = 'AUTOENCODER '
        autoenc = True
        units2 = 0
        threshold = -1
    assert(output_seq > 0)

    if loss_weights == 0:
        loss_name = 'unweighted_loss'
    elif loss_weights == 1:
        loss_name = 'ln_weighted_loss'
    elif loss_weights == 2:
        loss_name = 'mean_weighted_loss'
    else:
        loss_name = 'mix_weighted_loss'

    if id_emb_out > 0:
        data_pre = str(id_emb_out) + 'emb'
    else:
        data_pre = 'onehot'
    return act_layer, name, autoenc, data_pre, loss_name, output_seq, units2, threshold


def simulation_tag(name, inp, threshold, sim_ind):
    """
    Create a simulation identification tag to be used as reference when reporting output results
    :param name: application name
    :param inp: input list including most relevant simulation data --> [units1, units2, data_pre, loss_name, multiplier,
    crit_ids, max_samples, dropout, lrate, batch_size, lookback]
    :param threshold: minimum probability in pu to consider prediction as True (only valid for RNN multilabel)
    :param sim_ind: simulation ID number reference
    :return: simulation tag
    """
    name += str(inp[0]) + '-' + str(inp[1]) + '_' + inp[2] + '_' + inp[3] + '_(' + str(inp[4]) + 'x' + str(
        inp[5]) + ')_samples=' + str(inp[6]) + '_dropout=' + str(inp[7]) + '_lr=' + str(
        inp[8]) + '_batch=' + str(inp[9]) + '_lookback=' + str(inp[10])
    if 'multilabel' in name.lower() and threshold != -1:
        name += '_th=' + str(threshold)
    print('\nSIMULATION NUMBER: {}'.format(sim_ind + 1))
    print('SIMULATION DETAILS: {}'.format(name))
    return name


def data_scrubbing(df_list, min_occ, id_max):
    """
    Process input data to unify blade ID messages and remove repeated occurrences after applying this change.
    Moreover, it creates a list of uncommon IDs which does not have at least min_occ occurrences in all turbines.
    :param df_list: list of turbine dataframes
    :param min_occ: minimum occurrences in all turbine dataframes to consider the corresponding ID
    :param id_max: maximum ID number to limit the loop search
    :return: updated list of turbine dataframes and a list of uncommon IDs
    """
    to_remove = []
    for i, df in enumerate(df_list):
        df['ID'].replace([213, 214], 212, inplace=True)
        df['ID'].replace([340, 341], 145, inplace=True)
        df['ID'].replace([1099, 1100], 1098, inplace=True)
        df['ID'].replace([1614, 1615], 1613, inplace=True)
        df['ID'].replace([2001, 2002], 2000, inplace=True)
        df['ID'].replace([3523, 3524], 3522, inplace=True)
        df = df[df['ID'].shift() != df['ID']]
        c = dict(collections.Counter(df['ID'].values.tolist()))
        c = dict(sorted(c.items(), key=lambda item: item[1], reverse=True))
        for ids in range(id_max):
            if ids not in c.keys() or c[ids] < min_occ:
                to_remove.append(ids)
        df_list[i] = df
    return to_remove, df_list


def feature_engineering(df_list, to_remove, lookback, lookforward):
    """
    Apply feature engineering in the input data and calculate effective total samples. Moreover, the definitive ID list
    is transformed to ordinal to have all IDs between [0 - (total number of IDs - 1)]
    :param df_list: list of turbine dataframes
    :param to_remove: list of uncommon IDs
    :param lookback: number of past steps to focus when making predictions
    :param lookforward: number of steps ahead to make predictions
    :return: updated turbine dataframes, effective total samples and definitive ordinal ID list
    """
    total_rows = 0
    for i, df in enumerate(df_list):
        df = df[~df['ID'].isin(to_remove)]
        df = df[df['ID'].shift() != df['ID']]
        df['time'] = pd.to_datetime(df['time'])
        df['time'] = df['time'] - df['time'].shift()
        df['time'] = df['time'].dt.total_seconds()
        df.iloc[0, df.columns.get_loc('time')] = 0
        df['power'] = df['power'].apply(lambda row: np.where(row < 10, 0, 1))
        df['wind_speed'] = df['wind_speed'].apply(lambda row: np.where(row < 8, 0, np.where(row < 16, 1, 2)))
        df[['wind1', 'wind2', 'wind3']] = pd.get_dummies(df['wind_speed'])
        df.drop(columns='wind_speed', inplace=True)
        total_rows += df.shape[0]
        df_list[i] = df
    ids = df_list[0]['ID'].unique()
    total_ids = len(ids)
    print('\nTotal number of IDs: {}'.format(total_ids))
    print('List of IDs: {}'.format(sorted(ids)))
    ordinal_ids = {}
    for index, real_id in enumerate(sorted(ids)):
        ordinal_ids[real_id] = index
    print('IDs to ordinal values: {}'.format(ordinal_ids))
    total_rows -= (lookback + lookforward - 1) * len(df_list)
    print('\nNormalized total data rows: {}'.format(total_rows))
    return total_rows, df_list, ordinal_ids, total_ids


def data_preparation(total_rows, lookback, lookforward, df_list, output_seq, id_dict, names, shuffle=True, rseed=42):
    """
    Generate numpy arrays input samples according to lookback, lookforward and output_seq for the neural network.
    This data preparation process simplifies the data generator by only selecting the corresponding indexes.
    :param total_rows: total number of samples
    :param lookback: number of past steps to focus when making predictions
    :param lookforward: number of steps ahead to make predictions
    :param df_list: list of turbine dataframes
    :param output_seq: output sequence length
    :param id_dict: dictionary to map real IDs in the turbine dataframes to a predefined ID nomenclature
    :param names: turbine names to reference the corresponding plots
    :param shuffle: True to shuffle samples and mixed different turbines data (recommended)
    :param rseed: random seed to shuffle samples
    :return: numpy array for IDs and input data to introduce in the neural network
    """
    cursor = 0
    ids = np.empty([total_rows, lookback])
    time = np.empty([total_rows, lookback])
    power = np.empty([total_rows, lookback])
    wind = np.empty([total_rows, lookback, 3])
    target = np.empty([total_rows, output_seq])
    for i, df in enumerate(df_list):
        df['ID'] = df['ID'].map(id_dict)
        display.plot_turbine_ids(df['ID'], 'Turbine number ' + str(names[i]), folder='plots\\turbines')
        narray = df.to_numpy()
        for j in range(lookback, narray.shape[0] - lookforward - output_seq + 2):
            time[cursor] = narray[j - lookback:j, 0]
            ids[cursor] = narray[j - lookback:j, 1]
            power[cursor] = narray[j - lookback:j, 2]
            wind[cursor] = narray[j - lookback:j, 3:]
            target[cursor] = narray[j + lookforward - 1:j + lookforward + output_seq - 1, 1]
            cursor += 1
    t = target[:, 0]
    for ipred in range(1, target.shape[1]):
        t = np.concatenate((t, target[:, ipred]), axis=0)
    display.plot_turbine_ids(t, 'Total data', folder='plots')
    if shuffle:  # Shuffle data randomly
        random.seed(rseed)
        list_random_index = list(range(total_rows))
        random.shuffle(list_random_index)
        target = target[list_random_index]
        ids = ids[list_random_index]
        time = time[list_random_index]
        power = power[list_random_index]
        wind = wind[list_random_index]
    return ids, time, power, wind, target


def stratified_train_val_test_split(y_original, train_split, val_split, power, ids, wind, time):
    """
    Split among validation, test and train sets using a stratified approach. It generates three indexes list and slice
    the original data with this indexing to create the three sets.
    :param y_original: target data
    :param train_split: percentage for training set size (test size is calculated as 1 - train_split - val_split)
    :param val_split: percentage for validation set size (test size is calculated as 1 - train_split - val_split)
    :param power: power input data
    :param ids: IDs input data
    :param wind: wind input data
    :param time: time input data
    :return: training, validation and test splits
    """
    assert((train_split + val_split) < 1)
    y = y_original[:, 0]
    train_list = []
    val_list = []
    test_list = []
    train_samp = round(y.shape[0] * train_split)
    val_samp = round(y.shape[0] * val_split)
    train_limit = {}
    val_limit = {}
    c = dict(collections.Counter(y))
    for key, value in c.items():
        train_limit[key] = int(value * train_samp / y.shape[0])
        val_limit[key] = int(value * val_samp / y.shape[0])
    dict_train = dict.fromkeys(train_limit, 0)
    dict_val = dict.fromkeys(val_limit, 0)
    for i, tgt in enumerate(y):
        if dict_train[tgt] < train_limit[tgt]:
            dict_train[tgt] += 1
            train_list.append(i)
        elif dict_val[tgt] < val_limit[tgt]:
            dict_val[tgt] += 1
            val_list.append(i)
        else:
            test_list.append(i)
    train_target = y_original[train_list]
    val_target = y_original[val_list]
    test_target = y_original[test_list]
    train_ids = ids[train_list]
    val_ids = ids[val_list]
    test_ids = ids[test_list]
    train_time = time[train_list]
    val_time = time[val_list]
    test_time = time[test_list]
    train_power = power[train_list]
    val_power = power[val_list]
    test_power = power[test_list]
    train_wind = wind[train_list]
    val_wind = wind[val_list]
    test_wind = wind[test_list]
    return [train_target, train_ids, train_time, train_power, train_wind], \
        [val_target, val_ids, val_time, val_power, val_wind], [test_target, test_ids, test_time, test_power, test_wind]


def target_processing(train, val, test, app, n, max_samples, loss, crit_ids, multiplier):
    """
    Processing target to accommodate to neural network formatting needs by transforming it to categorical one hot
    vectors. If RNN multilabel, each output sequence target is summed and combined in a single step.
    Undersampling is applied in train data if specified in max_samples.
    ID weights are also calculated depending on train target distribution.
    :param train: training data
    :param val: validation data
    :param test: testing data
    :param app: application under simulation
    :param n: number of output classes
    :param max_samples: limit number of samples for a single ID by applying undersampling (-1 to define no limit)
    :param loss: weighting loss approach
    :param crit_ids: critical IDs to apply multiplier when weighting
    :param multiplier: multiplier to apply to critical IDs when weighting
    :return: target data and weights ID for loss calculation
    """
    if app == 1:
        train_target_original = np.clip(np.sum(to_categorical(train[0], num_classes=n), axis=1), 0, 1)
    else:
        train_target_original = to_categorical(train[0], num_classes=n)
    axis = [i for i in range(len(train_target_original.shape) - 1)]
    axis = tuple(axis)
    display.plot_turbine_ids(np.sum(train_target_original, axis=axis), 'Train data', folder='plots', series=False)
    train_target, train_data = undersampling(train_target_original, max_samples, n, train)
    display.plot_turbine_ids(np.sum(train_target, axis=axis), 'Train data under sampled', folder='plots', series=False)
    weights_id = weight_calculation(train_target, axis, n, loss, crit_ids, multiplier)

    if app == 1:
        val_target = np.clip(np.sum(to_categorical(val, num_classes=n), axis=1), 0, 1)
    else:
        val_target = to_categorical(val, num_classes=n)
    display.plot_turbine_ids(np.sum(val_target, axis=axis), 'Validation data', folder='plots', series=False)

    if app == 1:
        test_target = np.clip(np.sum(to_categorical(test, num_classes=n), axis=1), 0, 1)
    else:
        test_target = to_categorical(test, num_classes=n)
    display.plot_turbine_ids(np.sum(test_target, axis=axis), 'Test data', folder='plots', series=False)
    return train_target, val_target, test_target, train_data, weights_id


def undersampling(target, max_samples, n, data, rseed=42):
    """
    Undersample train data to ensure there is no ID with > max_samples
    :param target: target train data
    :param max_samples: limit number of samples for a single ID by applying undersampling (-1 to define no limit)
    :param n: number of output classes
    :param data: original training dataset
    :param rseed: random seed to shuffle the undersampled data
    :return: undersampled train data
    """
    if max_samples != -1:
        current = [0] * n
        index_list = []
        if len(target.shape) == 3:
            target = np.sum(target, axis=1)
        for ind in range(target.shape[0]):
            tgt = target[ind]
            for pos in range(n):
                if tgt[pos] >= 1 and current[pos] >= max_samples:
                    break
            else:
                index_list.append(ind)
                for pos in range(n):
                    if tgt[pos] >= 1:
                        current[pos] += 1
        random.seed(rseed)
        random.shuffle(index_list)
        target = target[index_list]
        data = [data[0][index_list], data[1][index_list], data[2][index_list], data[3][index_list], data[4][index_list]]
    return target, data


def weight_calculation(target, axis, n, loss, crit_ids, multiplier):
    """
    Calculate weight calculation depending on the target train IDs distribution
    :param target: target train data
    :param axis: axis to quantify the amount of target train ID
    :param n: number of output classes
    :param loss: weighting loss approach
    :param crit_ids: critical IDs to apply multiplier when weighting
    :param multiplier: multiplier to apply to critical IDs when weighting
    :return: class weights for loss calculation
    """
    train_target_counter = np.sum(target, axis=axis)
    weights_id_ln = [1 + np.log(np.max(train_target_counter) / val) for val in list(train_target_counter)]
    weights_id_mean = [np.mean(train_target_counter) / val for val in list(train_target_counter)]
    if loss == 0:
        weights_id = np.ones(n)
    elif loss == 1:
        weights_id = weights_id_ln
    elif loss == 2:
        weights_id = weights_id_mean
    else:
        weights_id = np.mean(np.array([weights_id_mean, weights_id_ln]), axis=0)
    for i in crit_ids:
        weights_id[i] *= multiplier
    return np.array(weights_id)


def inputdata_processing(train_data, val_data, test_data):
    """
    Processing input data to accommodate to neural network formatting and scaling needs
    :param train_data: training data
    :param val_data: validation data
    :param test_data: testing data
    :return: input data
    """
    scaler = StandardScaler()
    scaler.fit_transform(train_data[2])
    scaler.transform(val_data[2])
    scaler.transform(test_data[2])
    train_data[2].shape += 1,
    val_data[2].shape += 1,
    test_data[2].shape += 1,
    train_data[3].shape += 1,
    val_data[3].shape += 1,
    test_data[3].shape += 1,
    train_inputdata = np.concatenate((train_data[2], train_data[3], train_data[4]), axis=-1)
    val_inputdata = np.concatenate((val_data[2], val_data[3], val_data[4]), axis=-1)
    test_inputdata = np.concatenate((test_data[2], test_data[3], test_data[4]), axis=-1)
    return train_inputdata, val_inputdata, test_inputdata


def create_rnn(input_shape, units1, units2, dropout, nclasses, embedding_shape, lr, loss, metrics_list, activation):
    """
    Create RNN model based on GRU layers according to input parametrization, including compiling options
    :param input_shape: input data features (ID not considered)
    :param units1: number of neurons for first RNN layer
    :param units2: number of neurons for second RNN layer (if 0 --> only using single RNN layer)
    :param dropout: dropout and recurrent dropout
    :param nclasses: number of output classes
    :param embedding_shape: ID embedding size (if 0 --> no embedding layer is used)
    :param lr: learning rate
    :param loss: compiling loss
    :param metrics_list: compiling metrics
    :param activation: activation function for output layer (expected sigmoid for multilabel and softmax for one label)
    :return: neural network model
    """
    inp_data = layers.Input(shape=(None, input_shape), name='Input_data')
    if embedding_shape == 0:
        inp_emb = layers.Input(shape=(None, nclasses), name='Input_ids')
        merged = layers.concatenate([inp_data, inp_emb], axis=-1)
    else:
        inp_emb = layers.Input(shape=(None, ), name='Input_ids')
        out_emb = layers.Embedding(input_dim=nclasses, output_dim=embedding_shape, input_length=None)(inp_emb)
        merged = layers.concatenate([inp_data, out_emb], axis=-1)
    if units2 == 0:
        rnn = layers.GRU(units1, 'relu', return_sequences=False, dropout=dropout, recurrent_dropout=dropout)(merged)
    else:
        rnn = layers.GRU(units1, 'relu', return_sequences=True, dropout=dropout, recurrent_dropout=dropout)(merged)
        rnn = layers.GRU(units2, 'relu', return_sequences=False, dropout=dropout, recurrent_dropout=dropout)(rnn)
    outputs = layers.Dense(nclasses, activation)(rnn)
    model = models.Model([inp_data, inp_emb], outputs)
    model.summary()
    if activation == 'softmax':
        name = 'RNN Categorical Model Diagram.png'
    elif activation == 'sigmoid':
        name = 'RNN Multilabel Model Diagram.png'
    else:
        name = 'RNN Model Diagram.png'
    plot_model(model, show_shapes=True, to_file=os.path.join(os.getcwd(), 'plots', name))
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), metrics=metrics_list, loss=loss)
    return model


def create_autoencoder(input_shape, units, dropout, nclasses, embedding_shape, lr, loss, metrics_list):
    """
    Create Autoencoder RNN model based on GRU layers according to input parametrization, including compiling options
    :param input_shape: input data features (ID not considered)
    :param units: autoencoder neurons for RNN layers
    :param dropout: dropout and recurrent dropout
    :param nclasses: number of output classes
    :param embedding_shape: ID embedding size (if 0 --> no embedding layer is used)
    :param lr: learning rate
    :param loss: compiling loss
    :param metrics_list: compiling metrics
    :return: neural network model
    """
    inp_data = layers.Input(shape=(None, input_shape), name='Input_data')
    if embedding_shape == 0:
        inp_emb = layers.Input(shape=(None, nclasses), name='Input_ids')
        merged = layers.concatenate([inp_data, inp_emb], axis=-1, name='Input_encoder')
    else:
        inp_emb = layers.Input(shape=(None,), name='Input_ids')
        out_emb = layers.Embedding(input_dim=nclasses, output_dim=embedding_shape, input_length=None)(inp_emb)
        merged = layers.concatenate([inp_data, out_emb], axis=-1, name='Input_encoder')
    _, states = layers.GRU(units, 'relu', return_state=True, dropout=dropout, recurrent_dropout=dropout)(merged)
    inp_decoder = layers.Input(shape=(None, nclasses), name='Input_decoder')
    rnn = layers.GRU(units, 'relu', return_sequences=True, dropout=dropout,
                     recurrent_dropout=dropout)(inp_decoder, initial_state=states)
    outputs = layers.TimeDistributed(layers.Dense(nclasses, 'softmax'))(rnn)
    model = models.Model([inp_data, inp_emb, inp_decoder], outputs)
    model.summary()
    plot_model(model, show_shapes=True, to_file=os.path.join(os.getcwd(), 'plots', 'Autoencoder Model Diagram.png'))
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), metrics=metrics_list, loss=loss)
    return model


def generator(samples, target, batch_samples, autoencoder=False):
    """
    Model input data and target generator to be used during simulation
    :param samples: two position input data list --> [inputdata, IDs]
    :param target: target data
    :param batch_samples: batch size
    :param autoencoder: True if application autoencoder to additionally return decoder input data
    :return: neural network batch input
    """
    ind = 0
    while True:
        if (ind + batch_samples) > target.shape[0]:
            gen_inputdata = samples[0][ind:]
            gen_ids = samples[1][ind:]
            gen_target = target[ind:]
            ind = 0
        else:
            gen_inputdata = samples[0][ind:ind+batch_samples]
            gen_ids = samples[1][ind:ind + batch_samples]
            gen_target = target[ind:ind+batch_samples]
            ind += batch_samples
        if autoencoder:
            gen_decoder = np.zeros_like(gen_target)
            yield [gen_inputdata, gen_ids, gen_decoder], gen_target
        else:
            yield [gen_inputdata, gen_ids], gen_target
