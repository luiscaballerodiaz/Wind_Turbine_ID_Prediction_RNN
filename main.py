import pandas as pd
import numpy as np
import collections
import utils
import custom_metrics
import os
import random
import itertools
from keras import callbacks
from keras import metrics
from keras import losses
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


# Parametrization
min_occ = 30
train_size = 0.8
val_size = 0.1
patience_stop = 10
epochs = 100

lookforward = 1
weighted_loss_list = [True]
lookback_list = [10]
batch_list = [512]
learning_rate_list = [0.001]
max_samples_list = [-1]
dropout_list = [0.2]
units1_list = [128]
units2_list = [0]

data_path = os.path.join(os.getcwd(), 'data')
data_files = [f for f in os.listdir(data_path) if '.csv' in f]
params_sweep = [max_samples_list, dropout_list, units1_list, units2_list, learning_rate_list, lookback_list, batch_list,
                weighted_loss_list]
combs = list(itertools.product(*params_sweep))
print('NUMBER OF SIMULATIONS: {}'.format(len(combs)))

for sim_ind, comb in enumerate(combs):
    max_samples = comb[0]
    dropout = comb[1]
    units1 = comb[2]
    units2 = comb[3]
    learning_rate = comb[4]
    lookback = comb[5]
    batch_size = comb[6]
    weighted_loss = comb[7]
    if weighted_loss:
        loss_name = 'weighted_categorical_loss'
    else:
        loss_name = 'categorical_loss'
    name = 'RNN ' + str(units1) + '-' + str(units2) + ' units, ' + loss_name + ', max samples=' + str(max_samples) + \
           ', dropout=' + str(dropout) + ', lr=' + str(learning_rate) + ', batch=' + str(batch_size) + \
           ' and lookback=' + str(lookback)
    df_list = [pd.read_csv(os.path.join(data_path, file), names=['time', 'ID', 'wind_speed', 'power'])
               for file in data_files]
    print('\nSIMULATION NUMBER: {}'.format(sim_ind + 1))
    print('SIMULATION DETAILS: {}'.format(name))

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
        for ids in range(6000):
            if ids not in c.keys() or c[ids] < min_occ:
                to_remove.append(ids)
        df_list[i] = df

    ids = []
    input_data = 0
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
        input_data += df.shape[0]
        utils.plot_turbine_ids(df['ID'], 'Turbine number ' + str(i+1), folder='plots\\turbines')
        new_ids = df['ID'].unique()
        for new_id in new_ids:
            if new_id not in ids:
                ids.append(new_id)
        df_list[i] = df

    nclasses = len(ids)
    print('Total number of IDs: {}'.format(nclasses))
    print('List of IDs: {}'.format(sorted(ids)))
    ordinal_ids = {}
    for index, real_id in enumerate(sorted(ids)):
        ordinal_ids[real_id] = index
    print('IDs to ordinal values: {}'.format(ordinal_ids))
    input_data -= (lookback + lookforward - 1) * len(data_files)
    print('Normalized total data rows: {}'.format(input_data))

    cursor = 0
    ids = np.empty([input_data, lookback])
    time = np.empty([input_data, lookback])
    power = np.empty([input_data, lookback])
    wind = np.empty([input_data, lookback, 3])
    target = np.empty([input_data])
    for df in df_list:
        df['ID'] = df['ID'].map(ordinal_ids)
        narray = df.to_numpy()
        for j in range(lookback, narray.shape[0]-lookforward+1):
            time[cursor, :] = narray[j - lookback:j, 0]
            ids[cursor, :] = narray[j - lookback:j, 1]
            power[cursor, :] = narray[j - lookback:j, 2]
            wind[cursor, :, :] = narray[j - lookback:j, 3:]
            target[cursor] = narray[j + lookforward - 1, 1]
            cursor += 1
    utils.plot_turbine_ids(target, 'Total data', folder='plots')

    # Shuffle data randomly
    random.seed(42)
    list_random_index = list(range(input_data))
    random.shuffle(list_random_index)
    target = target[list_random_index]
    ids = ids[list_random_index, :]
    time = time[list_random_index, :]
    power = power[list_random_index, :]
    wind = wind[list_random_index, :, :]

    # Split between train, val and test sets
    train_indexes, val_indexes, test_indexes = utils.stratified_train_val_test_split(target, train_size, val_size)
    train_target = target[train_indexes]
    val_target = target[val_indexes]
    test_target = target[test_indexes]
    train_ids = ids[train_indexes, :]
    val_ids = ids[val_indexes, :]
    test_ids = ids[test_indexes, :]
    train_time = time[train_indexes, :]
    val_time = time[val_indexes, :]
    test_time = time[test_indexes, :]
    train_power = power[train_indexes, :]
    val_power = power[val_indexes, :]
    test_power = power[test_indexes, :]
    train_wind = wind[train_indexes, :, :]
    val_wind = wind[val_indexes, :, :]
    test_wind = wind[test_indexes, :, :]
    utils.plot_turbine_ids(train_target, 'Train data', folder='plots')
    utils.plot_turbine_ids(val_target, 'Validation data', folder='plots')
    utils.plot_turbine_ids(test_target, 'Test data', folder='plots')

    # Calculate weights per ID based on total train data
    c = dict(collections.Counter(train_target))
    c = dict(sorted(c.items(), key=lambda item: item[0], reverse=False))
    c_values = list(c.values())
    weightsID = [np.mean(c_values) / val for val in list(c_values)]
    weightsID = np.array(weightsID)

    # Undersampling training set
    if max_samples != -1:
        indexes = []
        c = dict(collections.Counter(train_target))
        for i, tgt in enumerate(train_target):
            if c[tgt] <= max_samples:
                indexes.append(i)
            else:
                c[tgt] -= 1
        train_target = train_target[indexes]
        train_ids = train_ids[indexes, :]
        train_time = train_time[indexes, :]
        train_power = train_power[indexes, :]
        train_wind = train_wind[indexes, :, :]
        utils.plot_turbine_ids(train_target, 'Train data under sampled', folder='plots')

    train_target = to_categorical(train_target, num_classes=nclasses)
    val_target = to_categorical(val_target, num_classes=nclasses)
    test_target = to_categorical(test_target, num_classes=nclasses)
    train_ids = to_categorical(train_ids, num_classes=nclasses)
    val_ids = to_categorical(val_ids, num_classes=nclasses)
    test_ids = to_categorical(test_ids, num_classes=nclasses)
    scaler = StandardScaler()
    scaler.fit_transform(train_time)
    scaler.transform(val_time)
    scaler.transform(test_time)
    train_time.shape += 1,
    val_time.shape += 1,
    test_time.shape += 1,
    train_power.shape += 1,
    val_power.shape += 1,
    test_power.shape += 1,
    train_samples = np.concatenate((train_ids, train_time, train_power, train_wind), axis=2)
    val_samples = np.concatenate((val_ids, val_time, val_power, val_wind), axis=2)
    test_samples = np.concatenate((test_ids, test_time, test_power, test_wind), axis=2)
    print('TRAIN SAMPLES SHAPE: {}'.format(train_samples.shape))
    print('VAL SAMPLES SHAPE: {}'.format(val_samples.shape))
    print('TEST SAMPLES SHAPE: {}'.format(test_samples.shape))

    nfeats = train_samples.shape[2]
    train_steps = -(-train_samples.shape[0] // batch_size)
    val_steps = -(-val_samples.shape[0] // batch_size)
    test_steps = -(-test_samples.shape[0] // batch_size)
    print('TRAIN STEPS: {} (real value = {:.2f})'.format(train_steps, train_samples.shape[0]/batch_size))
    print('VAL STEPS: {} (real value = {:.2f})'.format(val_steps, val_samples.shape[0]/batch_size))
    print('TEST STEPS: {} (real value = {:.2f})'.format(test_steps, test_samples.shape[0]/batch_size))

    if loss_name == 'weighted_categorical_loss':
        loss = custom_metrics.weighted_categorical_crossentropy(weights=weightsID, name=loss_name)
        metrics_list = [metrics.CategoricalCrossentropy(name='categorical_loss'),
                        custom_metrics.MacroTPR(name='TPR_macro', in_classes=nclasses),
                        custom_metrics.WeightTPR(name='TPR_weight', in_classes=nclasses)]
    else:
        loss = losses.CategoricalCrossentropy(name=loss_name)
        metrics_list = [custom_metrics.weighted_categorical_crossentropy(weights=weightsID,
                                                                         name='weighted_categorical_loss'),
                        custom_metrics.MacroTPR(name='TPR_macro', in_classes=nclasses),
                        custom_metrics.WeightTPR(name='TPR_weight', in_classes=nclasses)]
    metrics_name = ['loss']
    for m in metrics_list:
        try:
            metrics_name.append(m.name)
        except (Exception,):
            metrics_name.append(m.__name__)
    callbacks_list = [callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=patience_stop, verbose=1),
                      callbacks.ModelCheckpoint(os.path.join(os.getcwd(), 'models', name), monitor='val_loss',
                                                save_best_only=True, mode='min', save_weights_only=True, verbose=1)]

    model = utils.create_rnn(nfeats, units1, units2, dropout, nclasses, learning_rate, loss, metrics_list)

    train_gen = utils.generator(train_samples, train_target, batch_size)
    val_gen = utils.generator(val_samples, val_target, batch_size)

    history = model.fit(train_gen, steps_per_epoch=train_steps, epochs=epochs,
                        callbacks=callbacks_list, validation_data=val_gen,
                        validation_steps=val_steps)

    model = utils.create_rnn(nfeats, units1, units2, dropout, nclasses, learning_rate, loss, metrics_list)
    model.load_weights(os.path.join(os.getcwd(), 'models', name))

    # model = models.load_model(os.path.join(os.getcwd(), 'models', name), custom_objects={
    #     'TPR_macro': custom_metrics.MacroTPR(name=metrics_name[2], in_classes=nclasses),
    #     'TPR_weight': custom_metrics.WeightTPR(name=metrics_name[3], in_classes=nclasses)})

    test_gen = utils.generator(test_samples, test_target, batch_size)
    test_results = model.evaluate(test_gen, steps=test_samples.shape[0] // batch_size)
    for i in range(len(metrics_name)):
        print('TEST {}: {:.4f}'.format(metrics_name[i].upper(), test_results[i]))
    utils.plot_results(history.history, test_results, metrics_name, loss_name, tag=name, subplots=(2, 2))

    test_gen = utils.generator(test_samples, test_target, batch_size)
    preds = model.predict(test_gen, steps=test_steps)
    y_pred = np.argmax(preds, axis=1)
    confusion_matrix = np.zeros([nclasses, nclasses])
    y_true = np.argmax(test_target, axis=1)
    y_true = y_true.astype(int)
    for i in range(y_pred.shape[0]):
        confusion_matrix[y_pred[i], y_true[i]] += 1
    confusion_matrix = confusion_matrix.astype(int)
    utils.plot_confusion_matrix(confusion_matrix, name)

    print('TEST CLASSIFICATION REPORT:\n{}'.format(classification_report(y_true, y_pred, zero_division=0.0)))
    report = classification_report(y_true, y_pred, zero_division=0.0, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(os.getcwd(), 'reports', name + '.csv'))

