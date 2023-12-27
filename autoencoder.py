import pandas as pd
import numpy as np
import collections
import utils
import custom_metrics
import os
import random
from keras import callbacks
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler


# Parametrization
min_occ = 30
train_size = 0.8
val_size = 0.1
epochs = 50
patience_stop = 4
lookforward = 1
output_seq = 3

# units, max_samples, learning_rate, batch, embedding_size (0 = onehot), lookback,
# weighted_loss --> 0 - standard categorical crossentropy loss
#                   1 - ln weighted categorical crossentropy loss
#                   2 - mean weighted categorical crossentropy loss
#                   3 - mix weighted categorical crossentropy loss
# critical_ids, critical_multiplier
sims = [[128, 0.2, -1, 0.001, 512, 100, 10, 1, [6, 8], 2]]

data_path = os.path.join(os.getcwd(), 'data')
data_files = [f for f in os.listdir(data_path) if '.csv' in f]
turb_names = [int(name.partition('.')[0]) for name in data_files]

print('NUMBER OF SIMULATIONS: {}'.format(len(sims)))
for sim_ind, comb in enumerate(sims):

    units = comb[0]
    dropout = comb[1]
    max_samples = comb[2]
    lrate = comb[3]
    batch_size = comb[4]
    id_emb_out = comb[5]
    lookback = comb[6]
    loss_weights = comb[7]
    crit_ids = comb[8]
    multiplier = comb[9]

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
        id_emb_out = None
    inp = [units, data_pre, loss_name, multiplier, crit_ids, max_samples, dropout, lrate, batch_size, lookback]
    name = 'AUTOENCODER ' + str(inp[0]) + ', ' + inp[1] + ', ' + inp[2] + ' (' + str(inp[3]) + 'x' + \
           str(inp[4]) + '), max_samples=' + str(inp[5]) + ', dropout=' + str(inp[6]) + ', lr=' + str(inp[7]) + \
           ', batch=' + str(inp[8]) + ' & lookback=' + str(inp[9])
    print('\nSIMULATION NUMBER: {}'.format(sim_ind + 1))
    print('SIMULATION DETAILS: {}'.format(name))

    to_remove = []
    df_list = [pd.read_csv(os.path.join(data_path, file), names=['time', 'ID', 'wind_speed', 'power'])
               for file in data_files]
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
        df_list[i] = df

    ids = df_list[0]['ID'].unique()
    nclasses = len(ids)
    print('\nTotal number of IDs: {}'.format(nclasses))
    print('List of IDs: {}'.format(sorted(ids)))
    ordinal_ids = {}
    for index, real_id in enumerate(sorted(ids)):
        ordinal_ids[real_id] = index
    print('IDs to ordinal values: {}'.format(ordinal_ids))
    input_data -= (lookback + lookforward + output_seq - 2) * len(data_files)
    print('\nNormalized total data rows: {}'.format(input_data))

    cursor = 0
    ids = np.empty([input_data, lookback])
    time = np.empty([input_data, lookback])
    power = np.empty([input_data, lookback])
    wind = np.empty([input_data, lookback, 3])
    target = np.empty([input_data, output_seq])
    for i, df in enumerate(df_list):
        df['ID'] = df['ID'].map(ordinal_ids)
        utils.plot_turbine_ids(df['ID'], 'Turbine number ' + str(turb_names[i]), folder='plots\\turbines')
        narray = df.to_numpy()
        for j in range(lookback, narray.shape[0]-lookforward-output_seq+2):
            time[cursor] = narray[j-lookback:j, 0]
            ids[cursor] = narray[j-lookback:j, 1]
            power[cursor] = narray[j-lookback:j, 2]
            wind[cursor] = narray[j-lookback:j, 3:]
            target[cursor] = narray[j+lookforward-1:j+lookforward+output_seq-1, 1]
            cursor += 1
    t = target[:, 0]
    for ipred in range(1, target.shape[1]):
        t = np.concatenate((t, target[:, ipred]), axis=0)
    utils.plot_turbine_ids(t, 'Total data', folder='plots')

    # Shuffle data randomly
    random.seed(42)
    list_random_index = list(range(input_data))
    random.shuffle(list_random_index)
    target = target[list_random_index]
    ids = ids[list_random_index]
    time = time[list_random_index]
    power = power[list_random_index]
    wind = wind[list_random_index]

    # Split between train, val and test sets
    train_indexes, val_indexes, test_indexes = utils.stratified_train_val_test_split(target[:, 0], train_size, val_size)
    train_target = target[train_indexes]
    val_target = target[val_indexes]
    test_target = target[test_indexes]
    train_ids = ids[train_indexes]
    val_ids = ids[val_indexes]
    test_ids = ids[test_indexes]
    train_time = time[train_indexes]
    val_time = time[val_indexes]
    test_time = time[test_indexes]
    train_power = power[train_indexes]
    val_power = power[val_indexes]
    test_power = power[test_indexes]
    train_wind = wind[train_indexes]
    val_wind = wind[val_indexes]
    test_wind = wind[test_indexes]

    # Target
    train_target = to_categorical(train_target, num_classes=nclasses)
    val_target = to_categorical(val_target, num_classes=nclasses)
    test_target = to_categorical(test_target, num_classes=nclasses)

    utils.plot_turbine_ids(np.sum(val_target, axis=[0, 1]), 'Validation data', folder='plots', series=False)
    utils.plot_turbine_ids(np.sum(test_target, axis=[0, 1]), 'Test data', folder='plots', series=False)
    utils.plot_turbine_ids(np.sum(train_target, axis=[0, 1]), 'Train data', folder='plots', series=False)

    # Undersampling training set
    if max_samples != -1:
        targets = np.sum(train_target, axis=1)
        indexes = utils.undersampling(targets.tolist(), max_samples, nclasses)
        train_target = train_target[indexes]
        train_ids = train_ids[indexes]
        train_time = train_time[indexes]
        train_power = train_power[indexes]
        train_wind = train_wind[indexes]
        utils.plot_turbine_ids(np.sum(targets, axis=0), 'Train data under sampled', folder='plots', series=False)

    # Calculate weights per ID based on train data
    c = dict(collections.Counter(t))
    c = dict(sorted(c.items(), key=lambda item: item[0], reverse=False))
    train_target_counter = list(c.values())
    weightsID_ln = [1 + np.log(np.max(train_target_counter) / val) for val in list(train_target_counter)]
    weightsID_mean = [np.mean(train_target_counter) / val for val in list(train_target_counter)]
    if loss_weights == 0:
        weightsID = np.ones(nclasses)
    elif loss_weights == 1:
        weightsID = weightsID_ln
    elif loss_weights == 2:
        weightsID = weightsID_mean
    else:
        weightsID = np.mean(np.array([weightsID_mean, weightsID_ln]), axis=0)
    for i in crit_ids:
        weightsID[i] *= multiplier
    weightsID = np.array(weightsID)

    # Input data
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
    train_inputdata = np.concatenate((train_time, train_power, train_wind), axis=-1)
    val_inputdata = np.concatenate((val_time, val_power, val_wind), axis=-1)
    test_inputdata = np.concatenate((test_time, test_power, test_wind), axis=-1)

    # IDs
    if id_emb_out is None:
        train_ids = to_categorical(train_ids, num_classes=nclasses)
        val_ids = to_categorical(val_ids, num_classes=nclasses)
        test_ids = to_categorical(test_ids, num_classes=nclasses)

    print('\nTRAIN INPUT DATA SHAPE: {}'.format(train_inputdata.shape))
    print('VAL INPUT DATA SHAPE: {}'.format(val_inputdata.shape))
    print('TEST INPUT DATA SHAPE: {}'.format(test_inputdata.shape))
    print('TRAIN IDs SHAPE: {}'.format(train_ids.shape))
    print('VAL IDs SHAPE: {}'.format(val_ids.shape))
    print('TEST IDs SHAPE: {}'.format(test_ids.shape))
    print('TRAIN TARGET SHAPE: {}'.format(train_target.shape))
    print('VAL TARGET SHAPE: {}'.format(val_target.shape))
    print('TEST TARGET SHAPE: {}'.format(test_target.shape))

    in_feats = train_inputdata.shape[2]
    train_steps = -(-train_inputdata.shape[0] // batch_size)
    val_steps = -(-val_inputdata.shape[0] // batch_size)
    test_steps = -(-test_inputdata.shape[0] // batch_size)
    print('\nTRAIN STEPS: {} (real value = {:.2f})'.format(train_steps, train_inputdata.shape[0]/batch_size))
    print('VAL STEPS: {} (real value = {:.2f})'.format(val_steps, val_inputdata.shape[0]/batch_size))
    print('TEST STEPS: {} (real value = {:.2f})'.format(test_steps, test_inputdata.shape[0]/batch_size))

    loss = custom_metrics.weighted_crossentropy_loss(weights=weightsID, binary=False, name=loss_name)
    mets = [custom_metrics.MacroF1(name='F1_macro', in_classes=nclasses, multilabel=False),
            custom_metrics.MacroTPR(name='TPR_macro', in_classes=nclasses, multilabel=False),
            custom_metrics.TotalTPR(name='TPR_total', multilabel=False),
            custom_metrics.MacroPrecision(name='Precision_macro', in_classes=nclasses, multilabel=False),
            custom_metrics.TotalPrecision(name='Precision_total', multilabel=False)]

    metrics_name = ['loss']
    for m in mets:
        try:
            metrics_name.append(m.name)
        except (Exception,):
            metrics_name.append(m.__name__)
    callbacks_list = [callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=patience_stop, verbose=1),
                      callbacks.ModelCheckpoint(os.path.join(os.getcwd(), 'models', name), monitor='val_loss',
                                                save_best_only=True, mode='min', save_weights_only=True, verbose=1)]

    model = utils.autoencoder(in_feats, units, dropout, nclasses, id_emb_out, lrate, loss, mets)

    train_gen = utils.autoencoder_generator([train_inputdata, train_ids], train_target, batch_size)
    val_gen = utils.autoencoder_generator([val_inputdata, val_ids], val_target, batch_size)

    history = model.fit(train_gen, steps_per_epoch=train_steps, epochs=epochs,
                        callbacks=callbacks_list, validation_data=val_gen, validation_steps=val_steps)

    model = utils.autoencoder(in_feats, units, dropout, nclasses, id_emb_out, lrate, loss, mets)
    model.load_weights(os.path.join(os.getcwd(), 'models', name))

    # model = models.load_model(os.path.join(os.getcwd(), 'models', name), custom_objects={
    #     'TPR_macro': custom_metrics.MacroTPR(name=metrics_name[2], in_classes=nclasses),
    #     'TPR_weight': custom_metrics.WeightTPR(name=metrics_name[3], in_classes=nclasses)})

    test_gen = utils.autoencoder_generator([test_inputdata, test_ids], test_target, batch_size)
    test_results = model.evaluate(test_gen, steps=test_steps)
    for i in range(len(metrics_name)):
        print('TEST {}: {:.4f}'.format(metrics_name[i].upper(), test_results[i]))
    train_results, val_results = utils.plot_results(history.history, test_results, metrics_name, loss_name, tag=name)

    test_gen = utils.autoencoder_generator([test_inputdata, test_ids], test_target, batch_size)
    preds = model.predict(test_gen, steps=test_steps)
    for j in range(preds.shape[1]):
        tag = name + ' - Prediction ' + str(j+1)
        utils.classification_report(test_target[:, j, :], preds[:, j, :], multilabel=False, tag=tag)
        utils.plot_confusion_matrix(test_target[:, j, :], preds[:, j, :], tag=tag)
    utils.classification_report(test_target, preds, multilabel=False, tag=name + ' - Prediction total')
    utils.plot_confusion_matrix(test_target, preds, tag=name + ' - Prediction total')

