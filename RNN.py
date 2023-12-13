import pandas as pd
import numpy as np
import collections
import utils
import custom_metrics
import os
import random
from keras import callbacks
from keras import metrics
from keras import losses
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler


# Parametrization
min_occ = 30
train_size = 0.8
val_size = 0.1
epochs = 100
patience_stop = 10
lookforward = 1

# units1, units2, dropout, max_samples, learning_rate, batch, embedding_size (0 = onehot), lookback,
# weighted_loss (False - standard categorical crossentropy loss, True - ln weighted categorical crossentropy loss)
# critical_ids, critical_multiplier
sims = [[128, 0, 0.2, -1, 0.001, 512, 100, 5, True, [], 1]]
        #[128, 0, 0.2, -1, 0.001, 512, 0, 10, True, [], 1]],
        #[128, 0, 0.2, -1, 0.001, 512, 0, 15, True, [], 1]],
        #[128, 0, 0.2, -1, 0.001, 512, 25, 15, True, [], 1]],
        #[128, 0, 0.2, -1, 0.001, 512, 100, 15, True, [], 1]]]

data_path = os.path.join(os.getcwd(), 'data')
data_files = [f for f in os.listdir(data_path) if '.csv' in f]
turb_names = [int(name.partition('.')[0]) for name in data_files]

print('NUMBER OF SIMULATIONS: {}'.format(len(sims)))
for sim_ind, comb in enumerate(sims):

    units1 = comb[0]
    units2 = comb[1]
    dropout = comb[2]
    max_samples = comb[3]
    lrate = comb[4]
    batch_size = comb[5]
    id_emb_out = comb[6]
    lookback = comb[7]
    loss_weights = comb[8]
    crit_ids = comb[9]
    multiplier = comb[10]

    if loss_weights:
        loss_name = 'weight_cat_loss'
    else:
        loss_name = 'cat_loss'
    if id_emb_out > 0:
        data_pre = str(id_emb_out) + '_dim_emb'
    else:
        data_pre = 'onehot'
        id_emb_out = None
    inp = [units1, units2, data_pre, loss_name, multiplier, crit_ids, max_samples, dropout, lrate, batch_size, lookback]
    name = 'RNN ' + str(inp[0]) + '-' + str(inp[1]) + ', ' + inp[2] + ', ' + inp[3] + ' (' + str(inp[4]) + ' x ' + \
           str(inp[5]) + '), max_samples=' + str(inp[6]) + ', dropout=' + str(inp[7]) + ', lr=' + str(inp[8]) + \
           ', batch=' + str(inp[9]) + ' and lookback=' + str(inp[10])
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
    input_data -= (lookback + lookforward - 1) * len(data_files)
    print('\nNormalized total data rows: {}'.format(input_data))

    cursor = 0
    ids = np.empty([input_data, lookback])
    time = np.empty([input_data, lookback])
    power = np.empty([input_data, lookback])
    wind = np.empty([input_data, lookback, 3])
    target = np.empty([input_data])
    for i, df in enumerate(df_list):
        df['ID'] = df['ID'].map(ordinal_ids)
        utils.plot_turbine_ids(df['ID'], 'Turbine number ' + str(turb_names[i]), folder='plots\\turbines')
        narray = df.to_numpy()
        for j in range(lookback, narray.shape[0]-lookforward+1):
            time[cursor] = narray[j-lookback:j, 0]
            ids[cursor] = narray[j-lookback:j, 1]
            power[cursor] = narray[j-lookback:j, 2]
            wind[cursor] = narray[j-lookback:j, 3:]
            target[cursor] = narray[j+lookforward-1, 1]
            cursor += 1
    utils.plot_turbine_ids(target, 'Total data', folder='plots')

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
    train_indexes, val_indexes, test_indexes = utils.stratified_train_val_test_split(target, train_size, val_size)
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
    utils.plot_turbine_ids(train_target, 'Train data', folder='plots')
    utils.plot_turbine_ids(val_target, 'Validation data', folder='plots')
    utils.plot_turbine_ids(test_target, 'Test data', folder='plots')

    # Calculate weights per ID based on total train data
    c = dict(collections.Counter(train_target))
    c = dict(sorted(c.items(), key=lambda item: item[0], reverse=False))
    train_target_counter = list(c.values())
    weightsID = [1 + np.log(np.max(train_target_counter) / val) for val in list(train_target_counter)]
    for i in crit_ids:
        weightsID[i] *= multiplier
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
        train_ids = train_ids[indexes]
        train_time = train_time[indexes]
        train_power = train_power[indexes]
        train_wind = train_wind[indexes]
        utils.plot_turbine_ids(train_target, 'Train data under sampled', folder='plots')

    # Target
    train_target = to_categorical(train_target, num_classes=nclasses)
    val_target = to_categorical(val_target, num_classes=nclasses)
    test_target = to_categorical(test_target, num_classes=nclasses)

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

    if loss_name == 'cat_loss':
        loss = losses.CategoricalCrossentropy(name=loss_name)
        mets = [custom_metrics.weighted_categorical_crossentropy(weights=weightsID, name='mean_weight_cat_loss'),
                custom_metrics.MacroTPR(name='TPR_macro', in_classes=nclasses),
                custom_metrics.TotalTPR(name='TPR_total', in_classes=nclasses),
                custom_metrics.MacroPrecision(name='Precision_macro', in_classes=nclasses),
                custom_metrics.MacroF1(name='F1_macro', in_classes=nclasses)]
    else:
        loss = custom_metrics.weighted_categorical_crossentropy(weights=weightsID, name=loss_name)
        mets = [metrics.CategoricalCrossentropy(name='cat_loss'),
                custom_metrics.MacroTPR(name='TPR_macro', in_classes=nclasses),
                custom_metrics.TotalTPR(name='TPR_total', in_classes=nclasses),
                custom_metrics.MacroPrecision(name='Precision_macro', in_classes=nclasses),
                custom_metrics.MacroF1(name='F1_macro', in_classes=nclasses)]

    metrics_name = ['loss']
    for m in mets:
        try:
            metrics_name.append(m.name)
        except (Exception,):
            metrics_name.append(m.__name__)
    callbacks_list = [callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=patience_stop, verbose=1),
                      callbacks.ModelCheckpoint(os.path.join(os.getcwd(), 'models', name), monitor='val_loss',
                                                save_best_only=True, mode='min', save_weights_only=True, verbose=1)]

    model = utils.create_rnn(in_feats, units1, units2, dropout, nclasses, id_emb_out, lrate, loss, mets)

    train_gen = utils.generator([train_inputdata, train_ids], train_target, batch_size)
    val_gen = utils.generator([val_inputdata, val_ids], val_target, batch_size)

    history = model.fit(train_gen, steps_per_epoch=train_steps, epochs=epochs,
                        callbacks=callbacks_list, validation_data=val_gen, validation_steps=val_steps)

    model = utils.create_rnn(in_feats, units1, units2, dropout, nclasses, id_emb_out, lrate, loss, mets)
    model.load_weights(os.path.join(os.getcwd(), 'models', name))

    # model = models.load_model(os.path.join(os.getcwd(), 'models', name), custom_objects={
    #     'TPR_macro': custom_metrics.MacroTPR(name=metrics_name[2], in_classes=nclasses),
    #     'TPR_weight': custom_metrics.WeightTPR(name=metrics_name[3], in_classes=nclasses)})

    test_gen = utils.generator([test_inputdata, test_ids], test_target, batch_size)
    test_results = model.evaluate(test_gen, steps=test_steps)
    for i in range(len(metrics_name)):
        print('TEST {}: {:.4f}'.format(metrics_name[i].upper(), test_results[i]))
    train_results, val_results = utils.plot_results(history.history, test_results, metrics_name, loss_name, tag=name)

    test_gen = utils.generator([test_inputdata, test_ids], test_target, batch_size)
    preds = model.predict(test_gen, steps=test_steps)
    y_pred = np.argmax(preds, axis=1)
    confusion_matrix = np.zeros([nclasses, nclasses])
    y_true = np.argmax(test_target, axis=1)
    y_true = y_true.astype(int)
    for i in range(y_pred.shape[0]):
        confusion_matrix[y_pred[i], y_true[i]] += 1
    confusion_matrix = confusion_matrix.astype(int)
    id_acc, id_prec = utils.plot_confusion_matrix(confusion_matrix, name)

    utils.update_results_excel(train_results, val_results, test_results, id_acc, id_prec, inp)
