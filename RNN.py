import pandas as pd
import utils
import display
import custom_metrics
import os
from keras import callbacks
from keras.utils import to_categorical

# PARAMETRIZATION
min_occ = 30
train_size = 0.8
val_size = 0.1
epochs = 50
patience_stop = 4
lookforward = 1

# SIMULATION INPUT DATA
# sims = [application, output_seq, units1, units2, dropout, max_samples, learning_rate, batch, embedding_size, lookback,
# weighted_loss, critical_ids, critical_multiplier, threshold]
#
# application --> 0 - RNN
#                 1 - RNN multilabel
#                 2 - Autoencoder
# output_seq --> output sequence length (only applicable for RNN multilabel and autoencoder, assigned to 1 for RNN)
# units1, units2 --> number of neurons of first and second RNN layers respectively (units2 = 0 for single RNN layer)
# dropout --> value for both dropout and recurrent dropout
# max_samples --> limit number of samples for a single ID by applying undersampling (-1 to define no limit)
# learning_rate --> simulation learning rate
# batch --> simulation batch size
# embedding_size --> size for the ID input embedding layer (0 to use onehot instead embedding)
# lookback --> number of past samples to focus when predicting future IDs
# weighted_loss --> 0 - standard categorical crossentropy loss
#                   1 - ln weighted categorical crossentropy loss
#                   2 - mean weighted categorical crossentropy loss
#                   3 - mix weighted categorical crossentropy loss
# critical_ids, critical_multiplier --> list of IDs to apply the defined extra multiplier
# threshold --> Select prediction with max prob only if > threshold (-1 to disable, assigned to -1 for RNN-Autoencoder)
sims = [[0, 3, 128, 0, 0.2, -1, 0.001, 512, 100, 10, 1, [6, 8], 2, 0.5]]
print('NUMBER OF SIMULATIONS: {}'.format(len(sims)))

# READ INPUT DATA
data_path = os.path.join(os.getcwd(), 'data')
data_files = [f for f in os.listdir(data_path) if '.csv' in f]
turb_names = [int(name.partition('.')[0]) for name in data_files]

# MAIN LOOP FOR EACH SIMULATION INPUT DATA
for sim_ind, comb in enumerate(sims):

    # DECODE INPUT DATA
    app = comb[0]
    output_seq = comb[1]
    units1 = comb[2]
    units2 = comb[3]
    dropout = comb[4]
    max_samples = comb[5]
    lrate = comb[6]
    batch_size = comb[7]
    emb_size = comb[8]
    lookback = comb[9]
    loss_weights = comb[10]
    crit_ids = comb[11]
    multiplier = comb[12]
    threshold = comb[13]
    act_layer, name, autoenc, data_pre, loss_name, output_seq, units2, threshold = utils.config_params(
        app, loss_weights, emb_size, output_seq, units2, threshold)

    # TAG SIMULATION FOR REFERENCE
    inp = [units1, units2, data_pre, loss_name, multiplier, crit_ids, max_samples, dropout, lrate, batch_size, lookback]
    name = utils.simulation_tag(name, inp, threshold, sim_ind)

    # DATA READING, PREPROCESSING, PREPARATION AND SET SPLITS
    df_list = [pd.read_csv(os.path.join(data_path, file), names=['time', 'ID', 'wind_speed', 'power'])
               for file in data_files]
    ids_to_remove, df_list = utils.data_scrubbing(df_list, min_occ, id_max=6000)
    total_rows, df_list, id_dict, nclasses = utils.feature_engineering(df_list, ids_to_remove, lookback, lookforward)
    ids, time, power, wind, target = utils.data_preparation(total_rows, nclasses, lookback, lookforward, df_list,
                                                            output_seq, id_dict, turb_names, shuffle=True)
    train_data, val_data, test_data = utils.stratified_train_val_test_split(target, train_size, val_size, power, ids,
                                                                            wind, time)

    # TARGET PROCESSING
    train_target, val_target, test_target, train_data, weightsID = utils.target_processing(
        train_data, val_data[0], test_data[0], app, nclasses, max_samples, loss_weights, crit_ids, multiplier)

    # INPUT DATA PROCESSING
    train_inputdata, val_inputdata, test_inputdata = utils.inputdata_processing(train_data, val_data, test_data)

    # IDs
    train_ids = train_data[1]
    val_ids = val_data[1]
    test_ids = test_data[1]
    if emb_size == 0:
        train_ids = to_categorical(train_data[1], num_classes=nclasses)
        val_ids = to_categorical(val_data[1], num_classes=nclasses)
        test_ids = to_categorical(test_data[1], num_classes=nclasses)

    # DISPLAY DATA INPUT SUMMARY
    in_feats = train_inputdata.shape[2]
    train_steps = -(-train_inputdata.shape[0] // batch_size)
    val_steps = -(-val_inputdata.shape[0] // batch_size)
    test_steps = -(-test_inputdata.shape[0] // batch_size)
    print('\nTRAIN INPUT DATA SHAPE: {}'.format(train_inputdata.shape))
    print('VAL INPUT DATA SHAPE: {}'.format(val_inputdata.shape))
    print('TEST INPUT DATA SHAPE: {}'.format(test_inputdata.shape))
    print('TRAIN IDs SHAPE: {}'.format(train_ids.shape))
    print('VAL IDs SHAPE: {}'.format(val_ids.shape))
    print('TEST IDs SHAPE: {}'.format(test_ids.shape))
    print('TRAIN TARGET SHAPE: {}'.format(train_target.shape))
    print('VAL TARGET SHAPE: {}'.format(val_target.shape))
    print('TEST TARGET SHAPE: {}'.format(test_target.shape))
    print('\nTRAIN STEPS: {} (real value = {:.2f})'.format(train_steps, train_inputdata.shape[0] / batch_size))
    print('VAL STEPS: {} (real value = {:.2f})'.format(val_steps, val_inputdata.shape[0] / batch_size))
    print('TEST STEPS: {} (real value = {:.2f})'.format(test_steps, test_inputdata.shape[0] / batch_size))

    # LOSS DEFINITION
    if app == 1:
        loss = custom_metrics.weighted_crossentropy_loss(weights=weightsID, binary=True, name=loss_name)
    else:
        loss = custom_metrics.weighted_crossentropy_loss(weights=weightsID, binary=False, name=loss_name)

    # METRIC DEFINITION
    if app == 1:
        if threshold != -1:
            mets = [custom_metrics.MultilabelTotalAccuracy(name='Total_multilabel_acc', threshold=threshold),
                    custom_metrics.MultilabelTotalPrecision(name='Total_multilabel_precision', threshold=threshold),
                    custom_metrics.PredictionRatio(name='Pred_ratio', threshold=threshold)]
        else:
            mets = [custom_metrics.MacroF1(name='F1_macro', in_classes=nclasses, multilabel=True),
                    custom_metrics.MacroTPR(name='TPR_macro', in_classes=nclasses, multilabel=True),
                    custom_metrics.TotalTPR(name='TPR_total', multilabel=True),
                    custom_metrics.MacroPrecision(name='Precision_macro', in_classes=nclasses, multilabel=True),
                    custom_metrics.TotalPrecision(name='Precision_total', multilabel=True)]
    else:
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

    # CREATE MODEL
    train_gen = utils.generator([train_inputdata, train_ids], train_target, batch_size, autoencoder=autoenc)
    val_gen = utils.generator([val_inputdata, val_ids], val_target, batch_size, autoencoder=autoenc)
    if autoenc:
        model = utils.create_autoencoder(in_feats, units1, dropout, nclasses, emb_size, lrate, loss, mets)
    else:
        model = utils.create_rnn(in_feats, units1, units2, dropout, nclasses, emb_size, lrate, loss, mets, act_layer)

    # TRAIN MODEL
    history = model.fit(train_gen, steps_per_epoch=train_steps, epochs=epochs,
                        callbacks=callbacks_list, validation_data=val_gen, validation_steps=val_steps)

    # LOAD BEST MODEL AND EVALUATE PERFORMANCE IN TEST SET
    train_gen = utils.generator([train_inputdata, train_ids], train_target, batch_size, autoencoder=autoenc)
    val_gen = utils.generator([val_inputdata, val_ids], val_target, batch_size, autoencoder=autoenc)
    if autoenc:
        model = utils.create_autoencoder(in_feats, units1, dropout, nclasses, emb_size, lrate, loss, mets)
    else:
        model = utils.create_rnn(in_feats, units1, units2, dropout, nclasses, emb_size, lrate, loss, mets, act_layer)
    model.load_weights(os.path.join(os.getcwd(), 'models', name))
    test_gen = utils.generator([test_inputdata, test_ids], test_target, batch_size, autoencoder=autoenc)
    test_results = model.evaluate(test_gen, steps=test_steps)
    for i in range(len(metrics_name)):
        print('TEST {}: {:.4f}'.format(metrics_name[i].upper(), test_results[i]))
    train_results, val_results = display.plot_results(history.history, test_results, metrics_name, loss_name, tag=name)

    # PREDICT TEST SET AND GENERATE OUTPUT DATA FILES
    test_gen = utils.generator([test_inputdata, test_ids], test_target, batch_size, autoencoder=autoenc)
    preds = model.predict(test_gen, steps=test_steps)
    if app == 0:
        display.plot_classification_report(test_target, preds, multilabel=False, tag=name)
        id_acc, id_prec, id_f1 = display.plot_confusion_matrix(test_target, preds, tag=name)
        display.write_results_excel(train_results, val_results, test_results, id_acc, id_prec, id_f1, inp)
    elif app == 1:
        if threshold != -1:
            display.plot_preds_distribution(test_target, preds, threshold=threshold, tag=name)
        else:
            display.plot_classification_report(test_target, preds, multilabel=True, tag=name)
    else:
        for j in range(preds.shape[1]):
            tag = name + ' - Prediction ' + str(j + 1)
            display.plot_classification_report(test_target[:, j, :], preds[:, j, :], multilabel=False, tag=tag)
            display.plot_confusion_matrix(test_target[:, j, :], preds[:, j, :], tag=tag)
        display.plot_classification_report(test_target, preds, multilabel=False, tag=name + ' - Prediction total')
        display.plot_confusion_matrix(test_target, preds, tag=name + ' - Prediction total')
