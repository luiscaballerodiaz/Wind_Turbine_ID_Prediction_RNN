import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras import layers
from keras import optimizers
from keras import models
import collections
from keras.utils import plot_model
import os


def create_rnn(input_shape, units1, units2, dropout, output_shape, lr, loss, metrics_list):
    inp = layers.Input(shape=(None, input_shape))
    if units2 == 0:
        rnn = layers.GRU(units1, activation='relu', return_sequences=False, dropout=dropout, recurrent_dropout=dropout)(
            inp)
    else:
        rnn = layers.GRU(units1, activation='relu', return_sequences=True, dropout=dropout, recurrent_dropout=dropout)(
            inp)
        rnn = layers.GRU(units2, activation='relu', return_sequences=False, dropout=dropout, recurrent_dropout=dropout)(
            rnn)
    outputs = layers.Dense(output_shape, activation='softmax')(rnn)
    model = models.Model(inp, outputs)
    model.summary()
    #plot_model(model, show_shapes=True, to_file=os.path.join(os.getcwd(), 'plots', 'Model Diagram.png'))

    model.compile(optimizer=optimizers.Adam(learning_rate=lr), metrics=metrics_list, loss=loss)
    return model


def plot_turbine_ids(data, tag, folder='plots', maxbars=-1):
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.rcParams['font.size'] = 14
    data = pd.Series(data)
    data = data.astype(int)
    if maxbars == -1:
        maxbars = data.value_counts().shape[0] + 1
    data.value_counts()[:maxbars].plot(kind='bar', ax=ax)
    ax.set_ylabel('NUMBER OF OCCURRENCES', fontsize=16, fontweight='bold')
    ax.set_xlabel('ID MESSAGES', fontsize=16, fontweight='bold')
    ax.set_title(tag.upper() + '(' + str(data.shape[0]) + ' datapoints)', fontweight='bold', fontsize=24)
    ax.tick_params(axis='x', labelrotation=70)
    ax.tick_params(axis='both', labelsize=14)
    ax.bar_label(ax.containers[0])
    ax.grid(visible=True)
    fig.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), folder, tag + '.png'), bbox_inches='tight')
    plt.close()


def generator(samples, target, batch_samples):
    ind = 0
    while True:
        if (ind + batch_samples) > target.shape[0]:
            gen_samples = samples[ind:, :, :]
            gen_target = target[ind:, :]
            ind = 0
        else:
            gen_samples = samples[ind:ind+batch_samples, :, :]
            gen_target = target[ind:ind+batch_samples, :]
            ind += batch_samples
        yield gen_samples, gen_target


def stratified_train_val_test_split(y, train_split, val_split):
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
    return train_list, val_list, test_list


def plot_results(data, test, met, loss, tag='', subplots=None):
    if tag != '':
        tag = ' ' + tag
    if subplots is None:
        subplots = (1, len(met))
    fig, axes = plt.subplots(subplots[0], subplots[1], figsize=(20, 10))
    ax = axes.ravel()
    best_ind = np.argmin(data['val_loss'])
    for i in range(len(met)):
        if i > 0:
            loss = met[i]
        ax[i].plot(range(1, len(data[met[i]]) + 1), data[met[i]], ls='-', lw=2, color='b',
                   label='Train ' + loss)
        ax[i].plot(range(1, len(data[met[i]]) + 1), data['val_' + met[i]], ls='--', lw=2, color='b',
                   label='Validation ' + loss)
        ax[i].set_xlabel('EPOCHS', fontweight='bold', fontsize=12)
        ax[i].set_ylabel(met[i].upper(), fontweight='bold', fontsize=12)
        text_str = '\nBEST MODEL TRAIN ' + loss.upper() + ' = {:.4f}'.format(data[met[i]][best_ind])
        text_str += '\nBEST MODEL VAL ' + loss.upper() + ' = {:.4f}'.format(data['val_' + met[i]][best_ind])
        text_str += '\nBEST MODEL TEST ' + loss.upper() + ' = {:.4f}'.format(test[i])
        ax[i].set_title(text_str, fontweight='bold', fontsize=12)
        ax[i].grid(visible=True)
        ax[i].legend()
    text_str = '\nBEST MODEL SAVED AT EPOCH {}'.format(best_ind + 1)
    plt.suptitle('SIMULATION RESULTS' + tag.upper() + text_str, fontweight='bold', fontsize=16)
    fig.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'plots', 'Simulation results' + tag + '.png'), bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(conf_matrix, tag=''):
    if tag != '':
        tag = ' ' + tag
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.pcolormesh(conf_matrix, cmap=plt.cm.cool)
    #plt.colorbar()
    labels = np.arange(conf_matrix.shape[0])
    yrange = [x + 0.5 for x in range(conf_matrix.shape[0])]
    xrange = [x + 0.5 for x in range(conf_matrix.shape[1])]
    plt.xticks(xrange, labels, rotation=75, ha='center')
    ax.xaxis.tick_top()
    plt.yticks(yrange, labels, va='center')
    for i in range(len(xrange)):
        for j in range(len(yrange)):
            ax.text(xrange[i], yrange[j], str(conf_matrix[j, i]),
                    ha="center", va="center", color="k", fontsize=10)
    plt.xlabel("TRUE ID", weight='bold', fontsize=16)
    plt.ylabel("PREDICTION ID", weight='bold', fontsize=16)
    text_str = ''
    ok = 0
    tot = 0
    acc = 0
    for i in range(conf_matrix.shape[0]):
        ok += conf_matrix[i, i]
        tot += sum(conf_matrix[:, i])
        acc += 100 * conf_matrix[i, i] / sum(conf_matrix[:, i])
        text_str += 'ID' + str(i) + '=' + str(round(100 * conf_matrix[i, i] / sum(conf_matrix[:, i]), 1)) + '%   '
        if (i+1) == int(-((-conf_matrix.shape[0] / 3) // 1)) or (i+1) == int(-((-conf_matrix.shape[0]*2 / 3) // 1)):
            text_str += '\n'
    acc_tot = round(100 * ok / tot, 1)
    acc_macro = round(acc / conf_matrix.shape[0], 1)
    fig.suptitle('CONFUSION MATRIX' + tag + ' - Weight Acc=' + str(acc_tot) + '% Macro Acc=' + str(acc_macro)
                 + '%', weight='bold', fontsize=14)
    ax.set_title(text_str, weight='bold', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    fig.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'plots', 'Confusion Matrix' + tag + '.png'), bbox_inches='tight')
    plt.close()
