import keras.backend as K
import tensorflow as tf
import numpy as np
# K.print_tensor(tensor, message='\n \n', summarize=-1)


def weighted_crossentropy_loss(weights, binary=False, name='loss'):
    """A weighted version of crossentropy loss
    weights: numpy array of shape np.array([C1, C2, C3, ... , Cn]) where n is the number of classes and Cx represents
    the weight multiplier for class x"""
    def fn(y_true, y_pred):
        nclass = weights.shape[0]
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())  # clip to prevent inf
        if binary:  # BINARY --> predictions are the probability between [0, 1] to be 1
            weights_bin = np.sqrt(weights)
            w_pu = (nclass * weights_bin) / np.sum(weights_bin)  # scale weights to sum altogether equal to nclasses
            weights_inv = 1 / weights_bin  # calculate inverse weights to apply to 0
            w_pu_inv = (nclass * weights_inv) / np.sum(weights_inv)  # scale weights to sum altogether equal to nclasses
            return -K.sum((y_true * K.log(y_pred)) * w_pu + ((1 - y_true) * K.log(1 - y_pred)) * w_pu_inv, axis=-1)
        else:  # CATEGORICAL --> predictions are a probability distribution among all classes summing in total 1
            w_pu = (nclass * weights) / np.sum(weights)  # scale weights to sum altogether equal to nclasses
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            return -K.sum(y_true * K.log(y_pred) * w_pu, axis=-1)
    fn.__name__ = name
    return fn


class MacroTPR(tf.keras.metrics.Metric):
    def __init__(self, name='macro_tpr', in_classes=2, multilabel=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.nclasses = in_classes
        self.multilabel = multilabel
        self.tpr = self.add_weight(shape=self.nclasses, name='macro_tpr', initializer='zeros')
        self.tpr_tot = self.add_weight(shape=self.nclasses, name='macro_tpr_tot', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.multilabel:
            y_pred = K.round(y_pred)
        else:
            y_pred /= K.max(y_pred, axis=-1, keepdims=True)  # assign 1 to the predicted class
        ax = [i for i in range(len(K.int_shape(y_pred)) - 1)]
        self.tpr.assign_add(K.sum(tf.cast(tf.logical_and(K.equal(y_true, 1), K.equal(y_pred, 1)), 'float32'), axis=ax))
        self.tpr_tot.assign_add(K.epsilon() + K.sum(tf.cast(K.equal(y_true, 1), 'float32'), axis=ax))

    def result(self):
        return K.sum(self.tpr / self.tpr_tot) / self.nclasses

    def reset_state(self):
        self.tpr.assign(np.zeros(self.nclasses))
        self.tpr_tot.assign(np.zeros(self.nclasses))


class TotalTPR(tf.keras.metrics.Metric):
    def __init__(self, name='total_tpr', multilabel=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.multilabel = multilabel
        self.tpr = self.add_weight(name='total_tpr', initializer='zeros')
        self.tpr_tot = self.add_weight(name='total_tpr_tot', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.multilabel:
            y_pred = K.round(y_pred)
        else:
            y_pred /= K.max(y_pred, axis=-1, keepdims=True)  # assign 1 to the predicted class
        self.tpr.assign_add(K.sum(tf.cast(tf.logical_and(K.equal(y_true, 1), K.equal(y_pred, 1)), 'float32')))
        self.tpr_tot.assign_add(K.epsilon() + K.sum(tf.cast(K.equal(y_true, 1), 'float32')))

    def result(self):
        return self.tpr / self.tpr_tot

    def reset_state(self):
        self.tpr.assign(0)
        self.tpr_tot.assign(0)


class TotalPrecision(tf.keras.metrics.Metric):
    def __init__(self, name='total_tpr', multilabel=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.multilabel = multilabel
        self.prec = self.add_weight(name='total_prec', initializer='zeros')
        self.prec_tot = self.add_weight(name='total_prec_tot', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.multilabel:
            y_pred = K.round(y_pred)
        else:
            y_pred /= K.max(y_pred, axis=-1, keepdims=True)  # assign 1 to the predicted class
        self.prec.assign_add(K.sum(tf.cast(tf.logical_and(K.equal(y_true, 1), K.equal(y_pred, 1)), 'float32')))
        self.prec_tot.assign_add(K.epsilon() + K.sum(tf.cast(K.equal(y_pred, 1), 'float32')))

    def result(self):
        return self.prec / self.prec_tot

    def reset_state(self):
        self.prec.assign(0)
        self.prec_tot.assign(0)


class MacroPrecision(tf.keras.metrics.Metric):
    def __init__(self, name='macro_precision', in_classes=2, multilabel=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.nclasses = in_classes
        self.multilabel = multilabel
        self.prec = self.add_weight(shape=self.nclasses, name='macro_prec', initializer='zeros')
        self.prec_tot = self.add_weight(shape=self.nclasses, name='macro_prec_tot', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.multilabel:
            y_pred = K.round(y_pred)
        else:
            y_pred /= K.max(y_pred, axis=-1, keepdims=True)  # assign 1 to the predicted class
        ax = [i for i in range(len(K.int_shape(y_pred)) - 1)]
        self.prec.assign_add(K.sum(tf.cast(tf.logical_and(K.equal(y_true, 1), K.equal(y_pred, 1)), 'float32'), axis=ax))
        self.prec_tot.assign_add(K.epsilon() + K.sum(tf.cast(K.equal(y_pred, 1), 'float32'), axis=ax))

    def result(self):
        return K.sum(self.prec / self.prec_tot) / self.nclasses

    def reset_state(self):
        self.prec.assign(np.zeros(self.nclasses))
        self.prec_tot.assign(np.zeros(self.nclasses))


class MacroF1(tf.keras.metrics.Metric):
    def __init__(self, name='macro_f1', in_classes=2, multilabel=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.nclasses = in_classes
        self.multilabel = multilabel
        self.preds = self.add_weight(shape=self.nclasses, name='predictions', initializer='zeros')
        self.trues = self.add_weight(shape=self.nclasses, name='true_values', initializer='zeros')
        self.tp = self.add_weight(shape=self.nclasses, name='true_positives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.multilabel:
            y_pred = K.round(y_pred)
        else:
            y_pred /= K.max(y_pred, axis=-1, keepdims=True)  # assign 1 to the predicted class
        ax = [i for i in range(len(K.int_shape(y_pred)) - 1)]
        self.preds.assign_add(K.epsilon() + K.sum(tf.cast(K.equal(y_pred, 1), 'float32'), axis=ax))
        self.trues.assign_add(K.epsilon() + K.sum(tf.cast(K.equal(y_true, 1), 'float32'), axis=ax))
        self.tp.assign_add(K.sum(tf.cast(tf.logical_and(K.equal(y_true, 1), K.equal(y_pred, 1)), 'float32'), axis=ax))

    def result(self):
        return K.sum(2 / ((1 / (self.tp / self.preds)) + (1 / (self.tp / self.trues)))) / self.nclasses

    def reset_state(self):
        self.preds.assign(np.zeros(self.nclasses))
        self.trues.assign(np.zeros(self.nclasses))
        self.tp.assign(np.zeros(self.nclasses))


class MultilabelTotalPrecision(tf.keras.metrics.Metric):
    def __init__(self, name='multilabel_total_prec', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.th = threshold
        self.prec = self.add_weight(name='multilabel_total_prec', initializer='zeros')
        self.prec_tot = self.add_weight(name='multilabel_total_prec_tot', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_max = K.max(y_pred, axis=-1, keepdims=True)
        y_pred = tf.where(y_pred > self.th, y_pred / y_pred_max, 0)
        self.prec.assign_add(K.sum(K.clip(K.sum(tf.cast(tf.logical_and(
            K.equal(y_true, 1), K.equal(y_pred, 1)), 'float32'), axis=-1), 0, 1)))
        self.prec_tot.assign_add(K.epsilon() + K.sum(K.clip(K.sum(y_pred, axis=-1), 0, 1)))

    def result(self):
        return self.prec / self.prec_tot

    def reset_state(self):
        self.prec.assign(0)
        self.prec_tot.assign(0)


class MultilabelTotalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='multilabel_total_acc', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.th = threshold
        self.acc = self.add_weight(name='multilabel_total_acc', initializer='zeros')
        self.acc_tot = self.add_weight(name='multilabel_total_acc_tot', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_max = K.max(y_pred, axis=-1, keepdims=True)
        y_pred = tf.where(y_pred > self.th, y_pred / y_pred_max, 0)
        self.acc.assign_add(K.sum(K.clip(K.sum(tf.cast(tf.logical_and(
            K.equal(y_true, 1), K.equal(y_pred, 1)), 'float32'), axis=-1), 0, 1)))
        self.acc_tot.assign_add(K.epsilon() + K.sum(K.clip(K.sum(y_true, axis=-1), 0, 1)))

    def result(self):
        return self.acc / self.acc_tot

    def reset_state(self):
        self.acc.assign(0)
        self.acc_tot.assign(0)


class PredictionRatio(tf.keras.metrics.Metric):
    def __init__(self, name='pred_ratio', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.th = threshold
        self.pred = self.add_weight(name='pred_ratio', initializer='zeros')
        self.pred_tot = self.add_weight(name='pred_ratio_tot', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_max = K.max(y_pred, axis=-1, keepdims=True)
        y_pred = tf.where(y_pred > self.th, y_pred / y_pred_max, 0)
        self.pred.assign_add(K.sum(K.clip(K.sum(tf.cast(K.equal(y_pred, 1), 'float32'), axis=-1), 0, 1)))
        self.pred_tot.assign_add(K.epsilon() + K.sum(K.clip(K.sum(y_true, axis=-1), 0, 1)))

    def result(self):
        return self.pred / self.pred_tot

    def reset_state(self):
        self.pred.assign(0)
        self.pred_tot.assign(0)

