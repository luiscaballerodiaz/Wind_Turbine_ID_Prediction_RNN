import keras.backend as K
import tensorflow as tf


def weighted_categorical_crossentropy(weights, name='weighted_loss'):
    """A weighted version of keras.objectives.categorical_crossentropy
    weights: numpy array of shape np.array([C1, C2, C3, ... , Cn]) where n is the number of classes and Cx represents
    the weight multiplier for class x"""
    def fn(y_true, y_pred):
        pu_weights = (weights.shape[0] * weights) / sum(weights)  # scale weights to sum altogether equal to nclasses
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)  # sanity check to scale the sum to 1
        y_pred = K.clip(y_pred, K.epsilon(), 1)  # clip to prevent inf
        return -K.sum(y_true * K.log(y_pred) * pu_weights, axis=-1)
    fn.__name__ = name
    return fn


class MacroTPR(tf.keras.metrics.Metric):
    def __init__(self, name='macro_tpr', in_classes=2, **kwargs):
        super().__init__(name=name, **kwargs)
        self.nclasses = in_classes
        self.tpr = []
        self.tpr_tot = []
        for c in range(self.nclasses):
            self.tpr.append(self.add_weight(name='macro_tpr' + str(c), initializer='zeros'))
            self.tpr_tot.append(self.add_weight(name='macro_tpr' + str(c) + '_tot', initializer='zeros'))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.argmax(y_true, axis=-1)
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)  # sanity check to scale the sum to 1
        y_pred = K.argmax(y_pred, axis=-1)  # select position with max value as prediction
        for c in range(self.nclasses):
            self.tpr_tot[c].assign_add(K.epsilon() + tf.reduce_sum(tf.cast(K.equal(y_true, c), 'float32')))
            self.tpr[c].assign_add(tf.reduce_sum(tf.cast(tf.logical_and(K.equal(y_true, c), K.equal(y_pred, c)), 'float32')))

    def result(self):
        output = 0.
        for c in range(self.nclasses):
            output += self.tpr[c] / self.tpr_tot[c]
        return output / self.nclasses

    def reset_state(self):
        for c in range(self.nclasses):
            self.tpr[c].assign(0)
            self.tpr_tot[c].assign(0)


class TotalTPR(tf.keras.metrics.Metric):
    def __init__(self, name='total_tpr', in_classes=2, **kwargs):
        super().__init__(name=name, **kwargs)
        self.nclasses = in_classes
        self.tpr = self.add_weight(name='total_tpr', initializer='zeros')
        self.tpr_tot = self.add_weight(name='total_tpr_tot', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.argmax(y_true, axis=-1)
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)  # sanity check to scale the sum to 1
        y_pred = K.argmax(y_pred, axis=-1)  # select position with max value as prediction
        for c in range(self.nclasses):
            self.tpr_tot.assign_add(K.epsilon() + tf.reduce_sum(tf.cast(K.equal(y_true, c), 'float32')))
            self.tpr.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(K.equal(y_true, c), K.equal(y_pred, c)), 'float32')))

    def result(self):
        return self.tpr / self.tpr_tot

    def reset_state(self):
        self.tpr.assign(0)
        self.tpr_tot.assign(0)


class MacroPrecision(tf.keras.metrics.Metric):
    def __init__(self, name='macro_precision', in_classes=2, **kwargs):
        super().__init__(name=name, **kwargs)
        self.nclasses = in_classes
        self.prec = []
        self.prec_tot = []
        for c in range(self.nclasses):
            self.prec.append(self.add_weight(name='macro_precision' + str(c), initializer='zeros'))
            self.prec_tot.append(self.add_weight(name='macro_precision' + str(c) + '_tot', initializer='zeros'))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.argmax(y_true, axis=-1)
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)  # sanity check to scale the sum to 1
        y_pred = K.argmax(y_pred, axis=-1)  # select position with max value as prediction
        for c in range(self.nclasses):
            self.prec_tot[c].assign_add(K.epsilon() + tf.reduce_sum(tf.cast(K.equal(y_pred, c), 'float32')))
            self.prec[c].assign_add(tf.reduce_sum(tf.cast(tf.logical_and(K.equal(y_true, c), K.equal(y_pred, c)), 'float32')))

    def result(self):
        output = 0.
        for c in range(self.nclasses):
            output += self.prec[c] / self.prec_tot[c]
        return output / self.nclasses

    def reset_state(self):
        for c in range(self.nclasses):
            self.prec[c].assign(0)
            self.prec_tot[c].assign(0)


class MacroF1(tf.keras.metrics.Metric):
    def __init__(self, name='macro_f1', in_classes=2, **kwargs):
        super().__init__(name=name, **kwargs)
        self.nclasses = in_classes
        self.preds = []
        self.trues = []
        self.tp = []
        for c in range(self.nclasses):
            self.preds.append(self.add_weight(name='predicitons' + str(c), initializer='zeros'))
            self.trues.append(self.add_weight(name='true_values' + str(c), initializer='zeros'))
            self.tp.append(self.add_weight(name='true_positives' + str(c), initializer='zeros'))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.argmax(y_true, axis=-1)
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)  # sanity check to scale the sum to 1
        y_pred = K.argmax(y_pred, axis=-1)  # select position with max value as prediction
        for c in range(self.nclasses):
            self.preds[c].assign_add(K.epsilon() + tf.reduce_sum(tf.cast(K.equal(y_pred, c), 'float32')))
            self.trues[c].assign_add(K.epsilon() + tf.reduce_sum(tf.cast(K.equal(y_true, c), 'float32')))
            self.tp[c].assign_add(tf.reduce_sum(tf.cast(tf.logical_and(K.equal(y_true, c), K.equal(y_pred, c)), 'float32')))

    def result(self):
        output = 0.
        for c in range(self.nclasses):
            prec = self.tp[c] / self.preds[c]
            tpr = self.tp[c] / self.trues[c]
            output += 2 / ((1 / prec) + (1 / tpr))
        return output / self.nclasses

    def reset_state(self):
        for c in range(self.nclasses):
            self.preds[c].assign(0)
            self.trues[c].assign(0)
            self.tp[c].assign(0)
