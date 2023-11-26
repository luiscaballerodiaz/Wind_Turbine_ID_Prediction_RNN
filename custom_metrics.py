import keras.backend as K
import tensorflow as tf


def unbalanced_loss(weights, name='custom_metric'):
    def fn(y_true, y_pred):
        return tf.reduce_sum(tf.multiply(K.exp(K.abs(y_true - y_pred)), weights))
    fn.__name__ = name
    return fn


class WeightTPR(tf.keras.metrics.Metric):

    def __init__(self, name='weight_tpr', in_classes=2, **kwargs):
        super().__init__(name=name, **kwargs)
        self.nclasses = in_classes
        self.tpr = self.add_weight(name='weight_tpr', initializer='zeros')
        self.tpr_tot = self.add_weight(name='weight_tpr_tot', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.argmax(y_true, axis=1)
        y_pred = K.argmax(y_pred, axis=1)
        for c in range(self.nclasses):
            self.tpr_tot.assign_add(tf.reduce_sum(tf.cast(K.equal(y_true, c), 'float32')))
            self.tpr.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(K.equal(y_true, c), K.equal(y_pred, c)), 'float32')))

    def result(self):
        return self.tpr / self.tpr_tot

    def reset_state(self):
        self.tpr.assign(0)
        self.tpr_tot.assign(0)


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
        y_true = K.argmax(y_true, axis=1)
        y_pred = K.argmax(y_pred, axis=1)
        for c in range(self.nclasses):
            self.tpr_tot[c].assign_add(tf.reduce_sum(tf.cast(K.equal(y_true, c), 'float32')))
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
