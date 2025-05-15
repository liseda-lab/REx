from __future__ import division
from __future__ import absolute_import
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  


class baseline(object):
    def get_baseline_value(self):
        pass
    def update(self, target):
        pass

class ReactiveBaseline(baseline):
    def __init__(self, l):
        self.l = l
        self.b = tf.Variable( 0.0, trainable=False)
    def get_baseline_value(self):
        # returns the current baseline value
        return self.b
    def update(self, target):
        # updates the baseline value based on the target value reward
        self.b = tf.add((1-self.l)*self.b, self.l*target)