try:
    import _pickle as cPickle
except:
    import cPickle

import gzip
import sys


def load_data():
    with gzip.open('C:\Users\Monitor\PycharmProjects\exam\mnist.pkl.gz', 'rb') as f:
        if sys.version_info < (3,):
            return cPickle.load(f)
        else:
            return cPickle.load(f, encoding='bytes')
