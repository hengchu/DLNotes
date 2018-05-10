# Taken from here: https://github.com/LargePanda/Information-Bottleneck-for-Deep-Learning/blob/master/mi_tool.py

import numpy as np
from collections import Counter
import math

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

class MI:
    def __init__(self, X, y, bin_size):
        self.X = X
        self.y = y
        self.bins = np.linspace(-1, 1, bin_size)
        self.n_samples = self.X.shape[0]
        self.unit = 1./self.n_samples

        self.pdf_x = Counter()
        self.pdf_y = Counter()

        self.en_mi_collector = []
        self.de_mi_collector = []
        self.epochs = []


    def discretize(self):
        self.X_d = np.digitize(self.X, self.bins).tolist()
        self.y_d = self.y.tolist()

    def pre_compute(self):
        for i in range(self.n_samples):
            self.pdf_x[totuple(self.X_d[i])] += self.unit
            self.pdf_y[totuple(self.y_d[i])] += self.unit

    def combine(self, a, b):
        ret = ( totuple(a) , totuple(b))
        return ret

    def joint_compute(self, hidden, batch):
        self.h = np.digitize(hidden, self.bins).tolist()
        self.pdf_t = Counter()
        self.pdf_xt = Counter()
        self.pdf_yt = Counter()

        for i in range(0, len(batch)):
            xt = self.combine(self.X_d[batch[i]], self.h[i])
            yt = self.combine(self.y_d[batch[i]], self.h[i])
            self.pdf_xt[xt] += self.unit
            self.pdf_yt[yt] += self.unit
            self.pdf_t[totuple(self.h[i])] += self.unit

    def encoder_mi(self):
        return sum([ self.pdf_xt[xt] * (math.log(self.pdf_xt[xt]) - math.log(self.pdf_x[xt[0]]) -  math.log(self.pdf_t[xt[1]])) for xt in self.pdf_xt ])

    def decoder_mi(self):
        return sum([ self.pdf_yt[yt] * (math.log(self.pdf_yt[yt]) - math.log(self.pdf_y[yt[0]]) -  math.log(self.pdf_t[yt[1]])) for yt in self.pdf_yt ])

    def mi_single_epoch(self, hiddens, batch, epoch):
        ens = []
        des = []
        for hidden in hiddens:
            self.joint_compute(hidden, batch)
            ens.append(self.encoder_mi())
            des.append(self.decoder_mi())
        self.en_mi_collector.append(ens)
        self.de_mi_collector.append(des)

        simple_ens = [round(a, 4) for a in ens]
        simple_des = [round(b, 4) for b in des]
        points = [(simple_ens[i], simple_des[i]) for i in range(len(simple_des))]
        # print "MI points", points

        self.epochs.append(epoch)
