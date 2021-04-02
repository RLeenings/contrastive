import numpy as np
import matplotlib.pyplot as plt
from contrastive import CPCA

data = np.genfromtxt('datasets/mHealth_subject1.log',
                     delimiter='\t', usecols=range(0, 23), filling_values=0)
classes = np.genfromtxt('datasets/mHealth_subject1.log',
                        delimiter='\t', usecols=range(23, 24), filling_values=0)

target_idx_A = np.where(classes == 8)[0]  # jogging
target_idx_B = np.where(classes == 9)[0]  # squatting

labels = len(target_idx_A)*[0] + len(target_idx_B)*[1]
target_idx = np.concatenate((target_idx_A, target_idx_B))
target = data[target_idx]

background_idx = np.where(classes == 3)[0]  # lying still
background = data[background_idx]

mdl = CPCA(n_components=2, max_log_alpha=5, standardize=False)
projected_data = mdl.fit_transform(target, background, labels)
mdl.plot(target, labels=labels)

debug = True