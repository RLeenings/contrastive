import numpy as np
from nilearn.datasets import fetch_oasis_vbm
from photonai_neuro import BrainMask

import matplotlib.pyplot as plt
from contrastive.auto_cpca import AutoCPCA
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit

# GET DATA FROM OASIS
n_subjects = 400
dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
gender = dataset_files.ext_vars['mf'].astype(str)
y = np.array(gender)
y = (y == 'M').astype(int)
X = np.array(dataset_files.gray_matter_maps)

from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import StandardScaler

mask = BrainMask(mask_image='MNI_ICBM152_GrayMatter', extract_mode='vec')
X = mask.transform(X)
print("BrainMask - done.")
X = StandardScaler().fit_transform(X, y)
#X = SelectPercentile(percentile=0.1).fit_transform(X, y)
print(X.shape)

# comparison between both
auto_cpca = AutoCPCA(n_components=2, max_log_alpha=3)
normal_pca = PCA(n_components=2)

# build fg and bg only on train_set
a = StratifiedShuffleSplit(1, test_size=0.2)
for train_index, test_index in a.split(X, y):
    auto_cpca.fit(X[train_index], y[train_index])
    normal_pca.fit(X[train_index],y[train_index])

    trans_normal_pca_X = normal_pca.transform(X[test_index])
    trans_X = auto_cpca.transform(X[test_index])


plt.subplot(2, 1, 1)
plt.scatter(trans_normal_pca_X[:, 0], trans_normal_pca_X[:, 1], c=y[test_index], cmap=plt.cm.coolwarm)
plt.title('PCA')

plt.subplot(2, 1, 2)
plt.scatter(trans_X[:, 0], trans_X[:, 1], c=y[test_index], cmap=plt.cm.coolwarm)
plt.title('CPCA')

plt.show()