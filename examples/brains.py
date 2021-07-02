import numpy as np
from nilearn.datasets import fetch_oasis_vbm
from photonai_neuro import BrainMask

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from contrastive.auto_cpca import AutoCPCA

dataset_files = fetch_oasis_vbm()
gender = dataset_files.ext_vars['mf'].astype(str)
y_gender = np.array(gender)
y_gender = (y_gender == 'M').astype(int)
y_age = dataset_files.ext_vars["age"].astype(int)
y_age = (y_age > 50).astype(int)
X = np.array(dataset_files.gray_matter_maps)

from sklearn.feature_selection import SelectPercentile
# -> FeatureSelection, proof intersection of unused feature and CPCA featrue_importance
from sklearn.preprocessing import StandardScaler

mask = BrainMask(mask_image='MNI_ICBM152_GrayMatter', extract_mode='vec')
X = mask.transform(X)
print("BrainMask - done.")
#X = SelectPercentile(percentile=0.1).fit_transform(X, y)
print(X.shape)

# comparison between both
auto_cpca = AutoCPCA(n_components=2, alpha=2, bg_strategy="both")
normal_pca = PCA(n_components=2)

y = y_gender
# build fg and bg only on train_set
a = StratifiedShuffleSplit(1, test_size=0.2)
for train_index, test_index in a.split(X, y):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_index])

    X_test = scaler.transform(X[test_index])
    auto_cpca.fit(X_train, y[train_index])
    normal_pca.fit(X_train,y[train_index])

    trans_normal_pca_X = normal_pca.transform(X_test)
    trans_X = auto_cpca.transform(X_test)

plt.subplot(2, 2, 1)
plt.scatter(trans_normal_pca_X[:, 0], trans_normal_pca_X[:, 1], c=y_gender[test_index], cmap=plt.cm.coolwarm)
plt.title('PCA-Gender')

plt.subplot(2, 2, 2)
plt.scatter(trans_X[:, 0], trans_X[:, 1], c=y_gender[test_index], cmap=plt.cm.coolwarm)
plt.title('CPCA-Gender')

plt.subplot(2, 2, 3)
plt.scatter(trans_normal_pca_X[:, 0], trans_normal_pca_X[:, 1], c=y_age[test_index], cmap=plt.cm.coolwarm)
plt.title('PCA-Age')

plt.subplot(2, 2, 4)
plt.scatter(trans_X[:, 0], trans_X[:, 1], c=y_age[test_index], cmap=plt.cm.coolwarm)
plt.title('CPCA-Age')

plt.show()