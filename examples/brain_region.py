import numpy as np
from nilearn.datasets import fetch_oasis_vbm
from photonai_neuro import BrainMask

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from contrastive.auto_cpca import AutoCPCA

dataset_files = fetch_oasis_vbm()
gray_matter_map_filenames = dataset_files.gray_matter_maps
gender = dataset_files.ext_vars['mf'].astype(str)
y_gender = np.array(gender)
y_gender = (y_gender == 'M').astype(int)
y_age = dataset_files.ext_vars["age"].astype(int)
y_age = (y_age > 50).astype(int)
X = np.array(dataset_files.gray_matter_maps)

mask = BrainMask(mask_image='MNI_ICBM152_GrayMatter', extract_mode='vec')
X = mask.transform(X)
print("BrainMask - done.")
print(X.shape)

# comparison between both
auto_cpca = AutoCPCA(n_components=10, alpha=2, bg_strategy="both")
pca = PCA(n_components=10)


from sklearn.svm import LinearSVC

y = y_gender
# build fg and bg only on train_set
a = StratifiedShuffleSplit(1, test_size=0.2)
svc_autocpca = LinearSVC()
svc_pca = LinearSVC()
for train_index, test_index in a.split(X, y):
    pass

X_train = X[train_index]

X_test = X[test_index]
Xtrain_autoCPCA = auto_cpca.fit_transform(X_train, y[train_index])
Xtrain_pca = pca.fit_transform(X_train, y[train_index])
svc_autocpca.fit(Xtrain_autoCPCA, y[train_index])
svc_pca.fit(Xtrain_pca, y[train_index])

trans_X = auto_cpca.transform(X_test)
y_pred_autocpca = svc_autocpca.predict(trans_X)
trans_X = pca.transform(X_test)
y_pred_pca = svc_pca.predict(trans_X)

from sklearn.metrics import confusion_matrix, balanced_accuracy_score
print("BAC-AutoPCA :"+ str(balanced_accuracy_score(y[test_index], y_pred_autocpca)))
print("BAC-PCA :"+ str(balanced_accuracy_score(y[test_index], y_pred_pca)))
print("MAT-AutoPCA :"+ str(confusion_matrix(y[test_index], y_pred_autocpca)))
print("MAT-PCA :"+ str(confusion_matrix(y[test_index], y_pred_pca)))

coef = svc_autocpca.coef_
inverse_cpca = auto_cpca.inverse_transform(coef)
inverse_cpca[inverse_cpca<np.max(inverse_cpca)*0.8] = 0
coef = svc_pca.coef_
inverse_pca = pca.inverse_transform(coef)
inverse_pca[inverse_pca<np.max(inverse_pca)*0.8] = 0

# Create the figure
from nilearn.plotting import plot_stat_map, show
bg_filename = gray_matter_map_filenames[0]
display = plot_stat_map(mask.inverse_transform(inverse_cpca), bg_img=bg_filename,
                        display_mode='z', cut_coords=[0, 15])
display.title("AutoCPCA weights")
show()

bg_filename = gray_matter_map_filenames[0]
display = plot_stat_map(mask.inverse_transform(inverse_pca), bg_img=bg_filename,
                        display_mode='z', cut_coords=[0, 15])
display.title("PCA weights")
show()


coef = svc_autocpca.coef_
inverse_cpca = auto_cpca.inverse_transform(coef)
coef = svc_pca.coef_
inverse_pca = pca.inverse_transform(coef)

bg_filename = gray_matter_map_filenames[0]
a = np.abs(inverse_pca-inverse_cpca)
a[a<np.max(a)*0.5]=0
display = plot_stat_map(mask.inverse_transform(a), bg_img=bg_filename)
display.title("Diff weights")
show()
