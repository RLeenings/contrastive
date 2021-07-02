import numpy as np
from nilearn.datasets import fetch_oasis_vbm
from photonai_neuro import BrainMask

from contrastive.auto_cpca import AutoCPCA

dataset_files = fetch_oasis_vbm()
gender = dataset_files.ext_vars['mf'].astype(str)
y_gender = np.array(gender)
y_gender = (y_gender == 'M').astype(int)
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
#auto_cpca = AutoCPCA(n_components=2, alpha=6, bg_strategy="both")

y = y_gender
from sklearn.model_selection import StratifiedShuffleSplit

from photonai.base import Hyperpipe, PipelineElement, Switch

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('basic_svm_pipe',
                    optimizer='grid_search',
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],
                    best_config_metric='balanced_accuracy',
                    outer_cv=StratifiedShuffleSplit(n_splits=3, test_size=0.2),
                    inner_cv=StratifiedShuffleSplit(n_splits=3, test_size=0.2),
                    verbosity=1,
                    project_folder='./tmp/')

switch = Switch("CA-switch")
switch += PipelineElement.create('CPCA', base_element=AutoCPCA(n_components=20, alpha=7,
                                                               preprocess_with_pca_dim=1000),
                                  hyperparameters={})
switch += PipelineElement('PCA', n_components=20)
my_pipe += switch


my_pipe += PipelineElement('LinearSVC')


my_pipe.fit(X, y)