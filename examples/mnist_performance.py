from contrastive.auto_cpca import AutoCPCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X, y = X[:1000], y[:1000].astype(int)
# CPCA with background
#X, y = X[y<4], y[y<4]
y_binary = (y == 5).astype(int)

from photonai.base import Hyperpipe, PipelineElement, Switch

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('basic_svm_pipe',
                    optimizer='grid_search',
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],
                    best_config_metric='balanced_accuracy',
                    outer_cv=StratifiedShuffleSplit(n_splits=3, test_size=0.15),
                    inner_cv=StratifiedShuffleSplit(n_splits=5, test_size=0.15),
                    verbosity=1,
                    project_folder='./tmp/')

switch = Switch("CA-switch")
switch += PipelineElement.create('CPCA', base_element=AutoCPCA(n_components=10, alpha=1),
                                  hyperparameters={})
switch += PipelineElement('PCA', n_components=10)
my_pipe += switch

my_pipe += PipelineElement('LinearSVC')  # SVC not working

my_pipe.fit(X, y_binary)