import matplotlib.pyplot as plt
from contrastive.auto_cpca import AutoCPCA
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import fetch_openml

# some data preprocessing
X, y = load_digits(return_X_y=True, n_class=5)
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X, y = X[:5000], y[:5000].astype(int)
# CPCA with background
y_binary = (y == 2).astype(int)

# comparison between both
auto_cpca = AutoCPCA(n_components=2, alpha=1.5, bg_strategy="zero")
normal_pca = PCA(n_components=2)

# build fg and bg only on train_set
a = StratifiedShuffleSplit(1, test_size=0.1)
for train_index, test_index in a.split(X, y):
    auto_cpca.fit(X[train_index], y_binary[train_index])
    normal_pca.fit(X[train_index], y_binary[train_index])

    trans_normal_pca_X = normal_pca.transform(X[test_index])
    trans_X = auto_cpca.transform(X[test_index])


plt.subplot(2, 2, 1)
plt.scatter(trans_normal_pca_X[:, 0], trans_normal_pca_X[:, 1], c=y[test_index], cmap=plt.cm.coolwarm)
plt.title('PCA all classes')

plt.subplot(2, 2, 2)
plt.scatter(trans_normal_pca_X[:, 0], trans_normal_pca_X[:, 1], c=y_binary[test_index], cmap=plt.cm.coolwarm)
plt.title('PCA binary classes')

plt.subplot(2, 2, 3)
plt.scatter(trans_X[:, 0], trans_X[:, 1], c=y[test_index], cmap=plt.cm.coolwarm)
plt.title('CPCA all classes')

plt.subplot(2, 2, 4)
plt.scatter(trans_X[:, 0], trans_X[:, 1], c=y_binary[test_index], cmap=plt.cm.coolwarm)
plt.title('CPCA binary classes')

plt.show()