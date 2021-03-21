from __future__ import print_function
import numpy as np
from numpy import linalg as LA
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA


class AutoCPCA(TransformerMixin):
    """
    Automatic Contrastive PCA (AutoCPCA)

    """
    def __init__(self, n_components: int = 2, n_alphas=40, preprocess_with_pca_dim: int = 1000, max_log_alpha=3):
        """
        preprocess_with_pca_dim: int
            If this parameter is provided (and it is greater than n_features), then both the foreground and background
            datasets undergo a preliminary round of PCA to reduce their dimension to this number. If it is not provided
            but n_features > 1,000, a preliminary round of PCA is automatically performed to reduce the
             dimensionality to 1,000.
        """
        self.n_alphas = n_alphas
        self.max_log_alpha = max_log_alpha
        self.n_components = n_components
        self.preprocess_with_pca_dim = preprocess_with_pca_dim
        self.verbose = True

        # Housekeeping
        self.fg_cov = None
        self.bg_cov = None
        self.v_top = None
        self.pca = None

    def fit_transform(self, X, y, **kwargs):
        """
            Finds the covariance matrices of the foreground and background datasets,
            and then transforms the foreground dataset based on the principal contrastive components

            Parameters: see self.fit() and self.transform() for parameter description
        """
        self.fit(X, y)
        return self.transform(X, y)

    def fit(self, X: np.ndarray, y: np.ndarray):

        # check y -> classification
        if len(set(list(y))) != 2:
            raise AttributeError("At this point, AutoCPCA is only available for "
                                 "binary classification. Concepts on multi_class data may follow later.")

        fg, bg, = self.divide(X, y)  # backgrounds: list of n_classes

        fg = StandardScaler().fit_transform(fg)
        bg = StandardScaler().fit_transform(bg)


        if fg.shape[1] > self.preprocess_with_pca_dim:
            data = np.concatenate((fg, bg), axis=0)
            self.pca = PCA(n_components=np.min([len(data), self.preprocess_with_pca_dim]))
            data = self.pca.fit_transform(data)
            fg = data[:fg.shape[0], :]
            bg = data[fg.shape[0]:, :]
            features_d = self.preprocess_with_pca_dim

            if self.verbose:
                print("Data dimensionality reduced to " + str(self.preprocess_with_pca_dim) +
                      ". Percent variation retained: ~" + str(int(100*np.sum(self.pca.explained_variance_ratio_)))+'%')

        # Calculate the covariance matrices
        self.bg_cov = bg.T.dot(bg) / (bg.shape[0] - 1)
        self.fg_cov = fg.T.dot(fg) / (fg.shape[0] - 1)

        best_alphas, all_alphas, _, _ = self.find_spectral_alphas(fg)
        best_alpha = best_alphas[0]
        self.cpca_alpha(alpha=best_alpha)
        return self

    def transform(self, X, y=None):
        # transform - feature selection via PCA for many features
        if self.pca is not None and X.shape[1] > self.preprocess_with_pca_dim:
            X = self.pca.transform(X)
        reduced_dataset = X.dot(self.v_top)
        sign_vector = np.sign(reduced_dataset[0, :])
        reduced_dataset = reduced_dataset * sign_vector

        # still dependent on n_components = 2
        reduced_dataset[:, 0] = reduced_dataset[:, 0] * np.sign(reduced_dataset[0, 0])
        reduced_dataset[:, 1] = reduced_dataset[:, 1] * np.sign(reduced_dataset[0, 1])
        return reduced_dataset

    def cpca_alpha(self, alpha=0.1):
        """
            Returns active and bg dataset projected in the cpca direction,
            as well as the top c_cpca eigenvalues indices.
            If specified, it returns the top_cpca directions.
        """
        sigma = self.fg_cov - alpha * self.bg_cov
        w, v = LA.eig(sigma)
        eig_idx = np.argpartition(w, -self.n_components)[-self.n_components:]
        eig_idx = eig_idx[np.argsort(-w[eig_idx])]
        self.v_top = v[:, eig_idx]

    def divide(self, X, y):
        a = StratifiedShuffleSplit(1, test_size=0.3)
        for bg_index, fg_index in a.split(X, y):
            pass
        bg_data, fg_data = X[bg_index], X[fg_index]
        bg_y, fg_y = y[bg_index], y[fg_index]
        bg = bg_data[bg_y==0]
        return fg_data,  bg

    def find_spectral_alphas(self, X):
        """
            This method performs spectral clustering on the affinity matrix of subspaces
            returned by contrastive pca, and returns (`=3) exemplar values of alpha.
        """
        self.affinity_matrix = self.create_affinity_matrix(X)
        alphas = self.generate_alphas()
        spectral = cluster.SpectralClustering(n_clusters=4, affinity='precomputed')
        spectral.fit(self.affinity_matrix)
        labels = spectral.labels_

        best_alphas = list()
        for i in range(4):
            idx = np.where(labels == i)[0]
            if not (0 in idx):  # because we don't want to include the cluster that includes alpha=0
                affinity_submatrix = self.affinity_matrix[idx][:, idx]
                sum_affinities = np.sum(affinity_submatrix, axis=0)
                exemplar_idx = idx[np.argmax(sum_affinities)]
                best_alphas.append(alphas[exemplar_idx])
        return np.sort(best_alphas), alphas, self.affinity_matrix[0, :], labels

    def generate_alphas(self):
        return np.concatenate(([0], np.logspace(-1, self.max_log_alpha, self.n_alphas)))


    def create_affinity_matrix(self, X):
        """
            This method creates the affinity matrix of subspaces returned by contrastive PCA.
        """
        subspaces = list()
        alphas = self.generate_alphas()
        k = len(alphas)
        affinity = 0.5 * np.identity(k)  # it gets doubled
        for alpha in alphas:
            self.cpca_alpha(alpha=alpha)
            q, r = np.linalg.qr(self.transform(X))
            subspaces.append(q)
        for i in range(k):
            for j in range(i + 1, k):
                v0 = subspaces[i]
                v1 = subspaces[j]
                u, s, v = np.linalg.svd(v0.T.dot(v1))
                affinity[i, j] = np.prod([np.cos(eigendings) for eigendings in s])
                affinity[i, j] = s[0] * s[1]
        affinity = affinity + affinity.T
        return np.nan_to_num(affinity)
