from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
from sklearn.decomposition import PCA


class AutoCPCA(TransformerMixin, BaseEstimator):
    """
    Automatic Contrastive PCA (AutoCPCA)

    """

    def __init__(self, n_components: int = 2, alpha: float = 1, bg_strategy="both", preprocess_with_pca_dim: int = 1000):
        """
        preprocess_with_pca_dim: int
            If this parameter is provided (and it is greater than n_features), then both the foreground and background
            datasets undergo a preliminary round of PCA to reduce their dimension to this number. If it is not provided
            but n_features > 1,000, a preliminary round of PCA is automatically performed to reduce the
             dimensionality to 1,000.
        """
        if bg_strategy in ["both", "zero", "one"]:
            self.bg_strategy = bg_strategy
        else:
            raise ValueError("Please use one of ['both', 'zero', 'one'] for bg_strategy.")
        self.n_components = n_components
        self.preprocess_with_pca_dim = preprocess_with_pca_dim
        self.verbose = True
        self.alpha = alpha

        # Housekeeping
        self.fg_cov = None
        self.bg_cov0, self.bg_cov1 = None, None
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

        fg, bg0, bg1 = self.divide(X, y)  # backgrounds: list of n_classes

        if fg.shape[1] > self.preprocess_with_pca_dim:
            self.pca = PCA(n_components=np.min([len(fg), self.preprocess_with_pca_dim]))
            fg = self.pca.fit_transform(fg)
            bg0 = self.pca.transform(bg0)
            bg1 = self.pca.transform(bg1)

            if self.verbose:
                print("Data dimensionality reduced to " + str(self.preprocess_with_pca_dim) +
                      ". Percent variation retained: ~" + str(
                    int(100 * np.sum(self.pca.explained_variance_ratio_))) + '%')

        # Calculate the covariance matrices
        self.bg_cov0 = np.cov(bg0.T)
        self.bg_cov1 = np.cov(bg1.T)
        self.fg_cov = np.cov(fg.T)

        self.fg_cov = (self.fg_cov - self.fg_cov.mean(axis=0))
        self.bg_cov0 = (self.bg_cov0 - self.bg_cov0.mean(axis=0))
        self.bg_cov1 = (self.bg_cov1 - self.bg_cov1.mean(axis=0))

        self.cpca_alpha()

        return self

    def transform(self, X, y=None):
        # transform - feature selection via PCA for many features
        if self.pca is not None and X.shape[1] > self.preprocess_with_pca_dim:
            X = self.pca.transform(X)
        reduced_dataset = X.dot(self.v_top)

        return reduced_dataset

    def inverse_transform(self, X, y=None):
        X = X.dot(self.v_top.T)
        if self.pca is not None:
            X = self.pca.inverse_transform(X)
        return X

    def cpca_alpha(self):
        """
            Returns active and bg dataset projected in the cpca direction,
            as well as the top c_cpca eigenvalues indices.
            If specified, it returns the top_cpca directions.
        """
        # multiple alpha (alpha_0, alpha_1) should be considered here
        if self.bg_strategy == "both":
            alpha0, alpha1 = self.alpha, self.alpha
        elif self.bg_strategy == "zero":
            alpha0, alpha1 = self.alpha, 0
        else:
            alpha0, alpha1 = 0, self.alpha
        sigma = self.fg_cov - alpha0 * self.bg_cov0 - alpha1 * self.bg_cov1
        w, v = np.linalg.eigh(sigma)
        eig_idx = np.argpartition(w, -self.n_components)[-self.n_components:]
        eig_idx = eig_idx[np.argsort(-w[eig_idx])]
        self.v_top = np.real(v[:, eig_idx])  # sigma is quasi symm. -> imaginary part only numerical reasons

    def divide(self, X, y):
        bg0 = X[y == 0]  # background 1
        bg1 = X[y == 1]  # background 2
        return X, bg0, bg1
