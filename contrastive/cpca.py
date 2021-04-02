import numpy as np
from numpy import linalg as LA
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.base import TransformerMixin
from sklearn.metrics import silhouette_score
from sklearn.exceptions import NotFittedError
import matplotlib.pyplot as plt


class CPCA(TransformerMixin):
    """
    Contrastive PCA (cPCA)

    Linear dimensionality reduction that uses eigenvalue decomposition
    to identify directions that have increased variance in the primary (foreground)
    dataset relative to a secondary (background) dataset. Then, those directions
    are used to project the data to a lower dimensional space.
    """

    def __init__(self, n_components=2, alpha_value=None, standardize=True, preprocess_with_pca_dim: int = 1000,
                 alpha_selection='auto', n_alphas=40, max_log_alpha=3, n_alphas_to_return=4,
                 verbose=False):
        """
        preprocess_with_pca_dim: int
            If this parameter is provided (and it is greater than n_features), then both the foreground and background
            datasets undergo a preliminary round of PCA to reduce their dimension to this number. If it is not provided
            but n_features > 1,000, a preliminary round of PCA is automatically performed to reduce the
             dimensionality to 1,000.
        """

        self.alpha_value = alpha_value
        self.alpha_selection = alpha_selection
        self.n_alphas = n_alphas
        self.max_log_alpha = max_log_alpha
        self.n_alphas_to_return = n_alphas_to_return
        self.standardize = standardize
        self.n_components = n_components
        self.preprocess_with_pca_dim = preprocess_with_pca_dim
        self.verbose = verbose

        # Housekeeping
        self.fitted = False
        self.pca = None
        self.affinity_matrix = None
        self.bg_cov = None
        self.fg_cov = None

        self.v_top = None
        self.alpha_values = None

    def fit_transform(self, foreground, background, labels=None):
        """
            Finds the covariance matrices of the foreground and background datasets,
            and then transforms the foreground dataset based on the principal contrastive components

            Parameters: see self.fit() and self.transform() for parameter description
        """
        self.fit(foreground, background, labels)
        return self.transform(dataset=foreground)

# todo: rename to foreground and background
    def fit(self, fg, bg, labels=None):
        """
        Computes the covariance matrices of the foreground and background datasets

        Parameters
        -----------
        foreground: array, shape (n_data_points, n_features)
            The dataset in which the interesting directions that we would like to discover are present or enriched

        background : array, shape (n_data_points, n_features)
            The dataset in which the interesting directions that we would like to discover are absent or unenriched
        """
        # Reset
        self.fitted = False
        self.pca = None
        self.affinity_matrix = None
        self.bg_cov = None
        self.fg_cov = None
        self.v_top = None
        self.alpha_values = None

        # Datasets and dataset sizes
        n_fg, features_d = fg.shape
        n_bg, features_d_bg = bg.shape

        if features_d != features_d_bg:
            raise ValueError('The dimensionality of the foreground and background datasets must be the same')

        # center the background and foreground data
        if self.standardize:  # Standardize if specified
            bg = self._standardize(bg)
            fg = self._standardize(fg)

        bg = bg - np.mean(bg, axis=0)
        fg = fg - np.mean(fg, axis=0)

        if features_d > self.preprocess_with_pca_dim:
            data = np.concatenate((fg, bg), axis=0)
            self.pca = PCA(n_components=self.preprocess_with_pca_dim)
            data = self.pca.fit_transform(data)
            fg = data[:n_fg, :]
            bg = data[n_fg:, :]
            features_d = self.preprocess_with_pca_dim

            if self.verbose:
                print("Data dimensionality reduced to " + str(self.preprocess_with_pca_dim) +
                      ". Percent variation retained: ~" + str(int(100*np.sum(self.pca.explained_variance_ratio_)))+'%')

        if self.verbose:
            print("Data loaded and preprocessed")

        # Calculate the covariance matrices
        self.bg_cov = bg.T.dot(bg)/(bg.shape[0]-1)
        self.fg_cov = fg.T.dot(fg)/(n_fg-1)

        if self.verbose:
            print("Covariance matrices computed")

        # todo: find alpha
        if self.alpha_selection not in ['auto', 'manual', 'all']:
            raise ValueError("Invalid argument for parameter alpha_selection: must be 'auto' or 'manual' or 'all'")

        if not self.alpha_value and self.alpha_selection == 'manual':
            raise ValueError('The the alpha_selection parameter is set to "manual", '
                             'the alpha_value parameter must be provided')

        self.alpha_values = None
        if self.alpha_selection == 'auto':
            self.alpha_values = self.find_spectral_alphas(fg)  # self.automated_cpca(dataset)
            if labels is not None:
                self.alpha_value = self.select_most_discriminating_alpha(fg, labels, self.alpha_values)
            else:
                # todo: !
                self.alpha_value = self.alpha_values[-1]
        elif self.alpha_selection == 'manual':
            self.alpha_values = [self.alpha_value]
        self.v_top = self.alpha_space(self.alpha_value)
        self.fitted = True
        return self

    def transform(self, dataset, y=None, **kwargs):
        if not self.fitted:
            raise NotFittedError("This model has not been fit to a foreground/background dataset yet. "
                                 "Please run the fit() function first.")

        # todo: preprocess with pca if dimension of dataset was too big
        transformed_data = self.cpca_alpha(dataset, self.v_top, self.alpha_value)
        return transformed_data

    def _standardize(self, array):
        standardized_array = (array - np.mean(array, axis=0)) / np.std(array, axis=0)
        return np.nan_to_num(standardized_array)

    def alpha_space(self, alpha):
        # fit
        sigma = self.fg_cov - alpha * self.bg_cov
        w, v = LA.eig(sigma)
        eig_idx = np.argpartition(w, -self.n_components)[-self.n_components:]
        eig_idx = eig_idx[np.argsort(-w[eig_idx])]
        v_top = v[:, eig_idx]
        return v_top

    def cpca_alpha(self, dataset, v_top, alpha=1):
        """
            Returns active and bg dataset projected in the cpca direction, as well as the top c_cpca eigenvalues indices.
            If specified, it returns the top_cpca directions
        """

        # transform
        reduced_dataset = dataset.dot(v_top)
        sign_vector = np.sign(reduced_dataset[0, :])
        reduced_dataset = reduced_dataset * sign_vector
        return reduced_dataset

    # def reverse_transform(self, dataset):
    #     reverter=pseudo_inverse(v_top)
    #     return dataset.dot(reverter)

    def generate_alphas(self):
        return np.concatenate(([0], np.logspace(-1, self.max_log_alpha, self.n_alphas)))

    def automated_cpca(self, dataset):
        """
            This function performs contrastive PCA using the alpha technique on the
            active and background dataset. It automatically determines n_alphas=4 important values
            of alpha up to based to the power of 10^(max_log_alpha=5) on spectral clustering
            of the top subspaces identified by cPCA.
            The final return value is the data projected into the top n subspaces with (n_components = 2)
            subspaces, which can be plotted outside of this function
        """
        pass
        # best_alpha = self.find_spectral_alphas()
        # data_to_plot = []
        # for alpha in best_alphas:
        #     transformed_dataset = self.cpca_alpha(dataset=dataset, alpha=alpha)
        #     data_to_plot.append(transformed_dataset)
        # return best_alphas

    def find_spectral_alphas(self, dataset):
        """
            This method performs spectral clustering on the affinity matrix of subspaces
            returned by contrastive pca, and returns (n_alphas_to_return) exemplar values of alpha
        """
        self.affinity_matrix = self.create_affinity_matrix(dataset)
        alphas = self.generate_alphas()
        # todo: actually we dont know if we can represent similarities in n_alphas_to-return clusters?
        # what if the medium value of the cluster is in a land of nowhere in-between two bubbles?
        spectral = SpectralClustering(n_clusters=self.n_alphas_to_return, affinity='precomputed')
        spectral.fit(self.affinity_matrix)
        labels = spectral.labels_
        
        best_alphas = list()
        for i in range(self.n_alphas_to_return):
            idx = np.where(labels == i)[0]
            if not(0 in idx):  # because we don't want to include the cluster that includes alpha=0 #todo: ????
                # how do we know if the that the first item belongs to the cluster and
                # that the cluster belongs to alpha=0?
                affinity_submatrix = self.affinity_matrix[idx][:, idx]
                sum_affinities = np.sum(affinity_submatrix, axis=0)
                best_cluster_alpha = alphas[idx[np.argmax(sum_affinities)]]
                best_alphas.append(best_cluster_alpha)

        # one of the alphas is always alpha=0
        best_alphas = np.sort(np.concatenate(([0], best_alphas)))
        return best_alphas

    def create_affinity_matrix(self, dataset):
        """
            This method creates the affinity matrix of subspaces returned by contrastive pca
        """
        subspaces = list()
        alphas = self.generate_alphas()
        k = len(alphas)
        affinity = 0.5*np.identity(k)  # it gets doubled
        for alpha in alphas:
            v_top = self.alpha_space(alpha=alpha)
            space = self.cpca_alpha(dataset=dataset, v_top=v_top, alpha=alpha)
            q, r = np.linalg.qr(space)
            subspaces.append(q)
        for i in range(k):
            for j in range(i+1, k):
                v0 = subspaces[i]
                v1 = subspaces[j]
                u, s, v = np.linalg.svd(v0.T.dot(v1))
                affinity[i, j] = np.prod([eigen_vec for eigen_vec in s])
        affinity = affinity + affinity.T
        return np.nan_to_num(affinity)

    def select_most_discriminating_alpha(self, dataset, labels, best_alpha_values, case='classification'):
        silhouettes = list()
        for alpha in best_alpha_values:
            v_top = self.alpha_space(alpha=alpha)
            space = self.cpca_alpha(dataset=dataset, v_top=v_top, alpha=alpha)
            if case == 'classification':
                silhouettes.append(silhouette_score(space, labels))
                best_alpha = best_alpha_values[np.argmax(silhouettes)]
            else:
                # regression
                lin_model = LinearRegression()
                lin_model.fit(space, labels)
                predictions = lin_model.predict(space)
                sum_of_residuals = np.sum(np.abs(predictions - labels))
                silhouettes.append(sum_of_residuals)
                best_alpha = best_alpha_values[np.argmin(silhouettes)]
        return best_alpha

    def all_cpca(self, dataset):
        """
            This function performs contrastive PCA using the alpha technique on the
            active and background dataset. It returns the cPCA-reduced data for all values of alpha specified,
            both the active and background, as well as the list of alphas
        """
        pass
        # alphas = self.generate_alphas()
        # data_to_plot = []
        # for alpha in alphas:
        #     transformed_dataset = self.cpca_alpha(dataset=dataset, alpha=alpha)
        #     data_to_plot.append(transformed_dataset)
        # return data_to_plot, alphas

    def gui(self, background, foreground, active_labels=None, colors=['k', 'r', 'b', 'g', 'c']):

        if self.alpha_selection != 'auto':
            raise ValueError('In order to use gui mode, set alpha_selection to "auto"')

        if not self.n_components == 2:
            # todo: RAISING SUCKS !!!!!!
            raise Warning('Plot cannot be used if the number of components is not 2. '
                          'Plot will only use the first two components.')

        if (foreground.shape[0] > 1000):
            print("The GUI may be slow to respond with large numbers of data points. "
                  "Consider using a subset of the original data.")

        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except ImportError:
            raise ImportError(
                "Something wrong while loading matplotlib.pyplot! You probably don't have plotting libraries installed.")
        try:
            from ipywidgets import widgets, interact, Layout
            from IPython.display import display
        except ImportError:
            raise ImportError(
                "To use the GUI, you must be running this code in a jupyter notebook that supports ipywidgets")

        if active_labels is None:
            active_labels = np.ones(foreground.shape[0])

        transformed_data_auto, alphas_auto = self.automated_cpca(foreground)
        transformed_data_manual, alphas_manual = self.all_cpca(foreground)

        def graph_foreground(ax, fg, active_labels, alpha):
            """
                Handles the plotting
            """
            for i, l in enumerate(np.sort(np.unique(active_labels))):
                ax.scatter(fg[np.where(active_labels == l), 0], fg[np.where(active_labels == l), 1],
                           color=colors[i % len(self.colors)], alpha=0.6)
            if (alpha == 0):
                ax.annotate(r'$\alpha$=' + str(np.round(alpha, 2)) + " (PCA)", (0.05, 0.05),
                            xycoords='axes fraction')
            else:
                ax.annotate(r'$\alpha$=' + str(np.round(alpha, 2)), (0.05, 0.05), xycoords='axes fraction')

        def update(value):
            """
                This code gets run whenever the widget slider is moved
            """
            fig = plt.figure(figsize=[10, 4])
            gs = GridSpec(2, 4)

            for i in range(4):
                ax1 = fig.add_subplot(gs[int(i // 2), i % 2])  # First row, first column
                fg = transformed_data_auto[i]
                graph_foreground(ax1, fg, active_labels, alphas_auto[i])

                ax5 = fig.add_subplot(gs[:, 2:])  # Second row, span all columns

                alpha_idx = np.abs(alphas_manual - 10 ** value).argmin()
                fg = transformed_data_manual[alpha_idx]
                graph_foreground(ax5, fg, active_labels, alphas_manual[alpha_idx])

            # if len(np.unique(self.active_labels))>1:
            # plt.legend()

            plt.tight_layout()
            plt.show()

        widg = interact(update,
                        value=widgets.FloatSlider(description=r'\(\log_{10}{\alpha} \)', min=-1, max=3, step=4 / 40,
                                                  continuous_update=False, layout=Layout(width='80%')))

        return

    def plot(self, foreground, labels=None, mode='', colors=['k', 'r', 'b', 'g', 'c'], background=None):
        if self.n_components > 3:
            raise Warning('Plot cannot be used if the number of components is not 2. '
                          'Plot will only use the first two components.')

        if labels is None:
            labels = np.ones(foreground.shape[0])

        if mode == 'all':
            alphas = self.generate_alphas()
        else:
            alphas = self.alpha_values
        n_alphas = len(alphas)

        unique_labels = np.sort(np.unique(labels))
        num_colors = len(colors)
        fig = plt.figure(figsize=[14, 3])
        for j, a in enumerate(alphas):
            v_top = self.alpha_space(alpha=a)
            fg = self.cpca_alpha(dataset=foreground, v_top=v_top, alpha=a)
            plt.subplot(1, n_alphas, j + 1)
            for i, l in enumerate(unique_labels):
                idx = np.where(labels == l)
                plt.scatter(fg[idx, 0], fg[idx, 1], color=colors[i % num_colors], alpha=0.6,
                            label='Class ' + str(i))
            if background is not None:
                bg = self.cpca_alpha(background, v_top, a)
                plt.scatter(bg[:, 0], bg[:, 1], color='green')
            plt.title('α=' + str(np.round(a, 2)))
        if len(unique_labels) > 1:
            plt.legend()
        # plt.tight_layout()
        plt.show()
        return


class Kernel_CPCA(CPCA):
    def __init__(self, n_components=2, standardize=True, verbose=False, kernel="linear", gamma=10):
        self.kernel=kernel
        self.gamma=gamma
        super().__init__(n_components, standardize, verbose)

    def fit_transform(self, foreground, background, plot=False, gui=False, alpha_selection='auto', n_alphas=40,  max_log_alpha=3, n_alphas_to_return=4, active_labels = None, colors=None, legend=None, alpha_value=None, return_alphas=False):
        self.fg = foreground
        self.bg = background
        self.n_fg, self.features_d = foreground.shape
        self.n_bg, self.features_d_bg = background.shape
        if (gui or plot):
            print("The parameters gui and plot cannot be set to True in Kernel PCA. Will return transformed data as an array instead")
        if not(alpha_selection=='manual'):
            print("The alpha parameter must be set manually for Kernel PCA. Will be using value of alpha = 2")
            alpha_value = 2
        return self.cpca_alpha(alpha_value)

    def fit(self, foreground, background, preprocess_with_pca_dim=None):
        raise ValueError("For Kernel CPCA, the fit() function is not defined. Please use the fit_transform() function directly")

    def transform(self, dataset, alpha_selection='auto', n_alphas=40, max_log_alpha=3, n_alphas_to_return=4, plot=False, gui=False, active_labels = None, colors=None, legend=None, alpha_value=None, return_alphas=False):
        raise ValueError("For Kernel CPCA, the transform() function is not defined. Please use the fit_transform() function directly")

    def cpca_alpha(self, alpha,degree=2,coef0=1):
        N=self.n_fg + self.n_bg
        Z=np.concatenate([self.fg,self.bg],axis=0)

        ## selecting the kernel and computing the kernel matrix
        if self.kernel=='linear':
            K=Z.dot(Z.T)
        elif method=='poly':
            K=(Z.dot(Z.T)+coef0)**degree
        elif method=='rbf':
            K=np.exp(-gamma*squareform(pdist(Z))**2)

        ## Centering the data
        K=centering(K,n)

        ## Using Kernel PCA to do the same
        K_til=np.zeros(K.shape)
        K_til[0:n,:]=K[0:n,:]/n
        K_til[n:,:]=-alpha*K[n:,:]/m
        Sig,A=np.linalg.eig(K_til)
        Sig=np.real(Sig)
        Sig[np.absolute(Sig)<1e-6]=0
        idx_nonzero=Sig!=0
        Sig=Sig[idx_nonzero]
        A=np.real(A[:,idx_nonzero])
        sort_idx=np.argsort(Sig)
        Sig=Sig[sort_idx]
        A=A[:,sort_idx]
        # Normalization
        A_norm=np.zeros(A.shape[1])
        for i in range(A.shape[1]):
            A_norm[i]=np.sqrt(A[:,i].dot(K).dot(A[:,i]).clip(min=0))
            A[:,i]/=A_norm[i]+1e-15

        # todo: why -2
        Z_proj_kernel=K.dot(A[:,-2:])
        X_proj_kernel=Z_proj_kernel[0:n, :]
        Y_proj_kernel=Z_proj_kernel[n:, :]

        return X_proj_kernel #,Y_proj_kernel,Sig[-2:],A[:,-2:]

    # ancillary functions
    def centering(K,n):
        m=K.shape[0]-n
        Kx=K[0:n,:][:,0:n]
        Ky=K[n:,:][:,n:]
        Kxy=K[0:n,:][:,n:]
        Kyx=K[n:,:][:,0:n]
        K_center=np.copy(K)
        K_center[0:n,:][:,0:n]=Kx - np.ones([n,n]).dot(Kx)/n - Kx.dot(np.ones([n,n]))/n \
                               +np.ones([n,n]).dot(Kx).dot(np.ones([n,n]))/n/n
        K_center[n:,:][:,n:]=Ky - np.ones([m,m]).dot(Ky)/m - Ky.dot(np.ones([m,m]))/m \
                             +np.ones([m,m]).dot(Ky).dot(np.ones([m,m]))/m/m
        K_center[0:n,:][:,n:]=Kxy - np.ones([n,n]).dot(Kxy)/n - Kxy.dot(np.ones([m,m]))/m \
                              +np.ones([n,n]).dot(Kxy).dot(np.ones([m,m]))/n/m
        K_center[n:,:][:,0:n]=Kyx - np.ones([m,m]).dot(Kyx)/m - Kyx.dot(np.ones([n,n]))/n \
                              +np.ones([m,m]).dot(Kyx).dot(np.ones([n,n]))/m/n
        return K_center


