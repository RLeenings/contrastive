from sklearn.base import BaseEstimator
from contrastive.contrastive import CPCA


class PipelineCPCA(BaseEstimator):

    def __init__(self, alpha_value=None, alpha_selection='auto',
                 split_var_name=None, background_name=None):

        # todo: remove alpha value and alpha_selection
        # use  ** kwargs and get rid of photonais sanity check
        # self.init_kwargs = kwargs

        # we split into background and foreground according to covariate
        self.split_var_name = split_var_name
        self.background_name = background_name

        # self.needs_y = True
        if self.split_var_name is not None or self.background_name is not None:
            self.needs_covariates = True

        self.cpca = None
        self.alpha_value = alpha_value
        self.alpha_selection = alpha_selection

    def fit(self, X, y=None, **kwargs):

        # init CPCA
        self.cpca = CPCA(alpha_value=self.alpha_value, alpha_selection=self.alpha_selection)

        # if var_name is given, get background/foreground seperator from covariate
        if self.background_name is not None:
            if self.background_name not in kwargs:
                raise ValueError("Could not find background variable {} in kwargs".format(self.split_var_name))

            foreground = X
            background = kwargs[self.background_name]
        else:
            if self.split_var_name is not None:
                if self.split_var_name not in kwargs:
                    raise ValueError("Could not find variable {} in kwargs".format(self.split_var_name))

                split_var = kwargs[self.split_var_name]
            else:
                # we use targets
                split_var = y

            # we assume background == 0 and foreground == 1
            background = X[split_var == 0, :]
            foreground = X[split_var == 1, :]

        # fit cpca
        self.cpca.fit(foreground, background, y)

        return self

    def transform(self, X, y=None, **kwargs):
        # return self.cpca.transform(X), y, kwargs
        return self.cpca.transform(X), kwargs