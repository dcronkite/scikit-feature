from __future__ import absolute_import
from . import LCSI


def mifs(X, y, n_selected_faetures=None, beta=0.5, **kwargs):
    """
    This function implements the MIFS feature selection

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be discrete
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select

    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature

    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature
        Selection." JMLR 2012.
    """
    return LCSI.lcsi(X, y, beta=beta, gamma=0, n_selected_features=n_selected_faetures)
