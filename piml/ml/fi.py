import joblib
import numpy as np


def get_permutation_importance(est, X, y, iters: int = 25, random_state: int = None, n_jobs: int = None,
                               lower_is_better: bool = True, **score_kwargs) -> np.ndarray:
    """ Compute feature importance based on permutation

    Parameters
    ----------
    est :
        Trained estimator following sklearn API. Most importantly, it needs to implement a `.score(X, y)` method.
    X : array-like
        Input data
    y : array-like
        Target data
    iters : int
        Number of permutation iterations per feature
    random_state : int
        Set to integer for reproducibility.
    n_jobs : int
        Number of parallel jobs to run. If None, all CPUs are used.
    lower_is_better : bool
        Whether lower scores are better. If True, the relative score will be > 1 for important features.
    """
    n, p = X.shape

    # Reproducible numpy randomness
    rng = np.random.default_rng(seed=random_state)

    # Reference score
    score_ref = est.score(X, y, **score_kwargs)

    def get_feature_fi(p_i: int) -> float:
        score = 0.
        for _ in range(iters):
            X_i_shuff = np.copy(X)
            X_i_shuff[:, p_i] = rng.permutation(X_i_shuff[:, p_i])
            score_shuff = est.score(X_i_shuff, y, **score_kwargs)
            score += score_shuff
        return score / iters

    # Randomly shuffle each feature and make prediction and compute score
    scores_avg = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(get_feature_fi)(p_i) for p_i in range(p)
    )
    scores_avg = np.array(scores_avg)

    # Make average of collected shuffled scores relative to reference score
    scores_avg = scores_avg / score_ref

    if lower_is_better:
        # If lower is better, relative scores_avg will be > 1 for important features
        return scores_avg - 1
    else:
        # If higher is better, relative scores_avg will be < 1 for important features
        return 1 - scores_avg
