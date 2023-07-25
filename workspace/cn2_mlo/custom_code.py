import numpy as np
from sklearn.preprocessing import FunctionTransformer


def cn2_to_cn2_tf(cn2):
    """ Transform Cn2 to Cn2^(3/2) for Pi-ML training because dimensional analysis is based on integer indices. """
    return np.power(cn2, 3/2)


def cn2_tf_to_cn2(cn2_tf):
    """ Inverse-transform of Cn2^(3/2) to Cn2 for Pi-ML predictions """
    return np.power(cn2_tf, 2/3)


def log10(x):
    return np.log10(x)


def log10_inv(x):
    return np.power(10, x)


power_transformer = FunctionTransformer(func=cn2_to_cn2_tf, inverse_func=cn2_tf_to_cn2)
log10_transformer = FunctionTransformer(func=log10, inverse_func=log10_inv)


if __name__ == '__main__':
    import numpy.testing as npt

    # Test that trafos are inverses of each other
    for tf in [power_transformer, log10_transformer]:
        cn2 = np.power(10, np.random.uniform(low=-16, high=-12, size=1000))
        lcn2_tf = tf.transform(cn2)
        cn2_inv = tf.inverse_transform(lcn2_tf)
        npt.assert_allclose(cn2, cn2_inv)
