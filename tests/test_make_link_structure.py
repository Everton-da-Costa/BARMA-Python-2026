import numpy as np

from src.make_link_structure import make_link_structure


def test_logit_linkfun():
    """
    Test the numerical accuracy of the logit link functions.

    Verifies that Python outputs match the known R baseline for
    scalar inputs and ensures the functions can process NumPy arrays.
    """
    linkfun, linkinv, mu_eta = make_link_structure("logit")

    p = 0.2  # a probability,        domain (0, 1)
    eta = 0.2  # a linear predictor,   domain ℝ

    np.testing.assert_allclose(linkfun(p), -1.386294361119, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(linkinv(eta), 0.549833997312, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(mu_eta(eta), 0.247516572711, rtol=1e-10, atol=1e-10)

    # 2. Vectorization test (checking if it handles lists of numbers)
    test_array = np.array([0.1, 0.5, 0.9])
    array_result = linkfun(test_array)

    # Verify that the function returned an array with exactly 3 results
    assert len(array_result) == 3
