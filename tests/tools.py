from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

__all__ = ['CustomDictAsserter']


def jacobian_finite_difference(func, outdim, x0):
    '''
    Compute the jacobian of func with respect to u using a finite differences scheme.

    '''
    indim = len(x0)
    jac = np.zeros((outdim, indim))
    h = np.sqrt(np.finfo(float).eps)
    f = func(x0).copy()
    for i in range(indim):
        x_tmp = x0.copy()
        x_tmp[i] += h
        f_tmp = func(x_tmp)
        jac[:, i] = (f_tmp - f) / h
    return jac


class CustomDictAsserter:
    """
    Customized methods to test dictionaries by iterating through their items.

    Attributes
    ----------
    _testcase: unittest.TestCase
        Testcase to run assertions.
    _recursion_counter : int
        Counter for recursive method-calls
    max_recursions : int
        Limit-number of recursion-calls
    """
    def __init__(self, testcase):
        self._testcase = testcase
        self._recursion_counter = 0
        self.max_recursions = 10

    def assert_dict_almost_equal(self, dict1, dict2):
        """
        Test if the entries in two dictionaries are almost equal. Especially recommended, if dictionaries contain
        floating-point numbers.

        Parameters
        ----------
        dict1 : dict
            First dictionary for comparison
        dict2 : dict
            Second dictionary for comparison

        Return
        ------
        None
        """
        self._recursion_counter += 1
        if self._recursion_counter <= self.max_recursions:
            for key, value in dict1.items():
                if isinstance(value, dict):
                    self.assert_dict_almost_equal(value, dict2[key])
                else:
                    if isinstance(value, np.ndarray):
                        assert_array_almost_equal(value, dict2[key])
                    else:
                        self._testcase.assertAlmostEqual(value, dict2[key])
            self._recursion_counter = 0
        else:
            raise RuntimeError('Recursion stopped to avoid infinite loops')

    def assert_dict_equal(self, dict1, dict2):
        """
        Test if the entries in two dictionaries are equal. Only recommended, if no floating-point-numbers are present.

        Parameters
        ----------
        dict1 : dict
            First dictionary for comparison
        dict2 : dict
            Second dictionary for comparison

        Return
        ------
        None
        """
        self._recursion_counter += 1
        if self._recursion_counter <= self.max_recursions:
            for key, value in dict1.items():
                if isinstance(value, dict):
                    self.assert_dict_equal(value, dict2[key])
                else:
                    if isinstance(value, np.ndarray):
                        assert_array_equal(value, dict2[key])
                    else:
                        self._testcase.assertEqual(value, dict2[key])
            self._recursion_counter = 0
        else:
            raise RuntimeError('Recursion stopped to avoid infinite loops')
