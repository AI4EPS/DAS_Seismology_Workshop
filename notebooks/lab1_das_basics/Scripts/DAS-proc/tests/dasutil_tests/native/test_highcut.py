import inspect
import unittest

import numpy as np

from .utils import compare_highcut


class HighCutFilterTestCase(unittest.TestCase):
    """
    This is a set of tests on the highcut() and highcut_double() native
    filter code.  This function is a building block of lfilter(), but doesn't
    have multithreading and only processes a single signal, so it's useful to
    exercise a smaller unit of code.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signal_specs = [[10, 20], [12, 18, 24], [5, 25, 30]]

    # Highcut filters on float32 values, 1-phase

    def test_highcut_f32_3sigs_order6_1phase(self):
        """
        Test highcut() with float32 values, 3 signals, order 6, 1-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_highcut(1000, self.signal_specs, 1, cutoff_freq=15, order=6,
            phase=1, output_filename=f'{test_name}.gz', dtype=np.float32)

    def test_highcut_f32_3sigs_order10_1phase(self):
        """
        Test highcut() with float32 values, 3 signals, order 10, 1-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_highcut(1000, self.signal_specs, 1, cutoff_freq=15, order=10,
            phase=1, output_filename=f'{test_name}.gz', dtype=np.float32)

    # Highcut filters on float32 values, 0-phase

    def test_highcut_f32_3sigs_order6_0phase(self):
        """
        Test highcut() with float32 values, 3 signals, order 6, 0-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_highcut(1000, self.signal_specs, 1, cutoff_freq=15, order=6,
            phase=0, output_filename=f'{test_name}.gz', dtype=np.float32)

    def test_highcut_f32_3sigs_order10_0phase(self):
        """
        Test highcut() with float32 values, 3 signals, order 10, 0-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_highcut(1000, self.signal_specs, 1, cutoff_freq=15, order=10,
            phase=0, output_filename=f'{test_name}.gz', dtype=np.float32)

    # Highcut filters on float64 values, 1-phase

    def test_highcut_f64_3sigs_order6_1phase(self):
        """
        Test highcut() with float64 values, 3 signals, order 6, 1-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_highcut(1000, self.signal_specs, 1, cutoff_freq=15, order=6,
            phase=1, output_filename=f'{test_name}.gz', dtype=np.float64)

    def test_highcut_f64_3sigs_order10_1phase(self):
        """
        Test highcut() with float64 values, 3 signals, order 6, 1-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_highcut(1000, self.signal_specs, 1, cutoff_freq=15, order=10,
            phase=1, output_filename=f'{test_name}.gz', dtype=np.float64)

    # Highcut filters on float64 values, 0-phase

    def test_highcut_f64_3sigs_order6_0phase(self):
        """
        Test highcut() with float64 values, 3 signals, order 6, 0-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_highcut(1000, self.signal_specs, 1, cutoff_freq=15, order=6,
            phase=0, output_filename=f'{test_name}.gz', dtype=np.float64)

    def test_highcut_f64_3sigs_order10_0phase(self):
        """
        Test highcut() with float64 values, 3 signals, order 10, 0-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_highcut(1000, self.signal_specs, 1, cutoff_freq=15, order=10,
            phase=0, output_filename=f'{test_name}.gz', dtype=np.float64)
