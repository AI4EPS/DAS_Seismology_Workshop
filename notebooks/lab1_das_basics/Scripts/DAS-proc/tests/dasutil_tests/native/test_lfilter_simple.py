import inspect
import unittest

import numpy as np

from .utils import compare_lfilter


class SimpleLinearFilterTests:
    """
    This is a generalized set of tests on the lfilter() and lfilter_double()
    native filter code.  After this class is a series of unittest classes to
    run these scenarios with various numbers of computation threads.
    """

    def init_params(self):
        self.signal_specs = [[10, 20], [12, 18, 24], [5, 25, 30]]
        self.num_threads = 1

    # Linear filters on float32 values, 1-phase

    def test_lfilter_f32_hicut_3sigs_order6_1phase(self):
        """
        Test lfilter() with float32 values, high-cut of 3 signals, using 1
        thread, order 6, 1-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_lfilter(1000, self.signal_specs, 1,
            freq_low=0, order_low=6, freq_high=15, order_high=6,
            phase=1, num_threads=self.num_threads,
            output_filename=f'{test_name}.gz',
            dtype=np.float32)

    def test_lfilter_f32_locut_3sigs_order6_1phase(self):
        """
        Test lfilter() with float32 values, low-cut of 3 signals, using 1
        thread, order 6, 1-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_lfilter(1000, self.signal_specs, 1,
            freq_low=15, order_low=6, freq_high=500, order_high=6,
            phase=1, num_threads=self.num_threads,
            output_filename=f'{test_name}.gz',
            dtype=np.float32)

    def test_lfilter_f32_hicut_3sigs_order10_1phase(self):
        """
        Test lfilter() with float32 values, high-cut of 3 signals, using 1
        thread, order 10, 1-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_lfilter(1000, self.signal_specs, 1,
            freq_low=0, order_low=10, freq_high=15, order_high=10,
            phase=1, num_threads=self.num_threads,
            output_filename=f'{test_name}.gz',
            dtype=np.float32)

    def test_lfilter_f32_locut_3sigs_order10_1phase(self):
        """
        Test lfilter() with float32 values, low-cut of 3 signals, using 1
        thread, order 10, 1-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_lfilter(1000, self.signal_specs, 1,
            freq_low=15, order_low=10, freq_high=500, order_high=10,
            phase=1, num_threads=self.num_threads,
            output_filename=f'{test_name}.gz',
            dtype=np.float32)

    # Linear filters on float64 values, 1-phase

    def test_lfilter_f64_hicut_3sigs_order6_1phase(self):
        """
        Test lfilter() with float64 values, high-cut of 3 signals, using 1
        thread, order 6, 1-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_lfilter(1000, self.signal_specs, 1,
            freq_low=0, order_low=6, freq_high=15, order_high=6,
            phase=1, num_threads=self.num_threads,
            output_filename=f'{test_name}.gz',
            dtype=np.float64)

    def test_lfilter_f64_locut_3sigs_order6_1phase(self):
        """
        Test lfilter() with float64 values, low-cut of 3 signals, using 1
        thread, order 6, 1-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_lfilter(1000, self.signal_specs, 1,
            freq_low=15, order_low=6, freq_high=500, order_high=6,
            phase=1, num_threads=self.num_threads,
            output_filename=f'{test_name}.gz',
            dtype=np.float64)

    def test_lfilter_f64_hicut_3sigs_order10_1phase(self):
        """
        Test lfilter() with float64 values, high-cut of 3 signals, using 1
        thread, order 10, 1-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_lfilter(1000, self.signal_specs, 1,
            freq_low=0, order_low=10, freq_high=15, order_high=10,
            phase=1, num_threads=self.num_threads,
            output_filename=f'{test_name}.gz',
            dtype=np.float64)

    def test_lfilter_f64_locut_3sigs_order10_1phase(self):
        """
        Test lfilter() with float64 values, low-cut of 3 signals, using 1
        thread, order 10, 1-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_lfilter(1000, self.signal_specs, 1,
            freq_low=15, order_low=10, freq_high=500, order_high=10,
            phase=1, num_threads=self.num_threads,
            output_filename=f'{test_name}.gz',
            dtype=np.float64)

    # Linear filters on float-32 values, 0-phase

    def test_lfilter_f32_hicut_3sigs_order6_0phase(self):
        """
        Test lfilter() with float32 values, high-cut of 3 signals, using 1
        thread, order 6, 1-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_lfilter(1000, self.signal_specs, 1,
            freq_low=0, order_low=6, freq_high=15, order_high=6,
            phase=0, num_threads=self.num_threads,
            output_filename=f'{test_name}.gz',
            dtype=np.float32)

    def test_lfilter_f32_locut_3sigs_order6_0phase(self):
        """
        Test lfilter() with float32 values, low-cut of 3 signals, using 1
        thread, order 6, 1-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_lfilter(1000, self.signal_specs, 1,
            freq_low=15, order_low=6, freq_high=500, order_high=6,
            phase=0, num_threads=self.num_threads,
            output_filename=f'{test_name}.gz',
            dtype=np.float32)

    def test_lfilter_f32_hicut_3sigs_order10_0phase(self):
        """
        Test lfilter() with float32 values, high-cut of 3 signals, using 1
        thread, order 10, 1-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_lfilter(1000, self.signal_specs, 1,
            freq_low=0, order_low=10, freq_high=15, order_high=10,
            phase=0, num_threads=self.num_threads,
            output_filename=f'{test_name}.gz',
            dtype=np.float32)

    def test_lfilter_f32_locut_3sigs_order10_0phase(self):
        """
        Test lfilter() with float32 values, low-cut of 3 signals, using 1
        thread, order 10, 1-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_lfilter(1000, self.signal_specs, 1,
            freq_low=15, order_low=10, freq_high=500, order_high=10,
            phase=0, num_threads=self.num_threads,
            output_filename=f'{test_name}.gz',
            dtype=np.float32)


    def test_lfilter_f64_hicut_3sigs_order6_0phase(self):
        """
        Test lfilter() with float64 values, high-cut of 3 signals, using 1
        thread, order 6, 1-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_lfilter(1000, self.signal_specs, 1,
            freq_low=0, order_low=6, freq_high=15, order_high=6,
            phase=0, num_threads=self.num_threads,
            output_filename=f'{test_name}.gz',
            dtype=np.float64)

    def test_lfilter_f64_locut_3sigs_order6_0phase(self):
        """
        Test lfilter() with float64 values, low-cut of 3 signals, using 1
        thread, order 6, 1-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_lfilter(1000, self.signal_specs, 1,
            freq_low=15, order_low=6, freq_high=500, order_high=6,
            phase=0, num_threads=self.num_threads,
            output_filename=f'{test_name}.gz',
            dtype=np.float64)

    def test_lfilter_f64_hicut_3sigs_order10_0phase(self):
        """
        Test lfilter() with float64 values, high-cut of 3 signals, using 1
        thread, order 10, 1-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_lfilter(1000, self.signal_specs, 1,
            freq_low=0, order_low=10, freq_high=15, order_high=10,
            phase=0, num_threads=self.num_threads,
            output_filename=f'{test_name}.gz',
            dtype=np.float64)

    def test_lfilter_f64_locut_3sigs_order10_0phase(self):
        """
        Test lfilter() with float64 values, low-cut of 3 signals, using 1
        thread, order 10, 1-phase.
        """
        test_name = inspect.stack()[0][3]
        compare_lfilter(1000, self.signal_specs, 1,
            freq_low=15, order_low=10, freq_high=500, order_high=10,
            phase=0, num_threads=self.num_threads,
            output_filename=f'{test_name}.gz',
            dtype=np.float64)


class TestNativeLinearFilter1Thread(unittest.TestCase, SimpleLinearFilterTests):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_params()
        self.num_threads = 1

class TestNativeLinearFilter2Threads(unittest.TestCase, SimpleLinearFilterTests):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_params()
        self.num_threads = 2

class TestNativeLinearFilter3Threads(unittest.TestCase, SimpleLinearFilterTests):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_params()
        self.num_threads = 3

class TestNativeLinearFilter4Threads(unittest.TestCase, SimpleLinearFilterTests):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_params()
        self.num_threads = 4
