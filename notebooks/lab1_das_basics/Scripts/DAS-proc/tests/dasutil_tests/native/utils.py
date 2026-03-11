# Set up the native-code context so Python can load it
from . import ncontext

import gzip
import os

import numpy as np

# Import faulthandler in case pyDAS decides to generate a segfault
import faulthandler
faulthandler.enable()


# print('got here')
# import sys
# sys.stdout.flush()

import pyDAS


def make_sines_signals(samples_per_sec: int, signal_specs: list[list[float]],
                       total_time: float = 1.0, dtype=np.float32):
    """
    Generate one or more signals, each of which is a sum of some number of
    sine waves.  This is used for testing the filter code.

    samples_per_sec is the sample rate of the output signal.

    signal_hz is a list of lists of frequencies to use for generating each
    output signal.  For example, an input of [[10, 20], [12, 18, 24]] causes
    two signals to be generated.  The first signal is made from (10Hz + 20Hz)
    sine waves, and the second signal is made from (12Hz + 18Hz + 24Hz) sine
    waves.

    The total time is specified in the total_time keyword argument; the
    default value is 1 second.

    The return value is (t, sigs), where t is a 1D NumPy array of time-steps,
    and sigs is a 2D NumPy array containing the samples for all signals.
    """
    # Time signal
    t = np.linspace(0, total_time, samples_per_sec, False, dtype=dtype)

    sigs = []
    for spec in signal_specs:
        sig = np.zeros_like(t, dtype=dtype)
        for s in spec:
            sig = sig + np.sin(2 * np.pi * s * t, dtype=dtype)

        sigs.append(sig)

    sigs = np.stack(sigs)

    return (t, sigs)


def lfilter_signals(inputs: np.ndarray,
        freq_low, order_low, freq_high, order_high, phase, num_threads: int):
    """
    Wrapper for pyDAS.lfilter() and pyDAS.lfilter_double() with a few assertion-checks.
    """
    assert phase in (0, 1), 'Phase must be 0 or 1'
    assert num_threads > 0, 'Number of threads must be positive'

    if inputs.dtype == np.float32:
        outputs = pyDAS.lfilter(inputs, freq_low, order_low,
                                freq_high, order_high, phase, num_threads)
    elif inputs.dtype == np.float64:
        outputs = pyDAS.lfilter_double(inputs, freq_low, order_low,
                                freq_high, order_high, phase, num_threads)
    else:
        raise RuntimeError(f'Cannot handle input dtype {inputs.dtype}')

    return outputs


def lowcut_signal(input: np.ndarray, cutoff_freq, order, phase):
    """
    Wrapper for pyDAS.lowcut() and pyDAS.lowcut_double() with a few
    assertion-checks.
    """
    assert phase in (0, 1), 'Phase must be 0 or 1'

    if input.dtype == np.float32:
        output = pyDAS.lowcut(input, cutoff_freq, order, phase)
    elif input.dtype == np.float64:
        output = pyDAS.lowcut_double(input, cutoff_freq, order, phase)
    else:
        raise RuntimeError(f'Cannot handle input dtype {input.dtype}')

    return output


def highcut_signal(input: np.ndarray, cutoff_freq, order, phase):
    """
    Wrapper for pyDAS.highcut() and pyDAS.highcut_double() with a few
    assertion-checks.
    """
    assert phase in (0, 1), 'Phase must be 0 or 1'

    if input.dtype == np.float32:
        output = pyDAS.highcut(input, cutoff_freq, order, phase)
    elif input.dtype == np.float64:
        output = pyDAS.highcut_double(input, cutoff_freq, order, phase)
    else:
        raise RuntimeError(f'Cannot handle input dtype {input.dtype}')

    return output


def compare_lfilter(input_hz, signal_specs: list[list[float]], total_time,
                    freq_low, order_low, freq_high, order_high, phase, num_threads,
                    output_filename, dtype=np.float32):
    """
    This function generates input signals, filters them with lfilter() or
    lfilter_double(), and then compares them to the expected results from a
    save-file.  (If expected results are not found then the computed results
    are saved.)

    Checks are performed with assert so that if the comparison fails then the
    test fails.
    """
    dt = 1.0 / input_hz
    (t, inputs) = make_sines_signals(input_hz, signal_specs, total_time, dtype=dtype)
    outputs = lfilter_signals(inputs, freq_low * dt, order_low, freq_high * dt, order_high, phase, num_threads)

    compare_numpy_results(outputs, output_filename,
                          gen_pdf=True, t=t, inputs=inputs)


def compare_lowcut(input_hz, signal_specs: list[list[float]], total_time,
                   cutoff_freq, order, phase, output_filename, dtype=np.float32):
    """
    This function generates an input signal, filters it with lowcut() or
    lowcut_double(), and then compares it to the expected results from a
    save-file.  (If expected results are not found then the computed results
    are saved.)

    Checks are performed with assert so that if the comparison fails then the
    test fails.
    """
    dt = 1.0 / input_hz
    (t, inputs) = make_sines_signals(input_hz, signal_specs, total_time, dtype=dtype)

    # Filter the data row by row
    outputs = []
    for i in range(inputs.shape[0]):
        input = inputs[i, :]
        output = lowcut_signal(input, cutoff_freq * dt, order, phase)
        outputs.append(output)

    outputs = np.stack(outputs)

    compare_numpy_results(outputs, output_filename,
                          gen_pdf=True, t=t, inputs=inputs)


def compare_highcut(input_hz, signal_specs: list[list[float]], total_time,
                    cutoff_freq, order, phase, output_filename, dtype=np.float32):
    """
    This function generates an input signal, filters it with highcut() or
    highcut_double(), and then compares it to the expected results from a
    save-file.  (If expected results are not found then the computed results
    are saved.)

    Checks are performed with assert so that if the comparison fails then the
    test fails.
    """
    dt = 1.0 / input_hz
    (t, inputs) = make_sines_signals(input_hz, signal_specs, total_time, dtype=dtype)

    # Filter the data row by row
    outputs = []
    for i in range(inputs.shape[0]):
        input = inputs[i, :]
        output = highcut_signal(input, cutoff_freq * dt, order, phase)
        outputs.append(output)

    outputs = np.stack(outputs)

    compare_numpy_results(outputs, output_filename,
                          gen_pdf=True, t=t, inputs=inputs)


def get_savefile_path(filename):
    """
    Given the filename of a save-file that holds pre-computed test results to
    check computations against, this function returns the full path to where
    that save-file is stored, relative to the current Python file.

    This function points all save-files into a "data" directory so we don't
    get too many files in the source directory.
    """
    if filename.startswith('test_'):
        filename = filename[5:]

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    return os.path.join(data_dir, filename)


def compare_numpy_results(outputs, output_filename,
                          gen_pdf=False, t=None, inputs=None):
    """
    Compare computed NumPy results in "outputs" to the contents of the file
    indicated by "output_filename".  The test will fail if the computed
    results don't match the contents of the save-file.

    If the save-file doesn't exist, the computed outputs are saved into a
    file by that name.  The file is gzip-compressed to try to save a bit of
    space, since these will be saved in the code repository.

    Optionally, when the outputs are being saved for the first time, a PDF
    file can be generated to facilitate manual review.  In this case, the
    "t" array (time-values) and "inputs" array (input values) should also
    be provided in order to generate a meaningful plot.
    """
    expected_outputs = None
    output_path = get_savefile_path(output_filename)
    if os.path.exists(output_path):
        # Try to load the file.  If file not found, generate one.
        # All other errors raised should kill the test.
        try:
            with gzip.open(output_path) as f:
                expected_outputs = np.load(f)
        except FileNotFoundError:
            print(f'NOTE:  Could not load expected file {output_filename}; generating.')

    if expected_outputs is not None:
        # Compare the expected and actual outputs
        assert outputs.shape == expected_outputs.shape, \
            f'Shape of arrays don\'t match.  Expected = {expected_outputs.shape}; actual = {outputs.shape}'

        if not np.allclose(outputs, expected_outputs):
            (dirpath, basepath) = os.path.split(output_path)
            err_output_path = os.path.join(dirpath, f'FAIL-{basepath}')

            # Save the actual outputs into a gzip-compressed NumPy file
            with gzip.open(err_output_path, mode='wb') as f:
                np.save(f, outputs)

            if gen_pdf:
                # Plot the expected and actual outputs
                plot_numpy_results(t, inputs, outputs, err_output_path,
                                   expected_outputs=expected_outputs)

            assert False, 'Array contents don\'t match.'

    else:
        # Save the actual outputs into a gzip-compressed NumPy file
        with gzip.open(output_path, mode='wb') as f:
            np.save(f, outputs)

        if gen_pdf:
            # Plot numpy results as well
            try:
                print(f'NOTE:  Generating PDF of output for review.')
                plot_numpy_results(t, inputs, outputs, output_path)
            except ImportError:
                print(f'ERROR:  Couldn\'t generate PDF.')
                pass


def plot_numpy_results(t, inputs, outputs, output_path, expected_outputs=None):
    """
    Generate a PDF plot of the inputs and outputs in this test so that they
    may be visually inspected.
    """
    assert len(t.shape) == 1, f'Time array should have only 1 dimension.  t.shape = {t.shape}'
    assert len(inputs.shape) == 2, f'Inputs array should have only 2 dimensions.  inputs.shape = {inputs.shape}'
    assert len(outputs.shape) == 2, f'Outputs array should have only 2 dimensions.  outputs.shape = {outputs.shape}'

    assert inputs.shape == outputs.shape, 'Shape of input/output arrays don\'t match.' + \
        f'  inputs.shape = {inputs.shape}; outputs.shape = {outputs.shape}'

    assert t.shape[0] == inputs.shape[-1], 'Time array has different length than ' + \
        'input/output array lengths.' + \
        f'  t.shape = {t.shape}; inputs.shape = {inputs.shape}; outputs.shape = {outputs.shape}'

    import matplotlib.pyplot as plt

    num_plots = inputs.shape[0]
    fig, axes = plt.subplots(num_plots, 1, sharex=True)

    # Plot each input and its corresponding output.  Input is dashed;
    # output is solid.
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(num_plots):
        axes[i].set_title(f'Signal {i}')
        color = colors[i % len(colors)]
        axes[i].plot(t, inputs[i], color=color, linestyle='dashed')
        axes[i].plot(t, outputs[i], color=color, linestyle='solid')

        if expected_outputs is not None:
            axes[i].plot(t, expected_outputs[i], color='black', linestyle='dotted')

    axes[-1].set_xlabel('Time [seconds]')

    '''
    # Plot inputs
    for i in range(inputs.shape[0]):
        ax1.plot(t, inputs[i])

    ax1.set_title('Inputs')

    for i in range(outputs.shape[0]):
        ax2.plot(t, outputs[i])

    ax2.set_title('Outputs')
    ax2.set_xlabel('Time [seconds]')
    '''

    plt.tight_layout()

    if output_path.endswith('.gz'):
        output_path = output_path[:-3]

    plt.savefig(output_path + '.pdf')

    plt.close()
