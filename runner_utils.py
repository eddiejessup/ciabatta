from os.path import basename, splitext
import glob
import numpy as np
import pickle
import os


def _f_to_i(f):
    """Infer a model's iteration number from its output filename.

    Parameters
    ----------
    f: str
        A path to a model output file.

    Returns
    -------
    i: int
        The iteration number of the model.
    """
    return int(splitext(basename(f))[0])


def get_filenames(dirname):
    """Return all model output filenames inside a model output directory,
    sorted by iteration number.

    Parameters
    ----------
    dirname: str
        A path to a directory.

    Returns
    -------
    filenames: list[str]
        Paths to all output files inside `dirname`, sorted in order of
        increasing iteration number.
    """
    filenames = glob.glob('{}/*.pkl'.format(dirname))
    return sorted(filenames, key=_f_to_i)


def get_recent_filename(dirname):
    """Get filename of latest-time model in a directory.
    """
    return get_filenames(dirname)[-1]


def get_recent_model(dirname):
    """Get latest-time model in a directory."""
    return filename_to_model(get_recent_filename(dirname))


def get_recent_time(dirname):
    """Get latest time in a directory."""
    return filename_to_model(get_recent_filename(dirname)).t


def get_output_every(dirname):
    """Get how many iterations between outputs have been done in a directory
    run.

    If there are multiple values used in a run, raise an exception.

    Parameters
    ----------
    dirname: str
        A path to a directory.

    Returns
    -------
    output_every: int
        The inferred number of iterations between outputs.

    Raises
    ------
    TypeError
        If there are multiple different values for `output_every` found. This
        usually means a run has been resumed with a different value.
    """
    fnames = get_filenames(dirname)
    i_s = np.array([_f_to_i(fname) for fname in fnames])
    everys = list(set(np.diff(i_s)))
    if len(everys) > 1:
        raise TypeError('Multiple values for `output_every` '
                        'found, {}.'.format(everys))
    return everys[0]


def filename_to_model(filename):
    """Load a model output file and return the model.

    Parameters
    ----------
    filename: str
        The path to a model output file.

    Returns
    -------
    m: Model
        The associated model instance.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


def model_to_file(model, filename):
    """Dump a model to a file as a pickle file.

    Parameters
    ----------
    model: Model
        Model instance.
    filename: str
        A path to the file in which to store the pickle output.
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def sparsify(dirname, output_every):
    """Remove files from an output directory at regular interval, so as to
    make it as if there had been more iterations between outputs. Can be used
    to reduce the storage size of a directory.

    If the new number of iterations between outputs is not an integer multiple
    of the old number, then raise an exception.

    Parameters
    ----------
    dirname: str
        A path to a directory.
    output_every: int
        Desired new number of iterations between outputs.

    Raises
    ------
    ValueError
        The directory cannot be coerced into representing `output_every`.
    """
    fnames = get_filenames(dirname)
    output_every_old = get_output_every(dirname)
    if output_every % output_every_old != 0:
        raise ValueError('Directory with output_every={} cannot be coerced to'
                         'desired new value.'.format(output_every_old))
    keep_every = output_every // output_every_old
    fnames_to_keep = fnames[::keep_every]
    fnames_to_delete = set(fnames) - set(fnames_to_keep)
    for fname in fnames_to_delete:
        os.remove(fname)
