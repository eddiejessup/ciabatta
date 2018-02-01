'''Convenience functions for formatting numbers in markdown.'''


def fmt(v, n=0):
    return str(round(v, n))


def fmt_pc(v, *args, **kwargs):
    return fmt(v, *args, **kwargs) + '%'


def fmt_pc_abs(v, *args, **kwargs):
    return fmt_pc(abs(v, *args, **kwargs))


def fmt_as_pc(v, *args, **kwargs):
    return fmt_pc(100 * v, *args, **kwargs)


def fmt_as_ipc(v):
    return fmt_pc(int(round(100 * v)))


def fmt_as_abs_pc(v, *args, **kwargs):
    return fmt_as_pc(abs(v), *args, **kwargs)


def fmt_as_abs_ipc(v, *args, **kwargs):
    return fmt_as_ipc(abs(v), *args, **kwargs)
