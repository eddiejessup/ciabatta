'''Convenience functions for formatting numbers in markdown.'''


def fmt(v):
    return str(int(round(v)))


def fmt_pc(v):
    return fmt(v) + '%'


def fmt_pc_abs(v):
    return fmt_pc(abs(v))


def fmt_as_pc(v):
    return fmt_pc(100 * v)


def fmt_as_abs_pc(v):
    return fmt_as_pc(abs(v))
