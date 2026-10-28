"""Opt-in deeper speculative verification (single lane). Default 3 keeps every verified 1-3 row path."""
import os

VERIFY_ROWS = int(os.environ.get('SPARK_VERIFY_ROWS', '3'))
if not 3 <= VERIFY_ROWS <= 8:
    raise ValueError('SPARK_VERIFY_ROWS must be 3..8')
VERIFY_WIDTHS = tuple(range(1, VERIFY_ROWS+1))


def per_row(rows):
    """Rows that keep one M=1 call each to reproduce ordinary decode bits."""
    return 2 <= rows <= VERIFY_ROWS
