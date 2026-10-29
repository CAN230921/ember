"""Commit independent GDN tapes through the bit-verified original kernel.

The experimental cross-layer kernel failed strict BF16 equality and is not
part of the accepted path. All tapes are checked before any work is queued;
outputs remain private until the enclosing transaction publishes its state.
"""


def commit_many(tapes, count):
    for tape in tapes:
        tape.validate(count)
    return [tape.commit(count) for tape in tapes]
