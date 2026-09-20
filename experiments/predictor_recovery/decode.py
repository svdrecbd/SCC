"""Inverse of the declared two-outcome root encoding; no world label needed."""


def decode_root(row, denominator):
    assert len(row)==2 and sum(weight for _,weight in row)==denominator
    return [[target,denominator-weight] for target,weight in row]
