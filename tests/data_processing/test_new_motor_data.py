"""Clock regressions for the November motor recordings."""

import numpy as np

from data_processing.sources.new_motor_data import datcon_ticks


def test_datcon_premature_high_word_does_not_insert_a_fifty_second_gap():
    wrap = 1 << 32
    # DatCon increments the high word just before the low word rolls over,
    # then increments again at the true rollover (FLY013).
    exported = [0, 2_053_970_717, wrap - 1_993_333, 2 * wrap - 8_125, 2 * wrap + 2_849_100]
    expected = [0, 2_053_970_717, wrap - 1_993_333, wrap - 8_125, wrap + 2_849_100]
    np.testing.assert_array_equal(datcon_ticks(exported), expected)
