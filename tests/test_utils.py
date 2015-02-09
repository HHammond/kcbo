from decimal import Decimal

import pytest
import numpy

from kcbo import utils


class TestIsNumeric(object):

    def test_is_numeric(self):
        assert utils.is_numeric(int())
        assert utils.is_numeric(float())
        assert utils.is_numeric(Decimal())
        assert utils.is_numeric(numpy.int())
        assert utils.is_numeric(numpy.int8())
        assert utils.is_numeric(numpy.int16())
        assert utils.is_numeric(numpy.int32())
        assert utils.is_numeric(numpy.int64())
        assert utils.is_numeric(numpy.uint8())
        assert utils.is_numeric(numpy.uint16())
        assert utils.is_numeric(numpy.uint32())
        assert utils.is_numeric(numpy.uint64())
        assert utils.is_numeric(numpy.float())
        assert utils.is_numeric(numpy.float16())
        assert utils.is_numeric(numpy.float32())
        assert utils.is_numeric(numpy.float64())


@pytest.fixture
def big_closed():
    return utils.Interval(0, 10, closed=True)


@pytest.fixture
def left_closed():
    return utils.Interval(0, 5, closed=True)


@pytest.fixture
def right_closed():
    return utils.Interval(5, 10, closed=True)


@pytest.fixture
def center_closed():
    return utils.Interval(2, 8, closed=True)


@pytest.fixture
def offset_closed():
    return utils.Interval(-2, 5, closed=True)


@pytest.fixture
def outside_closed():
    return utils.Interval(-10, -5, closed=True)


@pytest.fixture
def big_open():
    return utils.Interval(0, 10, closed=False)


@pytest.fixture
def left_open():
    return utils.Interval(0, 5, closed=False)


@pytest.fixture
def right_open():
    return utils.Interval(5, 10, closed=False)


@pytest.fixture
def center_open():
    return utils.Interval(2, 8, closed=False)


@pytest.fixture
def offset_open():
    return utils.Interval(-2, 5, closed=False)


@pytest.fixture
def outside_open():
    return utils.Interval(-10, -5, closed=False)


class TestInverval(object):

    def test_numeric_containment(self, big_closed, big_open):

        def assert_type_valid(type):
            assert type(0) in big_closed
            assert type(5) in big_closed
            assert type(10) in big_closed
            assert type(0) not in big_open
            assert type(5) in big_open
            assert type(10) not in big_open

        # Test base types
        assert_type_valid(int)
        assert_type_valid(float)
        assert_type_valid(Decimal)

        # Test numpy types
        assert_type_valid(numpy.int)
        assert_type_valid(numpy.int8)
        assert_type_valid(numpy.int16)
        assert_type_valid(numpy.int32)
        assert_type_valid(numpy.int64)
        assert_type_valid(numpy.uint8)
        assert_type_valid(numpy.uint16)
        assert_type_valid(numpy.uint32)
        assert_type_valid(numpy.uint64)
        assert_type_valid(numpy.float)
        assert_type_valid(numpy.float16)
        assert_type_valid(numpy.float32)
        assert_type_valid(numpy.float64)

        # Test Complex
        try:
            complex(0) in big_closed
            assert False
        except ValueError as e:
            assert e.message == \
                "Interval can only contain numerics or Intervals."

    def test_interval_containment(self,
                                  big_closed,
                                  left_closed,
                                  right_closed,
                                  center_closed,
                                  offset_closed,
                                  outside_closed,
                                  big_open,
                                  left_open,
                                  right_open,
                                  center_open,
                                  offset_open,
                                  outside_open):

        # Closed in Closed
        assert left_closed in big_closed
        assert right_closed in big_closed
        assert center_closed in big_closed
        assert offset_closed not in big_closed
        assert outside_closed not in big_closed
        assert big_closed in big_closed

        # Open in open
        assert left_open in big_open
        assert right_open in big_open
        assert center_open in big_open
        assert offset_open not in big_open
        assert outside_open not in big_open
        assert big_open in big_open

        # Open in closed
        assert left_open in big_closed
        assert right_open in big_closed
        assert center_open in big_closed
        assert offset_open not in big_closed
        assert outside_open not in big_closed
        assert big_open in big_closed

        # Closed in open
        assert left_closed not in big_open
        assert right_closed not in big_open
        assert center_closed in big_open
        assert offset_closed not in big_open
        assert outside_closed not in big_open
        assert big_closed not in big_open

    def test_equality(self, big_open, big_closed):
        assert big_open == big_open
        assert big_open == utils.Interval(0, 10, closed=False)
        assert big_closed == big_closed
        assert big_closed == utils.Interval(0, 10, closed=True)
        assert big_open != big_closed
        assert big_closed != big_open

        assert utils.Interval(0, 1) == utils.Interval(0.0, 1.0)

    def test_bad_construction(self):

        try:
            utils.Interval()
            assert False
        except TypeError:
            assert True

        try:
            utils.Interval(10, 0)
            assert False
        except ValueError as e:
            assert e.message == "Lower bound is greater than upper bound."


@pytest.fixture
def valid_samplers():
    def gen_val(i):
        """Create closure around lambda."""
        return lambda: i

    return [gen_val(i) for i in range(10)]


@pytest.fixture
def invalid_samplers():
    return [None for i in range(10)]


class TestCombineSamplers(object):

    def test_invalid_samplers(self, invalid_samplers):
        output = utils.combine_samplers(*invalid_samplers)
        try:
            output()
            assert False
        except TypeError:
            assert True

    def test_output(self, valid_samplers):
        output = utils.combine_samplers(*valid_samplers)
        assert output() == range(10)


class TestMultiGet(object):

    def test_multi_get(self):
        d = {1: 2, 3: 4, 5: 6}
        assert utils.multi_get(d, [1, 3]) == 2
        assert utils.multi_get(d, [999, 3]) == 4
        assert utils.multi_get(d, [999, 888, 5]) == 6
        assert utils.multi_get(d, [999]) is None
        assert utils.multi_get(d, [999], default=4) == 4


class TestHtmlTableInner(object):

    def test_html_table_inner(self):
        import textwrap

        data = [[1, numpy.int(2), 3.0, numpy.float(4)], ['a', 'b', 'c', 'd']]

        expected = """
        <tr>
        <td style='text-align: right;'>1</td>
        <td style='text-align: right;'>2</td>
        <td style='text-align: right;'>3.00000</td>
        <td style='text-align: right;'>4.00000</td>
        </tr>
        <tr>
        <td style='text-align: left;'>a</td>
        <td style='text-align: left;'>b</td>
        <td style='text-align: left;'>c</td>
        <td style='text-align: left;'>d</td>
        </tr>
        """[1:].rstrip()

        expected = textwrap.dedent(expected)
        assert utils.html_table_inner(data) == expected
