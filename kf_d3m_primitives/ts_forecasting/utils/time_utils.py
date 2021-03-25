import logging
import typing

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# define time constants
SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
DAYS_PER_MONTH_AVG = [28, 30, 31]
DAYS_PER_YEAR_AVG = [365, 366]
DAYS_PER_YEAR = 365.25
DAYS_PER_MONTH = DAYS_PER_YEAR / 12

S_PER_YEAR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_YEAR
S_PER_YEAR_0 = (
    SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_YEAR_AVG[0]
)
S_PER_YEAR_1 = (
    SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_YEAR_AVG[1]
)
S_PER_MONTH = SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_MONTH
S_PER_MONTH_28 = (
    SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_MONTH_AVG[0]
)
S_PER_MONTH_30 = (
    SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_MONTH_AVG[1]
)
S_PER_MONTH_31 = (
    SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_MONTH_AVG[2]
)
S_PER_WEEK = SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_WEEK
S_PER_DAY = SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY
S_PER_HR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR


def calculate_time_frequency(time_diff, model="var"):
    """method that calculates the frequency of a datetime difference (for prediction slices)

    Arguments:
        time_diff {timedelta or float} -- difference between two instances

    Returns:
        str -- string alias representing granularity of pd.datetime object
    """

    return_strings = {
        "var": ["YS", "MS", "M", "W-MON", "D", "H", "S"],
        "gluon": [
            ("12M", "YS"),
            ("M", "MS"),
            ("M", "M"),
            ("W", "W-MON"),
            ("D", "D"),
            ("H", "H"),
            ("S", "S"),
        ],
    }

    # convert to seconds representation
    if type(time_diff) is pd._libs.tslibs.timedeltas.Timedelta:
        time_diff = time_diff.total_seconds()

    if time_diff % S_PER_YEAR_0 == 0:
        logger.debug("granularity is years")
        return return_strings[model][0]
    elif time_diff % S_PER_YEAR_1 == 0:
        logger.debug("granularity is years")
        return return_strings[model][0]
    elif time_diff % S_PER_MONTH_31 == 0:
        # Sunspots monthly
        logger.debug("granularity is months")
        return return_strings[model][1]
    elif time_diff % S_PER_MONTH_30 == 0:
        logger.debug("granularity is months 30")
        return return_strings[model][1]
    elif time_diff % S_PER_MONTH_28 == 0:
        # PHEM monthly
        logger.debug("granularity is months 28")
        return return_strings[model][2]
    elif time_diff % S_PER_WEEK == 0:
        logger.debug("granularity is weeks")
        return return_strings[model][3]
    elif time_diff % S_PER_DAY == 0:
        logger.debug("granularity is days")
        return return_strings[model][4]
    elif time_diff % S_PER_HR == 0:
        logger.debug("granularity is hours")
        return return_strings[model][5]
    else:
        logger.debug("granularity is seconds")
        return return_strings[model][6]


def discretize_time_difference(
    times, initial_time, frequency, integer_timestamps=False, zero_index=False
) -> typing.Sequence[int]:
    """method that discretizes sequence of datetimes (for prediction slices)

    Arguments:
        times {Sequence[datetime] or Sequence[float]} -- sequence of datetime objects
        initial_time {datetime or float} -- last datetime instance from training set
            (to offset test datetimes)
        frequency {str} -- string alias representing granularity of pd.datetime object

    Keyword Arguments:
        integer_timestamps {bool} -- whether timestamps are integers or datetime values
        zero_index {bool} -- whether to subtract 1 from each index to account for 0-indexing

    Returns:
        typing.Sequence[int] -- prediction intervals expressed at specific time granularity

    """

    # take differences to convert to deltas
    time_differences = times - initial_time

    # edge case for integer timestamps
    if integer_timestamps:
        time_differences = time_differences.values.astype(int)
        if zero_index:
            time_differences = [diff - 1 for diff in time_differences]
        return time_differences

    # convert to seconds representation
    if type(time_differences.iloc[0]) is pd._libs.tslibs.timedeltas.Timedelta:
        time_differences = time_differences.apply(lambda t: t.total_seconds())

    if frequency == "YS" or frequency == "12M":
        time_differences = [round(x / S_PER_YEAR) for x in time_differences]
    elif frequency == "MS" or frequency == "M":
        time_differences = [round(x / S_PER_MONTH) for x in time_differences]
        #     round(x / ((1/12 * S_PER_MONTH_28) * (1/3 * S_PER_MONTH_30) + ( * S_PER_MONTH_31)))
        #     for x in time_differences
        # ]
    elif frequency == "W" or frequency == "W-MON":
        time_differences = [round(x / S_PER_WEEK) for x in time_differences]
    elif frequency == "D":
        time_differences = [round(x / S_PER_DAY) for x in time_differences]
    elif frequency == "H":
        time_differences = [round(x / S_PER_HR) for x in time_differences]
    else:
        time_differences = [round(x / SECONDS_PER_MINUTE) for x in time_differences]

    if zero_index:
        time_differences = [diff - 1 for diff in time_differences]

    return time_differences
