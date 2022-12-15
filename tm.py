
from typing import Callable
import time
from contextlib import contextmanager


@contextmanager
def measure_time(id: str, time_obj: dict):
    """ Context manager to measure the time of a code block and format the measured time
        in a string that can be printed by a callable of choice.
        Arguments:
            msg: string to prepend to the default string which is 'took {}s'
            fn: callable that takes the formatted string as input (default: the standard print function)
    """
    start = time.time()
    yield
    time_obj[id] = time.time() - start
