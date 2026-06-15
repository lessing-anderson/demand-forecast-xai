import time


def measure_execution_time(func, **kwargs):
    """
    Measure execution time of a function in seconds.
    """

    start = time.perf_counter()

    result = func(**kwargs)

    elapsed_seconds = time.perf_counter() - start

    return result, elapsed_seconds