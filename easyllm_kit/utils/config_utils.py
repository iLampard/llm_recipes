import ast
import os
from datetime import datetime
from typing import Any
import time
from functools import wraps


def make_json_compatible_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        return value
    else:
        return str(value)


def convert_str_2_list_or_float(value: str) -> Any:
    """
    Parse a string value into its appropriate Python type.
    
    Args:
        value (str): The string value to parse.
    
    Returns:
        The parsed value in its appropriate type.
    """
    # Try to evaluate as a literal first
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass

    # If it's not a literal, try other conversions
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False

    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            # If all else fails, return the original string
            return value


def generate_output_dir(base_dir, exp_name, **kwargs):
    timestamp = datetime.now().strftime("%m%d-%H%M")
    output_dir = base_dir
    temp_str = exp_name
    for k, v in kwargs.items():
        temp_str += f'{k}-{v}'
    temp_str += timestamp
    return os.path.join(output_dir, temp_str)


def measure_time(logger=None):
    """
    A decorator to measure the time taken by a function to execute in seconds.
    If a logger is provided, it logs the time; otherwise, it prints to the console.

    Usage:
    @measure_time(logger=your_logger)
    def your_function():
        # Your code
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()  # Start the timer
            result = func(*args, **kwargs)  # Call the function
            elapsed_time = time.time() - start_time  # Calculate the time difference
            message = f"Time taken by '{func.__name__}': {elapsed_time:.4f} seconds"

            if logger:
                logger.info(message)
            else:
                print(message)

            return result

        return wrapper

    return decorator

