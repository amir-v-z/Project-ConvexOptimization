import numpy as np

def evaluate_function(vars, func_str):

    try:
        if isinstance(vars, tuple):
            x, y = vars
        else:
            x = vars
        return eval(func_str)
    except:
        return None

def evaluate_derivative(vars, derivative_str):

    try:
        if isinstance(vars, tuple):
            x, y = vars
        else:
            x = vars
        return eval(derivative_str)
    except:
        return None