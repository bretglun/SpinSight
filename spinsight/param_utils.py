import numpy as np


def snap(value, values, mode='nearest'):
    match mode:
        case 'nearest':
            return min(values, key=lambda x: abs(x-value), default=None)
        case 'ceil':
            return min([v for v in values if v >= value], default=None)
        case 'floor':
            return max([v for v in values if v <= value], default=None)
        case _:
            raise ValueError(f'Invalid mode {mode}')


def val_in_bounds(val, minval, maxval):
    if minval is not None and val < minval:
        return False
    if maxval is not None and val > maxval:
        return False
    return True


def filter_objects(objects, minval=None, maxval=None):
    if isinstance(objects, dict):
        return {k: v for k, v in objects.items() if val_in_bounds(v, minval, maxval)}
    return [v for v in objects if val_in_bounds(v, minval, maxval)]


def value_in_objects(value, objects):
    if isinstance(objects, dict):
        return value in objects.values()
    return value in objects


def insert_value_in_list_sorted(value, list_):
    list_.append(value)
    return sorted(list_)


def insert_value_in_dict_sorted(key, value, dict_):
    dict_[key] = value
    return dict(sorted(dict_.items()))


def get_object_values(objects):
    if callable(getattr(objects, 'values', False)):
        return objects.values()
    return objects