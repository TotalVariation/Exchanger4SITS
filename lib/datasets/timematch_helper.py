import os
import yaml
import random

import numpy as np
import pandas as pd

import torch


ROOT = os.path.expanduser('~') + '/data/timematch_data'


def get_classes(*countries, root=ROOT, method=set.union,
                combine_spring_and_winter=False):
    class_sets = []
    for country in countries:
        code_to_class = get_code_to_class(country, root, combine_spring_and_winter)
        class_sets.append(set(code_to_class.values()))

    classes = sorted(list(method(*class_sets)))
    return classes


def read_yaml_class_mapping(country, root=ROOT):
    return yaml.load(open(os.path.join(root, 'class_mapping', f'{country}_class_mapping.yml')), Loader=yaml.FullLoader)


def get_code_to_class(country, root=ROOT, combine_spring_and_winter=False):
    class_mapping = read_yaml_class_mapping(country, root)

    code_to_class = {}
    for cls in class_mapping.keys():
        codes = class_mapping[cls]
        if codes is None:
            continue
        if 'spring' in codes and 'winter' in codes:
            if combine_spring_and_winter:
                combined = {**(codes['spring'] if codes['spring'] is not None else {}), **(codes['winter'] if codes['winter'] is not None else {})}
                code_to_class.update({code: cls for code in combined})
            else:
                if codes['spring'] is not None:
                    code_to_class.update({code: f'spring_{cls}' for code in codes['spring'].keys()})
                if codes['winter'] is not None:
                    code_to_class.update({code: f'winter_{cls}' for code in codes['winter'].keys()})
        else:
            code_to_class.update({code: cls for code in codes})
    return code_to_class


def get_shapefile_columns(country):
    cols = _shapefile_columns[country]
    return cols['id'], cols['crop_code']


def get_codification_table(country, root=ROOT):
    codification_table = os.path.join(root, 'class_mapping', f'{country}_codification_table.csv')
    delimiter = ';' if country in ['denmark', 'austria'] else ','
    crop_codes = pd.read_csv(f, delimiter=delimiter, header=None)
    crop_codes = {row[1]: row[2] for row in crop_codes.itertuples()}  # crop_code: (name, group)
    return crop_codes


_shapefile_columns = {
    'denmark': {
        'id': 'id',
        'crop_code': 'afgkode',
    },
    'france': {
        'id': 'id_parcel',
        'crop_code': 'code_cultu',
    },
    'austria': {
        'id': 'geo_id',
        'crop_code': 'snar_code',
    }
}


