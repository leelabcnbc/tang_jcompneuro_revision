import os.path

from sys import version_info

assert version_info >= (3, 6), "must be python 3.6 or higher!"

result_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
# print("result root is at {}".format(result_root))
dir_dictionary = {
    'datasets': os.path.join(result_root, 'datasets'),
    'features': os.path.join(result_root, 'features'),
    'models': os.path.join(result_root, 'models'),
    'analyses': os.path.join(result_root, 'analyses'),
    'plots': os.path.join(result_root, 'plots'),
    'private_data': os.path.abspath(os.path.join(result_root, '..', 'private_data')),
    'tang_data_root': os.path.abspath(os.path.join(result_root, '..', 'private_data',
                                                   'tang_data', 'data')),
    'shape_params_data': os.path.abspath(os.path.join(result_root, '..', 'private_data',
                                                      'shape_params_data')),
    'package': os.path.abspath(os.path.dirname(__file__)),
    'root': os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
}
