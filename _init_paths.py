import os
import sys

def add_path(path):
	if path not in sys.path:
		sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)

lib_path = os.path.abspath(os.path.join(this_dir, 'pytorch_models'))
add_path(lib_path)

lib_path = os.path.abspath(os.path.join(this_dir, 'Utils'))
add_path(lib_path)