from os.path import dirname, basename, isfile, join
import glob
import importlib

modules = glob.glob(join(dirname(__file__), "*.py"))
module_names = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

_all_classes = []

for module_name in module_names:
    try:
        module = importlib.import_module(f'env.{module_name}')
        for attr_name in dir(module):
            if not attr_name.startswith('_'):
                attr = getattr(module, attr_name)
                if isinstance(attr, type):
                    globals()[attr_name] = attr
                    _all_classes.append(attr_name)
    except Exception as e:
        pass

__all__ = module_names + _all_classes
