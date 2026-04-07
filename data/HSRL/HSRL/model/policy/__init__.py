from os.path import dirname, basename, isfile, join
import glob
import importlib

modules = glob.glob(join(dirname(__file__), "*.py"))
module_names = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

# 动态导入所有模块中的类
for module_name in module_names:
    try:
        module = importlib.import_module(f'model.policy.{module_name}')
        for attr_name in dir(module):
            if not attr_name.startswith('_'):
                attr = getattr(module, attr_name)
                if isinstance(attr, type):
                    globals()[attr_name] = attr
    except Exception as e:
        pass

__all__ = module_names
