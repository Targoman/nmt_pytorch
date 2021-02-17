def __load_all_modules():
    import os
    from glob import glob
    from importlib import import_module
    from .model import load_model_module

    __model_modules_path = os.path.dirname(__file__)

    for __module_path in glob(f'{__model_modules_path}/*.py'):
        __module_name = os.path.splitext(os.path.basename(__module_path))[0]
        if __module_name.startswith('_'):
            continue
        import_module(f'nmt.{__module_name}')

    for __module_path in glob(f'{__model_modules_path}/models/*.py'):
        __model_type = os.path.splitext(os.path.basename(__module_path))[0]
        load_model_module(__model_type)

__load_all_modules()