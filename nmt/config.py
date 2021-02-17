# coding: utf-8
"""
Configuration manager
"""

from typing import Any, T


class Configuration(object):
    def __init__(self, parent: T = None, name: str = '__root__'):
        self.__dict__['parent'] = parent
        self.__dict__['name'] = name
        self.__dict__['__submodules'] = {}
        self.__dict__['__parameters'] = {}

    def fullname(self):

        path = []

        node = self
        while node is not None:
            path.append(node.name)
            node = node.parent

        return '.'.join(reversed(path))

    def ensure_submodule(self, name: str):
        submodules = self.__dict__['__submodules']
        parameters = self.__dict__['__parameters']
        if name in submodules:
            return submodules[name]
        if name in parameters:
            raise Exception(
                '{}: Submodule name `{}` collides with a previously registered parameter.'
                .format(self.fullname(), name)
            )
        submodule = Configuration(self, name)
        submodules[name] = submodule
        return submodule

    def register_param(self, name: str, default: Any = None):
        submodules = self.__dict__['__submodules']
        parameters = self.__dict__['__parameters']
        if name in parameters:
            raise Exception(
                '{}: Parameter `{}` already exists!'.format(
                    self.fullname(), name
                )
            )
        if name in submodules:
            raise Exception(
                '{}: Parameter name `{}` collides with a previously registered submodule.'
                .format(self.fullname(), name)
            )
        parameters[name] = {"default": default}

    def ensure_param(self, name: str, default: Any = None):
        parameters = self.__dict__['__parameters']
        if name in parameters:
            if default is not None and parameters[name][
                'default'] is not None and parameters[name]['default'
                                                           ] != default:
                raise Exception(
                    '{}: Parameter `{}` has already been registered with a different default ({} != {}).'
                    .format(
                        self.fullname(), name, parameters[name]['default'],
                        default
                    )
                )
            return
        self.register_param(name, default)

    def __setattr__(self, name: str, value: Any):
        parameters = self.__dict__['__parameters']
        if not name in parameters:
            raise Exception(
                '{}: Parameter `{}` is not registered.'.format(
                    self.fullname(), name
                )
            )
        parameters[name]["value"] = value
        return value

    def __getattr__(self, name: str):
        parameters = self.__dict__['__parameters']
        if not name in parameters:
            raise Exception(
                '{}: Parameter `{}` is not registered.'.format(
                    self.fullname(), name
                )
            )
        return parameters[name]["value"] \
            if "value" in parameters[name] else \
            parameters[name]["default"]

    def __repr__(self):
        submodules = self.__dict__['__submodules']
        parameters = self.__dict__['__parameters']

        result = {k: v for k, v in parameters.items()}
        result.update({k: v for k, v in submodules.items()})

        return 'Configuration{}'.format(result)

    def load(self, source: dict):
        submodules = self.__dict__['__submodules']
        parameters = self.__dict__['__parameters']
        for name in parameters:
            if name in source:
                parameters[name]["value"] = source[name]
        for name in submodules:
            if name in source:
                if isinstance(source[name], dict):
                    submodules[name].load(source[name])
                else:
                    raise Exception(
                        '{}: Source data for a submodule `{}` must be a `dict` not `{}`.'
                        .format(
                            self.fullname(), name,
                            type(source[dict]).__name__
                        )
                    )

    def get_as_dict(self):
        result = {}
        submodules = self.__dict__['__submodules']
        parameters = self.__dict__['__parameters']
        for name in parameters:
            result[name] = parameters[name]["value"] \
                if "value" in parameters[name] else \
                parameters[name]["default"]

        for name in submodules:
            result[name] = submodules[name].get_as_dict()
        return result
