import json
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import List
from typing import NamedTuple
from typing import Type
from typing import TypedDict
from typing import get_args
from typing import get_origin
from typing import get_type_hints
from typing import is_typeddict

from src.utils.types_utils import STREnum

SPLIT_KEY = '.'
REMOVE_KEY = 'remove-'


def enum_values(enum: Type[STREnum]) -> List[str]:
    return [x.value for x in enum]  # noqa


def add_arguments_from_named_tuple(
        parser: ArgumentParser,
        named_tuple: Type[NamedTuple],
        **kwargs
):
    add_arguments_from_typed_dict(
        parser, named_tuple, named_tuple._field_defaults, **kwargs  # noqa
    )


def add_arguments_from_typed_dict(
        parser: ArgumentParser,
        typed_dict: Type[TypedDict],
        config_instance: TypedDict,
        force_non_positional=True,
        print_only=False,
        prefix='',
):
    def rec(_typed_dict: Type[TypedDict], instance: dict, prefix: str):
        for key, value_type in get_type_hints(_typed_dict).items():
            assert isinstance(key, str)
            arg_name = f"{prefix}{SPLIT_KEY}{key}" if prefix else key
            if key not in instance:
                assert key in map(
                    lambda action: action.dest, parser._actions
                ), f"Key {key} with no default should already be introduced in the parser"
                continue

            default_value = instance[key]
            is_optional = False
            if hasattr(value_type, '__supertype__'):
                # Handle NewType by extracting the underlying type
                value_type = value_type.__supertype__
            if type(None) in get_args(value_type):
                is_optional = True
                value_types = [t for t in get_args(value_type) if t is not type(None)]
                assert len(value_types) == 1, f"Expected only one type in Optional, got {value_types}"
                value_type = value_types[0]

            if is_typeddict(value_type):
                # Handle nested TypedDict
                rec(value_type, default_value, prefix=arg_name)
            else:
                origin_type = get_origin(value_type)
                args = get_args(value_type)

                # Determine the argument type
                if value_type == bool:
                    if default_value is True:
                        arg_name = f"{prefix}{SPLIT_KEY}no-{key}" if prefix else f"no-{key}"
                        action = 'store_false'
                        dest_name = key
                    else:
                        dest_name = None
                        action = 'store_true'
                    help_text = f"Default: {default_value}"
                    parser.add_argument(arg_name, action=action, help=help_text, dest=dest_name)
                elif is_optional and default_value is not None:
                    arg_type = value_type
                    remove_arg_name = f"{prefix}{SPLIT_KEY}{REMOVE_KEY}{key}" if prefix else f"{REMOVE_KEY}{key}"
                    if isinstance(arg_type, type) and issubclass(arg_type, STREnum):
                        choices = enum_values(arg_type)
                        help_text = f"Choices: {choices}. Default: {default_value}"
                        parser.add_argument(arg_name, type=str, choices=choices, help=help_text)
                        parser.add_argument(remove_arg_name, action='store_const', const=None, dest=arg_name)
                    else:
                        help_text = f"Default: {default_value}"
                        parser.add_argument(arg_name, type=arg_type, help=help_text)
                        parser.add_argument(remove_arg_name, action='store_const', const=None, dest=arg_name)
                elif isinstance(value_type, type) and issubclass(value_type, STREnum):
                    choices = enum_values(value_type)
                    help_text = f"Choices: {choices}. Default: {default_value}"
                    parser.add_argument(arg_name, type=str, choices=choices, help=help_text)
                elif origin_type is list:
                    list_type = args[0] if args else str
                    if isinstance(list_type, type) and issubclass(list_type, STREnum):
                        choices = enum_values(list_type)
                        help_text = f"Choices: {choices}. Nargs: '+'. Default: {default_value}"
                        parser.add_argument(arg_name, type=str, nargs='+', choices=choices, help=help_text)
                    else:
                        help_text = f"Nargs: '+'. Default: {default_value}"
                        parser.add_argument(arg_name, type=list_type, nargs='+', help=help_text)
                elif origin_type is dict and args[0] == str:
                    assert args[1] == Any, f"Expected Dict[str, Any], got Dict[{args[0].__name__}, {args[1].__name__}]"
                    help_text = f"Provide as JSON string. Default: {default_value}"
                    parser.add_argument(arg_name, type=json.loads, help=help_text)
                elif isinstance(value_type, type) and issubclass(value_type, (int, float, str)):
                    help_text = f"Default: {default_value}"
                    parser.add_argument(arg_name, type=value_type, help=help_text)
                elif isinstance(value_type, type) and issubclass(value_type, Path):
                    help_text = f"Default: {default_value}"
                    parser.add_argument(arg_name, type=Path, help=help_text)
                else:
                    raise ValueError(f"Unsupported type: {value_type}")

    _original_add_argument = parser.add_argument
    try:
        def new_add_argument(*args, **kwargs):
            if force_non_positional:
                args = (f"--{args[0]}",) + args[1:]

            if print_only:
                print(f"Args: {args}, Kwargs: {kwargs}")
            else:
                _original_add_argument(*args, **kwargs)

        parser.add_argument = new_add_argument
        rec(typed_dict, config_instance, prefix=prefix)  # noqa
    finally:
        parser.add_argument = _original_add_argument


def update_config_from_args(config_instance: TypedDict, args: Any):
    SPLIT_KEY = '.'
    for arg_key, arg_value in vars(args).items():
        arg_key = arg_key.removeprefix('--')
        keys = arg_key.split(SPLIT_KEY)
        config_section = config_instance

        # Traverse the keys to find the right section
        for key in keys[:-1]:
            config_section = config_section[key]

        if arg_key.startswith(REMOVE_KEY):
            key = arg_key[len(REMOVE_KEY):]
            if key in config_section:
                del config_section[key]
        elif arg_value is not None:
            config_section[keys[-1]] = arg_value


@contextmanager
def prefix_args(parser: ArgumentParser, prefix: str):
    old_prefix = parser.add_argument
    parser.add_argument = lambda *args, **kwargs: old_prefix(f"{prefix}{args[0]}", *args[1:], **kwargs)
    try:
        yield
    finally:
        parser.add_argument = old_prefix


@contextmanager
def prefix_args(parser: ArgumentParser, prefix: str):
    old_prefix = parser.add_argument
    parser.add_argument = lambda *args, **kwargs: old_prefix(f"{prefix}{args[0]}", *args[1:], **kwargs)
    try:
        yield
    finally:
        parser.add_argument = old_prefix
