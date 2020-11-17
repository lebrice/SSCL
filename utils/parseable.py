import dataclasses
import shlex
import sys
from argparse import Namespace
from dataclasses import Field, dataclass, field, is_dataclass
from typing import Dict, List, Optional, Tuple, Type, TypeVar, Union
from pytorch_lightning import LightningDataModule

from utils.utils import camel_case
from simple_parsing import ArgumentParser

from .logging_utils import get_logger

logger = get_logger(__file__)
P = TypeVar("T", bound="Parseable")


class Parseable:
    _argv: Optional[List[str]] = field(default=None, init=False, repr=False)

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    
    @classmethod
    def add_args(cls, parser: ArgumentParser) -> None:
        """ Adds the command-line arguments for this class to the parser.

        Override this if you don't use simple-parsing to add the args.
        """
        if is_dataclass(cls):
            dest = camel_case(cls.__qualname__)
            parser.add_arguments(cls, dest=dest)
        elif issubclass(cls, LightningDataModule):
            # TODO: Test this case out (using a LightningDataModule as a Setting).
            super().add_argparse_args(parser)  # type: ignore
        else:
            raise NotImplementedError(
                f"Don't know how to add command-line arguments for class "
                f"{cls}, since it isn't a dataclass. You must implement the "
                f"`add_args` classmethod yourself."
            )
        
    @classmethod
    def from_argparse_args(cls: Type[P], args: Namespace) -> P:
        """ Creates an instance of this class from the parsed arguments.
        
        Override this if you don't use simple-parsing.
        """
        if is_dataclass(cls):
            dest = camel_case(cls.__qualname__)
            return getattr(args, dest)

        # if issubclass(cls, LightningDataModule):
        #     # TODO: Test this case out (using a LightningDataModule as a Setting).
        #     return super()._from_argparse_args(args)  # type: ignore

        raise NotImplementedError(
            f"Don't know how to create an instance of class {cls} from the "
            f"parsed arguments, since it isn't a dataclass. You'll have to "
            f"override the `from_argparse_args` classmethod."
        )

    @classmethod
    def from_args(cls: Type[P],
                  argv: Union[str, List[str]] = None,
                  reorder: bool = True,
                  strict: bool = False) -> P:
        """Parse an instance of this class from the command-line args.

        Parameters
        ----------
        cls : Type[P]
            The class to instantiate. This only supports dataclasses by default.
            For other classes, you'll have to implement this method yourself.
        argv : Union[str, List[str]], optional
            The command-line string or list of string arguments in the style of
            sys.argv. Could also be the unused_args returned by
            .from_known_args(), for example. By default None
        reorder : bool, optional
            Wether to attempt to re-order positional arguments. Only really
            useful when using subparser actions. By default True.
        strict : bool, optional
            Wether to raise an error if there are extra arguments. By default
            False

            TODO: Might be a good idea to actually change this default to 'True'
            to avoid potential subtle bugs in various places. This would however
            make the code slightly more difficult to read, since we'd have to
            pass some unused_args around. Also might be a problem when the same
            argument e.g. batch_size (at some point) is in both the Setting and
            the Method, because then the arg would be 'consumed', and not passed
            to the second parser in the chain.

        Returns
        -------
        P
            The parsed instance of this class.

        Raises
        ------
        NotImplementedError
            [description]
        """
        if not is_dataclass(cls):
            raise NotImplementedError(
                f"Don't know how to create an instance of class {cls} from the "
                f"command-line, as it isn't a dataclass. You'll have to "
                f"override the `from_args` or `from_known_args` classmethods."
            )
        if isinstance(argv, str):
            argv = shlex.split(argv)
        instance, unused_args = cls.from_known_args(
            argv=argv,
            reorder=reorder,
            strict=strict,
        )
        return instance

    @classmethod
    def from_known_args(cls,
                        argv: Union[str, List[str]] = None,
                        reorder: bool = True,
                        strict: bool = False) -> Tuple[P, List[str]]:
        if not is_dataclass(cls):
            raise NotImplementedError(
                f"Don't know how to create an instance of class {cls} from the "
                f"command-line, as it isn't a dataclass. You'll have to "
                f"override the `from_known_args` classmethod."
            )

        if argv is None:
            argv = sys.argv[1:]
        logger.debug(f"parsing an instance of class {cls} from argv {argv}")
        if isinstance(argv, str):
            argv = shlex.split(argv)

        parser = ArgumentParser(description=cls.__doc__)
        cls.add_args(parser)

        instance: P
        if strict:
            args = parser.parse_args(argv)
            unused_args = []
        else:
            args, unused_args = parser.parse_known_args(argv, attempt_to_reorder=reorder)
            if unused_args:
                logger.debug(RuntimeWarning(
                    f"Unknown/unused args when parsing class {cls}: {unused_args}"
                ))
        instance = cls.from_argparse_args(args)
        # Save the argv that were used to create the instance on its `_argv`
        # attribute.
        instance._argv = argv
        return instance, unused_args


    # @classmethod
    # def fields(cls) -> Dict[str, Field]:
    #     return {f.name: f for f in dataclasses.fields(cls)}
