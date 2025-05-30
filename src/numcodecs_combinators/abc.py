"""
This module defines the [`CodecCombinatorMixin`][numcodecs_combinators.abc.CodecCombinatorMixin] mixin, a common interface for all codec combinator classes.
"""

__all__ = ["CodecCombinatorMixin"]

from abc import ABC, abstractmethod
from typing import Callable

from numcodecs.abc import Codec


class CodecCombinatorMixin(ABC):
    """
    Mixin class for combinators over [`Codec`][numcodecs.abc.Codec]s.
    """

    @abstractmethod
    def map(self, mapper: Callable[[Codec], Codec]) -> Codec:
        """
        Apply the `mapper` to all codecs that are combined by this combinator.
        This method should return a new instance of the combinator, where each
        internal codec is replaced by its mapped codec.

        The `mapper` should recursively apply itself to any inner codecs that
        also implement the [`CodecCombinatorMixin`][numcodecs_combinators.abc.CodecCombinatorMixin]
        mixin. Implementors of this method can thus assume that the `mapper`
        already handles the recursion on its own and can directly call
        `mapper(codec)` in their implementations.

        To automatically handle the recursive application as a caller, you can
        use
        ```py
        numcodecs_combinators.map_codec(codec, mapper)
        ```
        instead.


        Parameters
        ----------
        mapper : Callable[[Codec], Codec]
            The callable that should be applied to each internal codec to map
            over this codec combinator.

        Returns
        -------
        mapped : Codec
            The mapped codec combinator.
        """
