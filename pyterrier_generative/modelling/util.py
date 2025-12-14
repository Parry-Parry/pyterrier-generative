"""Utility classes for creating model variants."""


class Variants(type):
    """
    Metaclass for creating variant class attributes.

    This metaclass automatically creates class methods for each variant defined
    in the VARIANTS class attribute.

    Example::

        class MyRanker(GenerativeRanker, metaclass=Variants):
            VARIANTS = {
                'gpt4': 'gpt-4',
                'gpt35': 'gpt-3.5-turbo',
            }

            def __init__(self, model_id=None, **kwargs):
                model_id = model_id or next(iter(self.VARIANTS.values()))
                # ... initialization

        # Usage:
        ranker = MyRanker.gpt4()  # Creates instance with gpt-4
        ranker = MyRanker.gpt35()  # Creates instance with gpt-3.5-turbo
    """

    def __getattr__(cls, name):
        if hasattr(cls, 'VARIANTS') and cls.VARIANTS and name in cls.VARIANTS:
            @staticmethod
            def wrapped(*args, **kwargs):
                return cls(cls.VARIANTS[name], *args, **kwargs)
            wrapped = wrapped.__get__(cls)
            wrapped.__doc__ = f"Model: ``{cls.VARIANTS[name]}`` `[link] <https://huggingface.co/{cls.VARIANTS[name]}>`__"
            if name == next(iter(cls.VARIANTS)):
                wrapped.__doc__ = '*(default)* ' + wrapped.__doc__
            return wrapped
        raise AttributeError(f"'{cls.__name__}' has no attribute '{name}'")

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)


__all__ = ['Variants']
