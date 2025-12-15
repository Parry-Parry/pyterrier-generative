"""Utility classes for creating model variants."""


class Variants(type):
    """
    Simplified metaclass for creating variant class methods.

    This metaclass automatically creates class methods for each variant defined
    in the VARIANTS class attribute. Much simpler than before - just creates
    methods that call __init__ with the variant's model_id.

    Example::

        class MyRanker(BaseRanker, metaclass=Variants):
            VARIANTS = {
                'gpt4': 'gpt-4',
                'gpt35': 'gpt-3.5-turbo',
            }

            def __init__(self, model_id=None, **kwargs):
                # Use first variant as default
                self.model_id = model_id or next(iter(self.VARIANTS.values()))
                # ... rest of initialization

        # Usage:
        ranker = MyRanker.gpt4()  # Creates instance with model_id='gpt-4'
        ranker = MyRanker.gpt35(window_size=10)  # With custom params
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # If VARIANTS is defined in THIS class (not inherited), create classmethods
        if 'VARIANTS' in namespace and namespace['VARIANTS']:
            # First, collect all variant methods from base classes
            inherited_variants = set()
            for base in bases:
                if hasattr(base, 'VARIANTS') and base.VARIANTS:
                    inherited_variants.update(base.VARIANTS.keys())

            # Override inherited variants to raise AttributeError
            for inherited_variant in inherited_variants:
                def make_blocker(variant_name):
                    def blocker(kls):
                        raise AttributeError(
                            f"'{kls.__name__}' has no variant '{variant_name}'. "
                            f"This variant is from a parent class and not available here."
                        )
                    return classmethod(blocker)

                setattr(cls, inherited_variant, make_blocker(inherited_variant))

            # Create methods for this class's variants
            for variant_name, model_id in namespace['VARIANTS'].items():
                # Create a classmethod that instantiates with this model_id
                # Need to capture model_id in closure
                def make_variant(mid):
                    @classmethod
                    def variant_method(kls, *args, **kw):
                        return kls(mid, *args, **kw)
                    return variant_method

                # Create and set the method
                method = make_variant(model_id)
                setattr(cls, variant_name, method)

                # Add docstring
                is_default = variant_name == next(iter(namespace['VARIANTS']))
                prefix = '*(default)* ' if is_default else ''
                method.__func__.__doc__ = f"{prefix}Model: ``{model_id}``"

        return cls


__all__ = ['Variants']
