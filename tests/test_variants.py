"""
Test suite for Variants metaclass.

Tests the metaclass pattern used for creating model variants.
"""

import pytest
from pyterrier_generative.modelling.util import Variants


class TestVariantsMetaclass:
    """Test Variants metaclass functionality."""

    def test_basic_variant_creation(self):
        """Test creating a class with variants."""

        class TestModel(metaclass=Variants):
            VARIANTS = {
                'variant1': 'model1',
                'variant2': 'model2',
                'variant3': 'model3',
            }

            def __init__(self, model_id, **kwargs):
                self.model_id = model_id
                self.kwargs = kwargs

        # Test accessing variants
        assert hasattr(TestModel, 'variant1')
        assert hasattr(TestModel, 'variant2')
        assert hasattr(TestModel, 'variant3')

        # Test variant methods are callable
        assert callable(TestModel.variant1)
        assert callable(TestModel.variant2)
        assert callable(TestModel.variant3)

    def test_variant_instantiation(self):
        """Test that variants create correct instances."""

        class TestModel(metaclass=Variants):
            VARIANTS = {
                'small': 'model-small',
                'large': 'model-large',
            }

            def __init__(self, model_id, **kwargs):
                self.model_id = model_id
                self.kwargs = kwargs

        # Create instance via variant
        instance = TestModel.small()
        assert isinstance(instance, TestModel)
        assert instance.model_id == 'model-small'

        instance2 = TestModel.large()
        assert isinstance(instance2, TestModel)
        assert instance2.model_id == 'model-large'

    def test_variant_with_parameters(self):
        """Test passing parameters to variant methods."""

        class TestModel(metaclass=Variants):
            VARIANTS = {
                'v1': 'model1',
            }

            def __init__(self, model_id, param1=None, param2=None, **kwargs):
                self.model_id = model_id
                self.param1 = param1
                self.param2 = param2
                self.kwargs = kwargs

        # Pass parameters to variant
        instance = TestModel.v1(param1='value1', param2='value2')
        assert instance.model_id == 'model1'
        assert instance.param1 == 'value1'
        assert instance.param2 == 'value2'

    def test_variant_docstrings(self):
        """Test that variants have proper docstrings."""

        class TestModel(metaclass=Variants):
            VARIANTS = {
                'model1': 'org/model1',
                'model2': 'org/model2',
            }

            def __init__(self, model_id):
                self.model_id = model_id

        # Check docstrings
        assert TestModel.model1.__doc__ is not None
        assert 'org/model1' in TestModel.model1.__doc__

        assert TestModel.model2.__doc__ is not None
        assert 'org/model2' in TestModel.model2.__doc__

    def test_first_variant_is_default(self):
        """Test that first variant is marked as default in docstring."""

        class TestModel(metaclass=Variants):
            VARIANTS = {
                'default_model': 'model-default',
                'other_model': 'model-other',
            }

            def __init__(self, model_id):
                self.model_id = model_id

        # First variant should have *(default)* in docstring
        assert '*(default)*' in TestModel.default_model.__doc__
        assert '*(default)*' not in TestModel.other_model.__doc__

    def test_invalid_variant_raises_error(self):
        """Test that accessing invalid variant raises AttributeError."""

        class TestModel(metaclass=Variants):
            VARIANTS = {
                'valid': 'model1',
            }

            def __init__(self, model_id):
                self.model_id = model_id

        # Should raise AttributeError for invalid variant
        with pytest.raises(AttributeError, match="has no attribute 'invalid'"):
            TestModel.invalid()

    def test_empty_variants(self):
        """Test class with no variants."""

        class TestModel(metaclass=Variants):
            VARIANTS = {}

            def __init__(self):
                pass

        # Should raise AttributeError when trying to access any variant
        with pytest.raises(AttributeError):
            TestModel.anything()

    def test_no_variants_attribute(self):
        """Test class without VARIANTS attribute."""

        class TestModel(metaclass=Variants):
            def __init__(self):
                pass

        # Should raise AttributeError
        with pytest.raises(AttributeError):
            TestModel.anything()

    def test_variant_preserves_args_and_kwargs(self):
        """Test that variants correctly pass through args and kwargs."""

        class TestModel(metaclass=Variants):
            VARIANTS = {
                'v1': 'model1',
            }

            def __init__(self, model_id, *args, **kwargs):
                self.model_id = model_id
                self.args = args
                self.kwargs = kwargs

        # Test with both args and kwargs
        instance = TestModel.v1('arg1', 'arg2', key1='val1', key2='val2')
        assert instance.model_id == 'model1'
        assert instance.args == ('arg1', 'arg2')
        assert instance.kwargs == {'key1': 'val1', 'key2': 'val2'}

    def test_multiple_classes_with_variants(self):
        """Test that multiple classes can use Variants independently."""

        class Model1(metaclass=Variants):
            VARIANTS = {
                'small': 'model1-small',
                'large': 'model1-large',
            }

            def __init__(self, model_id):
                self.model_id = model_id

        class Model2(metaclass=Variants):
            VARIANTS = {
                'fast': 'model2-fast',
                'accurate': 'model2-accurate',
            }

            def __init__(self, model_id):
                self.model_id = model_id

        # Each class should have its own variants
        m1 = Model1.small()
        assert m1.model_id == 'model1-small'

        m2 = Model2.fast()
        assert m2.model_id == 'model2-fast'

        # Variants shouldn't cross-contaminate
        with pytest.raises(AttributeError):
            Model1.fast()

        with pytest.raises(AttributeError):
            Model2.small()

    def test_variant_method_is_static(self):
        """Test that variant methods behave like static methods."""

        class TestModel(metaclass=Variants):
            VARIANTS = {
                'v1': 'model1',
            }

            def __init__(self, model_id):
                self.model_id = model_id

        # Should be callable from class
        instance1 = TestModel.v1()
        assert instance1.model_id == 'model1'

        # Should also be callable from instance (though unusual)
        instance2 = instance1.v1()
        assert instance2.model_id == 'model1'

    def test_ordered_variants(self):
        """Test that variant order is preserved."""

        class TestModel(metaclass=Variants):
            VARIANTS = {
                'first': 'model1',
                'second': 'model2',
                'third': 'model3',
            }

            def __init__(self, model_id):
                self.model_id = model_id

        # First variant should be marked as default
        assert '*(default)*' in TestModel.first.__doc__
        assert '*(default)*' not in TestModel.second.__doc__
        assert '*(default)*' not in TestModel.third.__doc__

    def test_variants_with_inheritance(self):
        """Test Variants metaclass with inheritance."""

        class BaseModel(metaclass=Variants):
            VARIANTS = {
                'base_v1': 'base-model1',
            }

            def __init__(self, model_id):
                self.model_id = model_id

        class DerivedModel(BaseModel):
            VARIANTS = {
                'derived_v1': 'derived-model1',
            }

        # Derived class should have its own variants
        derived = DerivedModel.derived_v1()
        assert derived.model_id == 'derived-model1'

        # Should not have access to base variants
        with pytest.raises(AttributeError):
            DerivedModel.base_v1()


class TestVariantsExport:
    """Test Variants is properly exported."""

    def test_variants_in_all(self):
        """Test Variants is in __all__."""
        from pyterrier_generative.modelling import util
        assert 'Variants' in util.__all__

    def test_variants_importable(self):
        """Test Variants can be imported."""
        from pyterrier_generative.modelling.util import Variants
        assert Variants is not None
        assert isinstance(Variants, type)
