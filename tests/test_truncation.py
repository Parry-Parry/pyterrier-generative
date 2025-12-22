#!/usr/bin/env python
"""
Test script for document truncation functionality across different backends.
"""

import sys
import warnings

# Test tokenizers and token counting
def test_token_counters():
    """Test token counter implementations."""
    print("=" * 60)
    print("Testing Token Counters")
    print("=" * 60)

    from pyterrier_generative.truncation import (
        TiktokenCounter,
        HuggingFaceCounter
    )

    # Test TiktokenCounter with GPT models
    print("\n1. Testing TiktokenCounter (OpenAI)...")
    try:
        counter = TiktokenCounter("gpt-3.5-turbo")
        test_text = "This is a test document with some text."
        token_count = counter.count_tokens(test_text)
        print(f"   Text: '{test_text}'")
        print(f"   Token count: {token_count}")
        print("   ✓ TiktokenCounter works!")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Test HuggingFaceCounter
    print("\n2. Testing HuggingFaceCounter...")
    try:
        # Use a small model for testing
        counter = HuggingFaceCounter("bert-base-uncased")
        test_text = "This is a test document with some text."
        token_count = counter.count_tokens(test_text)
        print(f"   Text: '{test_text}'")
        print(f"   Token count: {token_count}")
        print("   ✓ HuggingFaceCounter works!")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    return True


def test_truncation_algorithm():
    """Test the iterative truncation algorithm."""
    print("\n" + "=" * 60)
    print("Testing Truncation Algorithm")
    print("=" * 60)

    from pyterrier_generative.truncation import (
        TiktokenCounter,
        truncate_documents_iterative,
        estimate_prompt_overhead
    )

    # Create test documents
    doc_texts = [
        "This is a very long document that contains a lot of information about various topics. " * 20,
        "Another document with substantial content that we will use for testing purposes. " * 20,
        "A third document with even more text to ensure we have enough content for testing. " * 20,
    ]

    query = "What is the main topic discussed in these documents?"

    print(f"\n1. Testing with {len(doc_texts)} documents...")
    print(f"   Query: '{query}'")

    try:
        # Use tiktoken counter
        counter = TiktokenCounter("gpt-3.5-turbo")

        # Count original tokens
        original_counts = counter.count_tokens_batch(doc_texts)
        original_total = sum(original_counts)
        overhead = estimate_prompt_overhead(query, len(doc_texts), counter)

        print(f"   Original token counts per doc: {original_counts}")
        print(f"   Original total (docs only): {original_total}")
        print(f"   Estimated overhead: {overhead}")
        print(f"   Original total (with overhead): {original_total + overhead}")

        # Set a tight limit to force truncation
        max_length = 1000
        print(f"   Max allowed length: {max_length}")

        # Apply truncation
        truncated_texts, success = truncate_documents_iterative(
            doc_texts=doc_texts,
            prompt_template_overhead=overhead,
            max_length=max_length,
            token_counter=counter,
            tokens_to_remove_per_iter=50,
            max_iterations=100
        )

        # Count truncated tokens
        truncated_counts = counter.count_tokens_batch(truncated_texts)
        truncated_total = sum(truncated_counts)

        print(f"\n   Truncation {'succeeded' if success else 'failed'}!")
        print(f"   Truncated token counts per doc: {truncated_counts}")
        print(f"   Truncated total (docs only): {truncated_total}")
        print(f"   Final total (with overhead): {truncated_total + overhead}")

        # Verify each doc was truncated
        for i, (orig, trunc) in enumerate(zip(doc_texts, truncated_texts)):
            if len(trunc) < len(orig):
                print(f"   ✓ Doc {i+1} was truncated: {len(orig)} → {len(trunc)} chars")
            else:
                print(f"   - Doc {i+1} unchanged")

        if success and (truncated_total + overhead) <= max_length:
            print("   ✓ Truncation algorithm works correctly!")
            return True
        else:
            print(f"   ✗ Final length {truncated_total + overhead} still exceeds {max_length}")
            return False

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backend_integration():
    """Test integration with mock backends."""
    print("\n" + "=" * 60)
    print("Testing Backend Integration")
    print("=" * 60)

    from pyterrier_generative.truncation import get_token_counter

    # Create mock backend classes
    class MockOpenAIBackend:
        def __init__(self):
            self.model_id = "gpt-3.5-turbo"
            self.max_input_length = 4096

    class MockHFBackend:
        def __init__(self):
            self.model_id = "bert-base-uncased"
            self.max_input_length = 512
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    print("\n1. Testing get_token_counter with OpenAIBackend...")
    try:
        backend = MockOpenAIBackend()
        backend.__class__.__name__ = 'OpenAIBackend'
        counter = get_token_counter(backend)
        test_text = "Hello, world!"
        tokens = counter.count_tokens(test_text)
        print(f"   Token count for '{test_text}': {tokens}")
        print("   ✓ OpenAI backend integration works!")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    print("\n2. Testing get_token_counter with HuggingFaceBackend...")
    try:
        backend = MockHFBackend()
        backend.__class__.__name__ = 'HuggingFaceBackend'
        counter = get_token_counter(backend)
        test_text = "Hello, world!"
        tokens = counter.count_tokens(test_text)
        print(f"   Token count for '{test_text}': {tokens}")
        print("   ✓ HuggingFace backend integration works!")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Document Truncation Test Suite")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Token Counters", test_token_counters()))
    results.append(("Truncation Algorithm", test_truncation_algorithm()))
    results.append(("Backend Integration", test_backend_integration()))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(result for _, result in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED! ✓")
        print("=" * 60)
        return 0
    else:
        print("Some tests FAILED! ✗")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
