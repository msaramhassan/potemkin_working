#!/usr/bin/env python3
"""
Test script to verify activation capture functionality works correctly.
"""

import os
import sys
import tempfile
import shutil
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    generate_inference_with_activations,
    answer_benchmark_question_with_activations,
    load_activations
)

def test_basic_activation_capture():
    """Test basic activation capture functionality."""
    print("Testing basic activation capture...")
    
    model = "meta-llama/Llama-3.2-3B-Instruct"
    prompt = "What is 2 + 2?"
    
    try:
        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        # Generate response with activations
        response, activation_data = generate_inference_with_activations(
            prompt=prompt,
            model=model,
            save_path=temp_path,
            layers_to_capture=[0, 1]  # Just capture first 2 layers for speed
        )
        
        print(f"✓ Response generated: {response[:50]}...")
        print(f"✓ Activations captured: {list(activation_data['activations'].keys())}")
        
        # Verify file was saved
        assert os.path.exists(temp_path), "Activation file was not saved"
        print("✓ Activation file saved successfully")
        
        # Test loading
        loaded_data = load_activations(temp_path)
        print("✓ Activation file loaded successfully")
        
        # Verify data integrity
        assert loaded_data['prompt'] == prompt
        assert 'activations' in loaded_data
        assert len(loaded_data['activations']) > 0
        print("✓ Data integrity verified")
        
        # Cleanup
        os.unlink(temp_path)
        print("✓ Basic activation capture test passed!\n")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic activation capture test failed: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return False

def test_benchmark_activation_capture():
    """Test benchmark question activation capture."""
    print("Testing benchmark activation capture...")
    
    model = "meta-llama/Llama-3.2-3B-Instruct"
    question = "What is the square root of 16?"
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Generate response with activations
        full_response, extracted_answer, activation_data = answer_benchmark_question_with_activations(
            question=question,
            model=model,
            save_activations=True,
            activation_save_dir=temp_dir
        )
        
        print(f"✓ Question answered: {extracted_answer}")
        print(f"✓ Full response length: {len(full_response)} characters")
        print(f"✓ Activations captured: {len(activation_data['activations'])} layers")
        
        # Verify activation file was created in directory
        pkl_files = [f for f in os.listdir(temp_dir) if f.endswith('.pkl')]
        assert len(pkl_files) > 0, "No activation files created"
        print(f"✓ Activation file created: {pkl_files[0]}")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        print("✓ Benchmark activation capture test passed!\n")
        
        return True
        
    except Exception as e:
        print(f"✗ Benchmark activation capture test failed: {e}")
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False

def test_activation_analysis():
    """Test activation analysis functionality."""
    print("Testing activation analysis...")
    
    model = "meta-llama/Llama-3.2-3B-Instruct"
    prompt = "Explain gravity in one sentence."
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        # Generate activations
        response, activation_data = generate_inference_with_activations(
            prompt=prompt,
            model=model,
            save_path=temp_path,
            layers_to_capture=[0]  # Just one layer for quick test
        )
        
        # Load and analyze
        loaded_data = load_activations(temp_path)
        activations = loaded_data['activations']
        
        # Basic analysis
        for layer_name, activation in activations.items():
            import torch
            mean_val = torch.mean(activation).item()
            std_val = torch.std(activation).item()
            shape = activation.shape
            
            print(f"✓ {layer_name}: shape={shape}, mean={mean_val:.6f}, std={std_val:.6f}")
        
        # Cleanup
        os.unlink(temp_path)
        print("✓ Activation analysis test passed!\n")
        
        return True
        
    except Exception as e:
        print(f"✗ Activation analysis test failed: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("ACTIVATION CAPTURE FUNCTIONALITY TEST")
    print("=" * 60)
    
    tests = [
        test_basic_activation_capture,
        test_benchmark_activation_capture,
        test_activation_analysis
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed with exception: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All tests passed! Activation capture is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
