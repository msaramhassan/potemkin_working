#!/usr/bin/env python3
"""
Example script demonstrating how to capture and reuse model activations.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    generate_inference_with_activations,
    answer_benchmark_question_with_activations,
    load_activations,
    sample_question
)
import torch

def main():
    # Example 1: Basic activation capture with a simple prompt
    print("=" * 50)
    print("Example 1: Basic activation capture")
    print("=" * 50)
    
    model = "meta-llama/Llama-3.2-3B-Instruct"
    prompt = "What is the capital of France?"
    
    # Generate response while capturing activations
    response, activation_data = generate_inference_with_activations(
        prompt=prompt,
        model=model,
        save_path="example_activations.pkl",
        layers_to_capture=[0, 1, 2]  # Capture first 3 layers
    )
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"Captured activations for layers: {list(activation_data['activations'].keys())}")
    
    # Show activation shapes
    for layer_name, activation in activation_data['activations'].items():
        print(f"{layer_name} shape: {activation.shape}")
    
    print("\n" + "=" * 50)
    print("Example 2: Benchmark question with activations")
    print("=" * 50)
    
    # Sample a question from a benchmark
    question, answer, subject = sample_question("mmlu")
    print(f"Question: {question}")
    print(f"Gold answer: {answer}")
    print(f"Subject: {subject}")
    
    # Answer the question while capturing activations
    full_response, extracted_answer, activation_data = answer_benchmark_question_with_activations(
        question=question,
        model=model,
        save_activations=True,
        activation_save_dir="benchmark_activations"
    )
    
    print(f"Model response: {extracted_answer}")
    print(f"Full response: {full_response[:200]}...")  # Show first 200 chars
    
    print("\n" + "=" * 50)
    print("Example 3: Loading and analyzing saved activations")
    print("=" * 50)
    
    # Load the activations we just saved
    if os.path.exists("example_activations.pkl"):
        loaded_data = load_activations("example_activations.pkl")
        
        print(f"Loaded data keys: {loaded_data.keys()}")
        print(f"Original prompt: {loaded_data['prompt']}")
        print(f"Original response: {loaded_data['response']}")
        print(f"Model used: {loaded_data['model']}")
        print(f"Timestamp: {loaded_data['timestamp']}")
        
        # Analyze activations
        activations = loaded_data['activations']
        print(f"\nActivation analysis:")
        for layer_name, activation in activations.items():
            # Calculate some basic statistics
            mean_activation = torch.mean(activation).item()
            std_activation = torch.std(activation).item()
            max_activation = torch.max(activation).item()
            min_activation = torch.min(activation).item()
            
            print(f"{layer_name}:")
            print(f"  Shape: {activation.shape}")
            print(f"  Mean: {mean_activation:.4f}")
            print(f"  Std: {std_activation:.4f}")
            print(f"  Max: {max_activation:.4f}")
            print(f"  Min: {min_activation:.4f}")
    
    print("\n" + "=" * 50)
    print("Example 4: Comparing activations across different prompts")
    print("=" * 50)
    
    # Generate activations for two different prompts
    prompts = [
        "What is 2 + 2?",
        "Explain the concept of photosynthesis."
    ]
    
    activation_datasets = []
    for i, prompt in enumerate(prompts):
        response, activation_data = generate_inference_with_activations(
            prompt=prompt,
            model=model,
            save_path=f"comparison_activations_{i}.pkl",
            layers_to_capture=[0, 1]  # Just first 2 layers for comparison
        )
        activation_datasets.append(activation_data)
        print(f"Prompt {i+1}: {prompt}")
        print(f"Response {i+1}: {response[:100]}...")
    
    # Compare layer 0 activations
    if len(activation_datasets) == 2:
        layer_0_act1 = activation_datasets[0]['activations']['layer_0']
        layer_0_act2 = activation_datasets[1]['activations']['layer_0']
        
        # Calculate cosine similarity (simplified version)
        flat_act1 = layer_0_act1.flatten()
        flat_act2 = layer_0_act2.flatten()
        
        # Take only the overlapping dimensions if they differ
        min_len = min(len(flat_act1), len(flat_act2))
        flat_act1 = flat_act1[:min_len]
        flat_act2 = flat_act2[:min_len]
        
        cos_sim = torch.nn.functional.cosine_similarity(
            flat_act1.unsqueeze(0), 
            flat_act2.unsqueeze(0)
        ).item()
        
        print(f"\nCosine similarity between layer_0 activations: {cos_sim:.4f}")
    
    print("\nExample completed! Check the saved activation files:")
    print("- example_activations.pkl")
    print("- benchmark_activations/ directory")
    print("- comparison_activations_0.pkl")
    print("- comparison_activations_1.pkl")

if __name__ == "__main__":
    main()
