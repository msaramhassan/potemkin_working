# Model Activation Capture Implementation Summary

## Overview

I've implemented a comprehensive system for capturing and saving model intermediate activations during inference. This allows you to save the internal representations of the model for later analysis or downstream tasks.

## Files Created/Modified

### New Files:

1. **`example_activation_capture.py`** - Demonstrates how to use the activation capture functionality
2. **`ACTIVATION_CAPTURE_README.md`** - Comprehensive documentation
3. **`analyze_activations.py`** - Utility script for analyzing saved activations
4. **`test_activation_capture.py`** - Test script to verify functionality

### Modified Files:

1. **`utils.py`** - Added activation capture functions and enhanced existing functions
2. **`main.py`** - Added command-line options for activation capture

## Key Functions Added

### Core Functions:

1. **`generate_inference_with_activations(prompt, model, save_path=None, layers_to_capture=None)`**

   - Generates inference while capturing intermediate activations
   - Returns response text and activation data
   - Allows selective layer capture for memory optimization

2. **`answer_benchmark_question_with_activations(question, model, save_activations=True, activation_save_dir="activations")`**

   - Wrapper for benchmark questions with activation capture
   - Integrates seamlessly with existing evaluation pipeline

3. **`load_activations(file_path)`**
   - Loads previously saved activation data
   - Returns complete activation dictionary with metadata

### Enhanced Functions:

4. **`answer_and_grade_benchmark_question()`** - Enhanced with optional activation capture
   - Backward compatible - existing code continues to work
   - New optional parameters for activation capture

## Usage Examples

### Basic Usage:

```python
from utils import generate_inference_with_activations

response, activation_data = generate_inference_with_activations(
    prompt="What is the capital of France?",
    model="meta-llama/Llama-3.2-3B-Instruct",
    save_path="my_activations.pkl"
)
```

### Command Line Usage:

```bash
python main.py --capture_activations --activation_save_dir "my_activations"
```

### Analysis:

```bash
python analyze_activations.py --file "my_activations.pkl"
python analyze_activations.py --directory "activations/"
python analyze_activations.py --compare file1.pkl file2.pkl
```

## Data Structure

Saved activation files contain:

```python
{
    "prompt": "Original input prompt",
    "response": "Model response",
    "model": "Model name/path",
    "timestamp": "ISO format timestamp",
    "input_ids": "Tokenized input tensor",
    "activations": {
        "embeddings": torch.Tensor,  # Embedding layer
        "layer_0": torch.Tensor,     # First transformer layer
        "layer_1": torch.Tensor,     # Second transformer layer
        # ... more layers
    }
}
```

## Features

### Memory Optimization:

- Selective layer capture with `layers_to_capture` parameter
- Activations stored on CPU to free GPU memory
- Immediate disk saving option to handle large models

### Analysis Tools:

- Statistical analysis (mean, std, min, max, norm)
- Activation comparison between different prompts
- Cosine similarity calculations
- Optional visualization (histograms, heatmaps)

### Integration:

- Backward compatible with existing code
- Optional activation capture via command-line flags
- Seamless integration with benchmark evaluation pipeline

## Testing

Run the test suite to verify functionality:

```bash
python test_activation_capture.py
```

The test covers:

- Basic activation capture
- Benchmark integration
- File saving/loading
- Data integrity verification

## Memory Considerations

- Activations can be large for bigger models
- Use `layers_to_capture` to select specific layers
- Consider disk space when saving many activation files
- CPU storage reduces GPU memory pressure

## Advanced Usage

### Comparing Activations:

```python
# Load two activation sets
data1 = load_activations("prompt1.pkl")
data2 = load_activations("prompt2.pkl")

# Compare layer activations
layer_acts1 = data1['activations']['layer_0']
layer_acts2 = data2['activations']['layer_0']

# Calculate similarity
similarity = torch.nn.functional.cosine_similarity(
    layer_acts1.flatten().unsqueeze(0),
    layer_acts2.flatten().unsqueeze(0)
).item()
```

### Custom Analysis:

```python
# Load and analyze specific patterns
data = load_activations("my_file.pkl")
for layer_name, activation in data['activations'].items():
    # Your custom analysis here
    attention_patterns = analyze_attention(activation)
    feature_importance = calculate_importance(activation)
```

## Next Steps

You can now:

1. Run `python test_activation_capture.py` to verify everything works
2. Try `python example_activation_capture.py` to see examples
3. Use `--capture_activations` flag with your main evaluation script
4. Analyze saved activations with `python analyze_activations.py`
5. Build custom analysis tools using the `load_activations()` function

The system is designed to be flexible and extensible - you can easily add new analysis functions or modify the capture process for your specific needs.
