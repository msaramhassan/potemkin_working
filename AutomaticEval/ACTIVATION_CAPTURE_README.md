# Model Activation Capture

This module provides functionality to capture and save model intermediate activations during inference, allowing for later analysis and reuse.

## Features

- Capture activations from specific transformer layers during inference
- Save activations to disk for later analysis
- Load and reuse previously captured activations
- Integration with existing benchmark evaluation pipeline
- Flexible layer selection for memory optimization

## Quick Start

### Basic Usage

```python
from utils import generate_inference_with_activations, load_activations

# Generate response while capturing activations
response, activation_data = generate_inference_with_activations(
    prompt="What is the capital of France?",
    model="meta-llama/Llama-3.2-3B-Instruct",
    save_path="my_activations.pkl",
    layers_to_capture=[0, 1, 2]  # Capture first 3 layers
)

# Load activations later
loaded_data = load_activations("my_activations.pkl")
```

### Benchmark Integration

```python
from utils import answer_benchmark_question_with_activations

# Answer benchmark question with activation capture
full_response, extracted_answer, activation_data = answer_benchmark_question_with_activations(
    question="What is the derivative of x^2?",
    model="meta-llama/Llama-3.2-3B-Instruct",
    save_activations=True,
    activation_save_dir="activations"
)
```

### Command Line Usage

```bash
# Run main script with activation capture enabled
python main.py --capture_activations --activation_save_dir "my_activations"

# Specify other parameters
python main.py --model "meta-llama/Llama-3.2-3B-Instruct" \
               --benchmark "mmlu" \
               --num_trials 5 \
               --capture_activations \
               --activation_save_dir "experiment_activations"
```

## Functions

### `generate_inference_with_activations(prompt, model, save_path=None, layers_to_capture=None)`

Generate inference while capturing intermediate activations.

**Parameters:**

- `prompt` (str): Input prompt
- `model` (str): Model name/path
- `save_path` (str, optional): Path to save activations
- `layers_to_capture` (list, optional): Layer indices to capture (default: all layers)

**Returns:**

- `tuple`: (response_text, activation_data)

### `answer_benchmark_question_with_activations(question, model, save_activations=True, activation_save_dir="activations")`

Answer a benchmark question while capturing model activations.

**Parameters:**

- `question` (str): Question to answer
- `model` (str): Model name/path
- `save_activations` (bool): Whether to save to disk
- `activation_save_dir` (str): Directory for saving

**Returns:**

- `tuple`: (full_response, extracted_answer, activation_data)

### `load_activations(file_path)`

Load previously saved activations from disk.

**Parameters:**

- `file_path` (str): Path to saved activation file

**Returns:**

- `dict`: Activation data including activations, metadata, and context

## Activation Data Structure

The activation data is saved as a dictionary containing:

```python
{
    "prompt": "Original input prompt",
    "response": "Model response",
    "model": "Model name/path",
    "timestamp": "ISO format timestamp",
    "input_ids": "Tokenized input tensor",
    "activations": {
        "embeddings": "Embedding layer activations",
        "layer_0": "First transformer layer activations",
        "layer_1": "Second transformer layer activations",
        # ... more layers
    }
}
```

## Memory Considerations

- Activations can be memory-intensive for large models
- Use `layers_to_capture` parameter to select specific layers
- Consider saving to disk immediately if working with limited RAM
- Activations are stored as CPU tensors to free GPU memory

## Example Analysis

```python
import torch
from utils import load_activations

# Load saved activations
data = load_activations("my_activations.pkl")
activations = data['activations']

# Analyze layer statistics
for layer_name, activation in activations.items():
    mean_act = torch.mean(activation).item()
    std_act = torch.std(activation).item()
    print(f"{layer_name}: mean={mean_act:.4f}, std={std_act:.4f}")

# Compare activations between different prompts
data1 = load_activations("prompt1_activations.pkl")
data2 = load_activations("prompt2_activations.pkl")

layer_0_act1 = data1['activations']['layer_0'].flatten()
layer_0_act2 = data2['activations']['layer_0'].flatten()

# Calculate cosine similarity
similarity = torch.nn.functional.cosine_similarity(
    layer_0_act1.unsqueeze(0),
    layer_0_act2.unsqueeze(0)
).item()
print(f"Activation similarity: {similarity:.4f}")
```

## Running the Example

```bash
cd AutomaticEval
python example_activation_capture.py
```

This will demonstrate:

1. Basic activation capture
2. Benchmark question processing with activations
3. Loading and analyzing saved activations
4. Comparing activations across different prompts

## File Organization

Activations are saved with descriptive filenames including:

- Timestamp for uniqueness
- Sanitized prompt text (for identification)
- Model information

Example filename: `benchmark_activations_20240915_143022_123456_What_is_the_capital_of.pkl`

## Integration with Existing Code

The activation capture is designed to be backward compatible. Existing code will continue to work unchanged, and you can opt-in to activation capture by:

1. Adding `capture_activations=True` parameter to function calls
2. Using the `--capture_activations` command line flag
3. Handling the additional return values when activations are captured

## Dependencies

- PyTorch
- Transformers
- Pickle (for serialization)
- Standard Python libraries (os, datetime, re)
