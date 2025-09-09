### just updating it for mistral cuz i am done with WSL.

import os
import json
import csv
import random
from openai import OpenAI
# from together import Together
# from anthropic import Anthropic
# import google.generativeai as genai
import re
from private.models import api_keys
from prompts import contains_concept_prompt, subquestion_generation_prompt
import time
# from google.api_core.exceptions import ResourceExhausted
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pickle
from datetime import datetime

# changed the models_to_developer dict to include llama3.1-8b
models_to_developer = {"meta-llama/Llama-3.2-3B-Instruct": "llama",
        #   "gpt-4.5-preview": "openai",
        #   "o3-mini": "openai",
        #   "o1-mini": "openai",
        #   "gemini-2.0-flash-exp": "gemini",
        #   "claude-3-5-sonnet-20241022": "claude",
        #   "mistralai/Mistral-7B-Instruct-v0.2": "together",
        #   "deepseek-ai/DeepSeek-V3": "together",
        #   "deepseek-ai/DeepSeek-R1": "together",
        #   "Qwen/Qwen2-VL-72B-Instruct": "together",
        #   "Qwen/Qwen2-72B-Instruct": "together",
}

models = list(models_to_developer.keys())

BENCHMARK_PATHS = {
    'bbh': 'raw_data/benchmarks/bbh',
    'mmlu': 'raw_data/benchmarks/mmlu'
}
FINAL_TAG = "FINAL ANSWER:"          


def answer_benchmark_question(question: str, model: str) -> tuple[str, str]:
    cot_prompt = (
        "You are an expert tutor. You may think step-by-step to reach the solution, "
        f"but you MUST finish with a line that starts exactly with `{FINAL_TAG}` "
        "followed by your single best answer.\n\n"
        f"Question:\n{question}\n"
    )
    full_msg = generate_inference(cot_prompt, model)
    m = re.search(rf"{re.escape(FINAL_TAG)}\s*(.+)", full_msg, flags=re.I | re.S)
    extracted = m.group(1).strip() if m else None
    return full_msg, extracted


def answer_open_ended_question(question: str, model: str) -> str:
    prompt = (
        "You are an expert tutor. You may think step-by-step to reach the solution, "
        f"but you MUST finish with an answer that starts exactly with `{FINAL_TAG}` "
        "followed by your single best answer. If the question asks you to explain your "
        f"answer, you should also include the explanation after `{FINAL_TAG}`.\n\n"
        f"Question:\n{question}\n"
    )
    full_answer = generate_inference(prompt, model)
    extracted_answer = extract_final_answer(full_answer)
    return extracted_answer


def answer_and_grade_benchmark_question(question: str, 
                                        model: str, 
                                        gold_answer: str,
                                        multiple_choice: bool,
                                        capture_activations: bool = False,
                                        activation_save_dir: str = "activations") -> tuple:
    """
    Answer and grade a benchmark question, optionally capturing activations.
    
    Args:
        question (str): The question to answer
        model (str): Model name/path
        gold_answer (str): The correct answer
        multiple_choice (bool): Whether this is a multiple choice question
        capture_activations (bool): Whether to capture model activations
        activation_save_dir (str): Directory to save activations
    
    Returns:
        tuple: (is_correct, full_response, activation_data_or_none)
    """
    if capture_activations:
        full_msg, extracted, activation_data = answer_benchmark_question_with_activations(
            question, model, save_activations=True, activation_save_dir=activation_save_dir
        )
        is_correct = grade_benchmark(extracted, gold_answer, multiple_choice)
        return is_correct, full_msg, activation_data
    else:
        full_msg, extracted = answer_benchmark_question(question, model)
        is_correct = grade_benchmark(extracted, gold_answer, multiple_choice)
        return is_correct, full_msg, None


def edit_to_introduce_error(question: str, initial_answer: str, model: str) -> str:
    prompt = (
        "Modify the following answer to introduce a subtle error. The error should be "
        "subtle but one such that a human who knows the concept would know the answer "
        "is incorrect. If the answer is already incorrect, you can leave it the same. "
        f"You can reason all you'd like, but end the response with `{FINAL_TAG}` followed "
        "by the full modified answer.\n\n"
        f"Question: {question}\n"
        f"Answer: {initial_answer}\n"
    )
    full_answer = generate_inference(prompt, model)
    modified_answer = extract_final_answer(full_answer)
    return full_answer, modified_answer


def edit_to_remove_error(question: str, initial_answer: str, model: str) -> str:
    prompt = (
        "Modify the following answer to remove any error. If there are errors, "
        "they might be subtle but one such that a human who knows the concept would "
        "know the answer is incorrect. If the answer is already correct, you can leave "
        "it the same. "
        f"You can reason all you'd like, but end the response with `{FINAL_TAG}` followed "
        "by the full modified answer.\n\n"
        f"Question: {question}\n"
        f"Answer: {initial_answer}\n"
    )
    full_answer = generate_inference(prompt, model)
    modified_answer = extract_final_answer(full_answer)
    return full_answer, modified_answer


def extract_final_answer(response: str) -> str:
    match = re.search(rf'{re.escape(FINAL_TAG)}\s*(.*)', response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def llama3_1_formatting(response: str) -> str:
    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 23 July 2024

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

{response}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    return prompt


def generate_inference(prompt, model):

        if not hasattr(generate_inference, "tokenizer"):
            generate_inference.tokenizer = AutoTokenizer.from_pretrained(model)
            generate_inference.model = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=torch.float16, device_map="auto"
            )
            print("Model and tokenizer initialized once.")

        tokenizer = generate_inference.tokenizer
        model_instance = generate_inference.model

        prompt = llama3_1_formatting(prompt)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = model_instance.generate(
                **inputs, max_new_tokens=2048, top_p=0.7, top_k=50, temperature=0.7, pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
          

        return response[input_len:]


def generate_inference_with_activations(prompt, model, save_path=None, layers_to_capture=None):
    """
    Generate inference while capturing intermediate activations.
    
    Args:
        prompt (str): The input prompt
        model (str): Model name/path
        save_path (str, optional): Path to save activations. If None, creates default path
        layers_to_capture (list, optional): List of layer indices to capture. If None, captures all layers
    
    Returns:
        tuple: (response_text, activations_dict)
    """
    
    if not hasattr(generate_inference_with_activations, "tokenizer"):
        generate_inference_with_activations.tokenizer = AutoTokenizer.from_pretrained(model)
        generate_inference_with_activations.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float16, device_map="auto"
        )
        print("Model and tokenizer initialized for activation capture.")

    tokenizer = generate_inference_with_activations.tokenizer
    model_instance = generate_inference_with_activations.model

    # Format prompt
    formatted_prompt = llama3_1_formatting(prompt)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[-1]

    # Dictionary to store activations
    activations = {}
    hooks = []

    # Determine which layers to capture
    if layers_to_capture is None:
        # Capture all transformer layers
        layers_to_capture = list(range(len(model_instance.model.layers)))

    # Hook function to capture activations
    def capture_activation(name):
        def hook(module, input, output):
            # Store the output activation (hidden states)
            if isinstance(output, tuple):
                # For some layers that return tuples, take the first element (hidden states)
                activations[name] = output[0].detach().cpu()
            else:
                activations[name] = output.detach().cpu()
        return hook

    # Register hooks for specified layers
    for layer_idx in layers_to_capture:
        if layer_idx < len(model_instance.model.layers):
            layer_name = f"layer_{layer_idx}"
            hook = model_instance.model.layers[layer_idx].register_forward_hook(
                capture_activation(layer_name)
            )
            hooks.append(hook)

    # Also capture embeddings and final output
    embedding_hook = model_instance.model.embed_tokens.register_forward_hook(
        capture_activation("embeddings")
    )
    hooks.append(embedding_hook)

    # Generate response
    with torch.no_grad():
        outputs = model_instance.generate(
            **inputs, 
            max_new_tokens=2048, 
            top_p=0.7, 
            top_k=50, 
            temperature=0.7, 
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True
        )
        
        # Get the response text
        response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        response_text = response[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Prepare activation data for saving
    activation_data = {
        "prompt": prompt,
        "response": response_text,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "input_ids": inputs["input_ids"].cpu(),
        "activations": activations
    }

    # Save activations if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(activation_data, f)
        print(f"Activations saved to: {save_path}")
    
    return response_text, activation_data


def answer_benchmark_question_with_activations(question: str, model: str, save_activations=True, activation_save_dir="activations"):
    """
    Answer a benchmark question while capturing model activations.
    
    Args:
        question (str): The question to answer
        model (str): Model name/path
        save_activations (bool): Whether to save activations to disk
        activation_save_dir (str): Directory to save activations
    
    Returns:
        tuple: (full_response, extracted_answer, activation_data)
    """
    cot_prompt = (
        "You are an expert tutor. You may think step-by-step to reach the solution, "
        f"but you MUST finish with a line that starts exactly with `{FINAL_TAG}` "
        "followed by your single best answer.\n\n"
        f"Question:\n{question}\n"
    )
    
    # Create save path if saving activations
    save_path = None
    if save_activations:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_question = re.sub(r'[^\w\s-]', '', question)[:50]  # Sanitize filename
        filename = f"benchmark_activations_{timestamp}_{safe_question}.pkl"
        save_path = os.path.join(activation_save_dir, filename)
    
    # Generate response with activations
    full_msg, activation_data = generate_inference_with_activations(cot_prompt, model, save_path)
    
    # Extract final answer
    m = re.search(rf"{re.escape(FINAL_TAG)}\s*(.+)", full_msg, flags=re.I | re.S)
    extracted = m.group(1).strip() if m else None
    
    return full_msg, extracted, activation_data


def load_activations(file_path):
    """
    Load previously saved activations from disk.
    
    Args:
        file_path (str): Path to the saved activation file
    
    Returns:
        dict: Loaded activation data
    """
    with open(file_path, 'rb') as f:
        activation_data = pickle.load(f)
    return activation_data


def generate_subquestions(question, concept, model, num_subquestions):
    prelude = (
        f"The following is a question about the following concept: {concept}. "
        f"Here is the question: {question}.\n\n"
        f"Write {num_subquestions} other questions that test whether someone who understands the concepts the question is testing truly understands them."
    )
    prompt = prelude + "\n\n" + subquestion_generation_prompt
    inference = generate_inference(prompt, model)
    questions = parse_questions(inference)
    return questions
    

def grade_benchmark(model_answer: str | None,
                    gold_answer: str,
                    multiple_choice: bool) -> bool:
    """
    Simple exact-match grader good enough for BBH & MMLU:
      • MMLU-style MCQ  -> compare first char as A/B/C/…
      • BBH open answer -> case-/space-insensitive string match
    """
    if model_answer is None:
        return False

    if multiple_choice:
        return model_answer[0].upper() == gold_answer.upper()
    else:
        return model_answer.strip().lower() == gold_answer.strip().lower()


def grade_open_ended_question(question: str, model_answer: str, model: str) -> tuple[str, str]:
    cot_prompt = (
        "You are an expert tutor. You will be given a question and a possible "
        "answer to the question. Your job is to determine if the answer is correct "
        "or incorrect. You should only grade it correct if the answer (including the reasoning) "
        "is completely correct. "
        f"You can reason all you'd like, but end the response with `{FINAL_TAG}` "
        "followed by either 'correct' or 'incorrect', and nothing should come after that.\n\n"
        f"Question: {question}\n"
        f"Answer: {model_answer}\n"
    )
    full_msg = generate_inference(cot_prompt, model)
    extracted_answer = extract_final_answer(full_msg)
    # Remove asterisks from extracted answer and removing leading or trailing whitespace
    extracted_answer = extracted_answer.replace("*", "").strip()
    return extracted_answer, full_msg


def parse_questions(inference):
    # Questions are included between <question> and </question> tags.
    questions = re.findall(r'<question>(.*?)</question>', inference, re.DOTALL)
    return questions


def relies_on_concept(question, model):
    prompt = contains_concept_prompt + "\n\n" + question + "\n"
    response = generate_inference(prompt, model)
    answer_match = re.search(r'ANSWER:\s*(Yes|No)', response, re.IGNORECASE)
    concept_match = re.search(r'CONCEPT:\s*(.*)', response, re.IGNORECASE)
    answer = answer_match.group(1) if answer_match else None
    concept = concept_match.group(1).strip() if concept_match else None
    try:
        classification = answer.lower() == "yes"
    except:
        if concept is None:
            return False, None
        else:
            return True, concept
    return classification, concept



def sample_question(benchmark, subject=None):
    base_path = BENCHMARK_PATHS[benchmark]
    if subject is None:
        files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
        subject = random.choice(files)
    file_path = os.path.join(base_path, subject)

    if benchmark == 'bbh':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            question_data = random.choice(data['examples'])
            question_str = question_data['input']
            answer = question_data['target']

    elif benchmark == 'mmlu':
        with open(file_path, 'r', encoding='utf-8') as f:
            rows = list(csv.reader(f))
            row = random.choice(rows)
            question_str = row[0]
            choices = row[1:-1]
            answer_letter = row[-1].strip()  # directly use the letter
            formatted_choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            question_str = f"{question_str}\n{formatted_choices}"
            answer = answer_letter

    else:
        raise ValueError("Unsupported benchmark.")

    return question_str, answer, subject