import pandas as pd
import torch
import sys

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = sys.argv[1]

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"

if model_name == "gpt2":
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("Loading model gpt2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

elif model_name == "gpt-j":
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    print("Loading model gpt-j...")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to(device)

elif model_name == "gpt-neo":
    print("Loading tokenizer gpt-neo...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-20B")
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-20B").to(device)

elif model_name == "flan":
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    print("Loading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    print("Loading model flan-t5-large...")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(device)

elif model_name == "opt-66":
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-66b", use_fast=False)
    print("Loading model opt-66b...")
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-66b").to(device)

elif model_name == "bloom":
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
    print("Loading model bloom...")
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom").to(device)

text = "I am a"

input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
output = model.generate(input_ids, max_length=30, do_sample=True, top_k=1)
result = tokenizer.decode(output[0], skip_special_tokens=True)

print(result)

# # Load the dataset
# print("Loading dataset...")
# df = pd.read_csv("data/HateXplain/prefix_prompt.csv'")


# # Generate the text
# for i in tqdm(range(len(df))):
#     prompt = df.iloc[i]["text"]
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
#     output = model.generate(input_ids, max_length=30, do_sample=True, top_p=0.95, top_k=1)
#     df.iloc[i]["generated_text"] = tokenizer.decode(output[0], skip_special_tokens=True)

# # Save the dataset
# print("Saving dataset...")
# df.to_csv("data/HateXplain/gpt-j_prefix.csv", index=False)
