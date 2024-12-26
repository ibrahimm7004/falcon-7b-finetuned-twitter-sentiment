# Falcon-7B Finetuned for Twitter Sentiment Analysis

## Overview

This project demonstrates how I fine-tuned the `ybelkada/falcon-7b-sharded-bf16` version of the Falcon-7B model for sentiment analysis of tweets. The fine-tuned model is available on Hugging Face and can predict positive or negative sentiments from social media posts, specifically those related to publicly traded stocks.

Model on Hugging Face: [ibrahim7004/falcon-7b-finetuned-twitter](https://huggingface.co/ibrahim7004/falcon-7b-finetuned-twitter)

---

## Dataset

The fine-tuning process used the **Stock Sentiment Analysis Dataset** by Surge AI, a dataset of social media mentions of publicly traded stocks labeled as positive or negative.

- Dataset source: [Surge AI Stock Sentiment](https://github.com/surge-ai/stock-sentiment/blob/main/sentiment.csv)
- This dataset provides high-quality sentiment annotations for training AI models in financial contexts.

---

## Fine-Tuning Process

You can also access the complete code in this Google Colab Notebook.

The fine-tuning process leveraged Hugging Face’s transformers, trl, and peft libraries. Below is a tutorial-like breakdown of the steps for fine-tuning.

1. Install Required Libraries

!pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git
!pip install -q datasets bitsandbytes einops wandb

2. Load and Prepare the Dataset

from datasets import load_dataset

total_dataset = load_dataset("json", data_files="tweets_human_assistant_fixed.json")["train"]
split_dataset = total_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

3. Configure the Model

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import get_kbit_device_map, get_quantization_config, ModelConfig

model_name = "ybelkada/falcon-7b-sharded-bf16"

model_args = ModelConfig(
    model_name_or_path=model_name,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # normal float 4
)

quantization_config = get_quantization_config(model_args)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    quantization_config=quantization_config,
    device_map=get_kbit_device_map(),
    use_cache=False,  # Required for gradient checkpointing
)

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

4. Set Up LoRA Configuration

peft_config = LoraConfig(
    r=16,  # Low-rank adaptation parameter
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.05,  # Dropout rate for LoRA
    bias="none",  # No bias tuning
    task_type="CAUSAL_LM",  # Fine-tuning for causal language modeling
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],  # Target Falcon layers
)

5. Define Training Arguments

from trl import SFTConfig

training_args = SFTConfig(
    output_dir="./results",
    max_seq_length=256,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=200,
    num_train_epochs=5,
    warmup_ratio=0.03,
    save_steps=10,
    logging_steps=10,
    fp16=True,
    gradient_checkpointing=True,
)

6. Train the Model

from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=peft_config,
)

trainer.train()

7. Save the Fine-Tuned Model

---

## Model Usage

The fine-tuned model predicts sentiments in a conversational format. Here’s an example of how to generate predictions:

### Predicting Sentiment:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("ibrahim7004/falcon-7b-finetuned-twitter", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("ibrahim7004/falcon-7b-finetuned-twitter", trust_remote_code=True)

input_text = "### Human: yayyyy thats amazing broo\n### Assistant:"
inputs = tokenizer(input_text, return_tensors="pt")
inputs.pop("token_type_ids", None)

outputs = model.generate(
    **inputs,
    max_new_tokens=1,
    num_return_sequences=1,
    repetition_penalty=2.0,
    temperature=0.7,
    top_k=2,
    top_p=0.95,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = response.split("### Assistant:")[1].strip()  # Extract sentiment
print(response)
```

### Example Output:
Input:
```
### Human: yayyyy thats amazing broo
### Assistant:
```
Output:
```
Positive
```

---

## Loading the Model

To load the model and tokenizer using the Hugging Face `transformers` library, use the following code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("ibrahim7004/falcon-7b-finetuned-twitter", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("ibrahim7004/falcon-7b-finetuned-twitter", trust_remote_code=True)
```

---

## Key Features
- **Model**: Falcon-7B fine-tuned for sentiment classification.
- **Dataset**: Stock Sentiment Dataset from Surge AI.
- **Training**: LoRA-based fine-tuning with 4-bit quantization for efficient training.
- **Inference**: Predicts positive or negative sentiment with a conversational input-output format.

---

## References
- Hugging Face Transformers: [https://huggingface.co/docs/transformers/](https://huggingface.co/docs/transformers/)
- Original Falcon Model: [ybelkada/falcon-7b-sharded-bf16](https://huggingface.co/ybelkada/falcon-7b-sharded-bf16)
- Stock Sentiment Dataset: [https://github.com/surge-ai/stock-sentiment/blob/main/sentiment.csv](https://github.com/surge-ai/stock-sentiment/blob/main/sentiment.csv)

