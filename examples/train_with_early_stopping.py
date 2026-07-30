"""Example: Training with EarlyStopping callback.

This example demonstrates how to use the EarlyStopping callback
with Hugging Face Transformers Trainer for supervised fine-tuning.
"""

import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from datasets import Dataset
from areno.callbacks import EarlyStopping


class EarlyStoppingTrainerCallback(TrainerCallback):
    """Wrapper to integrate EarlyStopping with Transformers Trainer.

    This callback monitors evaluation metrics and stops training when
    the specified metric stops improving.
    """

    def __init__(self, early_stopping):
        self.early_stopping = early_stopping

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after each evaluation.

        Args:
            args: TrainingArguments
            state: TrainerState
            control: TrainerControl
            metrics: Evaluation metrics dictionary
        """
        if metrics and self.early_stopping(metrics):
            print(f"\n{'='*60}")
            print(f"Early stopping triggered at step {state.global_step}")
            print(f"Best score: {self.early_stopping.best_score}")
            print(f"{'='*60}\n")
            control.should_training_stop = True
        return control


def load_alpaca_dataset(file_path):
    """Load Alpaca format dataset from JSON file.

    Args:
        file_path: Path to JSON file with instruction/input/output format

    Returns:
        Dataset object
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Format for training: instruction + input -> output
    formatted = []
    for item in data:
        instruction = item["instruction"]
        input_text = item.get("input", "")
        output = item["output"]

        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        formatted.append({
            "prompt": prompt,
            "completion": output,
            "text": prompt + output
        })

    return Dataset.from_list(formatted)


def main():
    """Run training with early stopping."""
    print("=" * 60)
    print("AReno EarlyStopping Training Example")
    print("=" * 60)

    # Configuration
    model_name = "gpt2"  # Small model for quick testing
    data_path = "data/training_data.json"
    output_dir = "outputs"

    # Early stopping configuration
    es_config = {
        "monitor": "eval_loss",
        "patience": 2,
        "mode": "min",
        "min_delta": 0.01,
        "verbose": True,
        "warmup": 1,  # Skip first evaluation
    }

    print(f"\nModel: {model_name}")
    print(f"Early stopping config: {es_config}")

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print(f"Loading dataset from {data_path}...")
    dataset = load_alpaca_dataset(data_path)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])

    # Split train/eval
    split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Setup early stopping
    early_stopping = EarlyStopping(**es_config)
    es_callback = EarlyStoppingTrainerCallback(early_stopping)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        learning_rate=5e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[es_callback],
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    try:
        trainer.train()
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best eval_loss: {early_stopping.best_score}")
        print(f"Total evaluations: {early_stopping.total_evals}")
        print("=" * 60)
    except Exception as e:
        print(f"\nTraining error: {e}")
        raise


if __name__ == "__main__":
    main()
