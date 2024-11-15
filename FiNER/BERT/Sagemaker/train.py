from transformers import (
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    AutoTokenizer
)
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2
import numpy as np
import logging
import sys
import argparse
import os
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    
    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    
    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load datasets
    train_dataset = torch.load(os.path.join(args.training_dir, "train/processed_data.pt"))
    val_dataset = torch.load(os.path.join(args.training_dir, "validation/processed_data.pt"))
    
    logger.info(f"Loaded train_dataset length: {len(train_dataset)}")
    logger.info(f"Loaded val_dataset length: {len(val_dataset)}")

    # Define id2label mapping
    id2label = {
        0: "O",
        1: "B-PER", 2: "I-PER",
        3: "B-LOC", 4: "I-LOC",
        5: "B-ORG", 6: "I-ORG"
    }
    label2id = {v: k for k, v in id2label.items()}

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = {
            "f1": f1_score(true_labels, true_predictions, mode='strict', scheme=IOB2)
        }
        return results

    # Initialize model
    model = BertForTokenClassification.from_pretrained(
        'bert-base-cased',
        num_labels=7,
        id2label=id2label,
        label2id=label2id
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=args.learning_rate,
        logging_steps=400,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    # Load tokenizer from the same location as the training data
    tokenizer = AutoTokenizer.from_pretrained(args.training_dir)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding=True
        )
    )

    # Train and evaluate
    trainer.train()
    eval_result = trainer.evaluate()

    # Save evaluation results
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        logger.info("***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            logger.info(f"{key} = {value}")
            writer.write(f"{key} = {value}\n")

    
    trainer.save_model(args.model_dir)
    model.config.save_pretrained(args.model_dir)
    # Save the tokenizer
    tokenizer.save_pretrained(args.model_dir)

    