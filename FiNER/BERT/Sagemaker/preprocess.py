import os
import argparse
import pandas as pd
from transformers import BertTokenizerFast
import torch
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Data paths
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--output-data", type=str, required=True)
    parser.add_argument("--model-id", type=str, default="bert-base-cased")
    
    return parser.parse_args()

def process_dataset(df, tokenizer):
    """Process a dataframe of sentences and labels."""
    processed_data = []
    
    for _, row in df.iterrows():
        # Tokenize the input
        tokenized = tokenizer(
            row['words'],
            is_split_into_words=True,
            truncation=True,
            max_length=512  # Make sure no input surpasses BERT's max length
        )
        # Align labels
        
        word_ids = tokenized.word_ids()
        aligned_labels = align_labels(row['labels'], word_ids)
        
        # Store processed example
        processed_data.append({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': aligned_labels
        })
    
    return processed_data

def align_labels(labels, word_ids):
    """Align labels with tokenized input."""
    aligned_labels = []
    last_word = None
    
    # Define label mapping (B- to I-)
    begin2inside = {1: 2, 3: 4, 5: 6}
    
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx != last_word:
            aligned_labels.append(labels[word_idx])
        else:
            label = labels[word_idx]
            if label in begin2inside:
                label = begin2inside[label]
            aligned_labels.append(label)
        last_word = word_idx
    
    return aligned_labels


if __name__ == "__main__":
    args = parse_args()
    
    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.model_id)
    
    # Process each split
    splits = ['train', 'validation', 'test']
    
    for split in splits:
        print(f"Processing {split} split...")
        
        # Load data from input path
        input_path = os.path.join(args.input_data, split, "data.parquet")
        df = pd.read_parquet(input_path)
        
        # Process dataset
        processed_data = process_dataset(df, tokenizer)
        
        # Create output directory for split
        split_output_dir = os.path.join(args.output_data, split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Save processed data
        output_file = os.path.join(split_output_dir, "processed_data.pt")
        torch.save(processed_data, output_file)
        
        print(f"Processed {len(processed_data)} examples for {split}")
        print(f"Saved to {output_file}")
    
    # Save tokenizer config for training (in parent output directory)
    tokenizer.save_pretrained(args.output_data)
    
    print("Done processing all splits!")