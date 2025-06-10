import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Import from your own modules
from ..utils.config import ProjectConfig

class StanceDataset(Dataset):
    """PyTorch Dataset for stance detection training and inference.

    Handles tokenization and encoding of text data for BERT processing,
    with proper attention masks and padding for batch training.

    Attributes:
        texts: List of input text strings
        labels: List of stance labels (0=against, 1=neutral, 2=favor)
        tokenizer: Pre-trained BERT tokenizer
        max_length: Maximum sequence length for tokenization
    """

    def __init__(self, texts: List[str], labels: List[int],
                 tokenizer: BertTokenizer, max_length: int = 512):
        """Initialize dataset with texts and labels.

        Args:
            texts: List of input text strings for stance classification
            labels: List of integer labels (0=against, 1=neutral, 2=favor)
            tokenizer: Pre-trained BERT tokenizer for text encoding
            max_length: Maximum sequence length, texts longer will be truncated
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with proper tokenization and encoding.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing input_ids, attention_mask, and labels as tensors
        """
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenize with attention masks and padding
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class StanceDetector:
    """BERT-based stance detection system for political text classification.

    This class implements a complete stance detection pipeline using BERT,
    including training, evaluation, and inference with confidence scoring.
    Designed to classify text as liberal/conservative/neutral with high accuracy.

    Attributes:
        config: Project configuration containing hyperparameters
        logger: Logger for tracking training progress and errors
        tokenizer: BERT tokenizer for text preprocessing
        model: Fine-tuned BERT model for classification
        device: Computing device (CPU/GPU) for model operations
    """

    def __init__(self, config: ProjectConfig, logger: logging.Logger):
        """Initialize stance detector with configuration and logging.

        Args:
            config: Project configuration containing model hyperparameters
            logger: Logger instance for tracking operations
        """
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device)

        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            config.bert_model_name,
            num_labels=3  # against, neutral, favor
        ).to(self.device)

        self.logger.info(f"Initialized StanceDetector with {config.bert_model_name} on {self.device}")

    def prepare_data(self, df: pd.DataFrame, text_column: str,
                     label_column: str, test_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders from DataFrame.

        Args:
            df: DataFrame containing text and labels
            text_column: Name of column containing text data
            label_column: Name of column containing stance labels
            test_size: Fraction of data to use for validation

        Returns:
            Tuple of (train_loader, val_loader) for model training

        Example:
            >>> df = pd.read_csv('debagreement_data.csv')
            >>> train_loader, val_loader = detector.prepare_data(df, 'text', 'stance')
        """
        from sklearn.model_selection import train_test_split

        texts = df[text_column].tolist()
        labels = df[label_column].tolist()

        # Split data maintaining label distribution
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, stratify=labels,
            random_state=self.config.random_seed
        )

        # Create datasets and loaders
        train_dataset = StanceDataset(train_texts, train_labels, self.tokenizer,
                                      self.config.max_sequence_length)
        val_dataset = StanceDataset(val_texts, val_labels, self.tokenizer,
                                    self.config.max_sequence_length)

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size,
                                  shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size,
                                shuffle=False)

        self.logger.info(f"Prepared data: {len(train_dataset)} train, {len(val_dataset)} val samples")
        return train_loader, val_loader

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, float]:
        """Train the stance detection model with validation monitoring.

        Implements full training loop with learning rate scheduling,
        validation monitoring, and best model checkpointing.

        Args:
            train_loader: DataLoader containing training data
            val_loader: DataLoader containing validation data

        Returns:
            Dictionary containing final training metrics

        Example:
            >>> metrics = detector.train(train_loader, val_loader)
            >>> print(f"Final accuracy: {metrics['val_accuracy']:.3f}")
        """
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)

        # Learning rate scheduler for better convergence
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        best_val_accuracy = 0.0
        best_model_state = None

        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")

            # Training phase
            self.model.train()
            total_train_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_train_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Validation phase
            val_accuracy, val_f1 = self.evaluate(val_loader)
            avg_train_loss = total_train_loss / len(train_loader)

            self.logger.info(
                f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, "
                f"Val Accuracy = {val_accuracy:.4f}, Val F1 = {val_f1:.4f}"
            )

            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = self.model.state_dict().copy()
                self.logger.info(f"New best model with validation accuracy: {val_accuracy:.4f}")

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return {
            'val_accuracy': best_val_accuracy,
            'val_f1': val_f1,
            'train_loss': avg_train_loss
        }

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model performance on given dataset.

        Args:
            data_loader: DataLoader containing evaluation data

        Returns:
            Tuple of (accuracy, f1_score) metrics
        """
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                predictions.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')

        return accuracy, f1

    def predict_with_confidence(self, text: str) -> Dict[str, Any]:
        """Predict stance with confidence scoring for single text input.

        This is the core inference method used throughout the temporal analysis
        pipeline for processing user posts and generating stance trajectories.

        Args:
            text: Input text to classify

        Returns:
            Dictionary containing stance prediction, confidence, and probabilities

        Example:
            >>> result = detector.predict_with_confidence("Healthcare should be free")
            >>> print(f"Stance: {result['stance']}, Confidence: {result['confidence']:.3f}")
        """
        self.model.eval()

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_sequence_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            confidence = torch.max(probabilities).item()
            prediction = torch.argmax(outputs.logits, dim=-1).item()

        stance_labels = {0: 'against', 1: 'neutral', 2: 'favor'}

        return {
            'text': text,
            'stance': stance_labels[prediction],
            'stance_id': prediction,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy().tolist()[0],
            'reliable': confidence > self.config.confidence_threshold
        }

    def save_model(self, filepath: str) -> None:
        """Save trained model and tokenizer to disk.

        Args:
            filepath: Path where to save the model files
        """
        save_path = Path(filepath)
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        self.logger.info(f"Model saved to {save_path}")

    def load_model(self, filepath: str) -> None:
        """Load trained model and tokenizer from disk.

        Args:
            filepath: Path containing saved model files
        """
        load_path = Path(filepath)

        self.model = BertForSequenceClassification.from_pretrained(load_path).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(load_path)

        self.logger.info(f"Model loaded from {load_path}")