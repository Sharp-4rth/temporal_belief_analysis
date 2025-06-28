import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import torch


@dataclass
class ProjectConfig:
    """Central configuration for the entire project.

    This class manages all configuration parameters, file paths, and model
    hyperparameters in one place for easy maintenance and reproducibility.

    Attributes:
        project_name: Name of the project for logging and outputs
        version: Current project version for tracking
        data_dir: Root directory for all data files
        model_dir: Directory for saved models and checkpoints
        results_dir: Directory for output results and reports
        bert_model_name: Pre-trained BERT model identifier
        lstm_hidden_size: Hidden dimension size for LSTM networks
        max_sequence_length: Maximum input sequence length for models
        batch_size: Training and inference batch size
        learning_rate: Learning rate for model training
        confidence_threshold: Minimum confidence for reliable predictions
        random_seed: Random seed for reproducibility
    """
    # Project metadata
    project_name: str = "temporal-belief-analysis"
    version: str = "1.0.0"

    # Directory structure
    data_dir: Path = Path("data")
    model_dir: Path = Path("models/saved")
    results_dir: Path = Path("reports/results")
    figures_dir: Path = Path("reports/figures")

    # Model configurations
    bart_model_name: str ="facebook/bart-large-mnli"
    bert_model_name: str = "bert-base-uncased"
    lstm_hidden_size: int = 128
    max_sequence_length: int = 512
    temporal_window_days: int = 30
    prediction_horizon_days: int = 7

    # Training parameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    confidence_threshold: float = 0.8

    # Analysis parameters
    event_window_days: int = 7
    change_threshold: float = 0.7
    significance_level: float = 0.05

    # System configuration
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        """Create necessary directories after initialization."""
        for directory in [self.data_dir, self.model_dir, self.results_dir, self.figures_dir]:
            directory.mkdir(parents=True, exist_ok=True)


# Classification labels
POLITICAL_TOPICS = [
    'healthcare policy',
    'immigration policy',
    'economic policy',
    'gun rights and control',
    'abortion and reproductive rights',
    'climate change and energy policy',
    'foreign policy and defense',
    'civil rights and social issues',
    'taxation and government spending',
    'education policy',
    'criminal justice and policing',
    'voting rights and elections',
    'political figures and campaigns',
    'congressional politics',
    'electoral politics',
    'political parties and ideology',
    'media and political commentary'
]


TOPIC_STANCE_HYPOTHESES = {
    'abortion and reproductive rights': {
        'strongly_favor': [
            'The author strongly supports abortion rights and reproductive freedom',
            'This comment advocates for unrestricted access to abortion services',
            'The author expresses strong pro-choice activist viewpoints'
        ],
        'moderately_favor': [
            'The author moderately supports abortion rights with some limitations',
            'This comment leans toward pro-choice but accepts some restrictions',
            'The author supports reproductive rights with reasonable regulations'
        ],
        'neutral': [
            'This comment provides neutral information about abortion policy',
            'The author presents balanced views on reproductive rights',
            'This text discusses abortion without taking a clear stance'
        ],
        'moderately_against': [
            'The author has concerns about abortion but accepts some circumstances',
            'This comment leans pro-life but allows for exceptions',
            'The author moderately opposes abortion with some flexibility'
        ],
        'strongly_against': [
            'The author strongly opposes abortion and supports pro-life positions',
            'This comment advocates for complete protection of unborn life',
            'The author expresses strong anti-abortion activist viewpoints'
        ]
    },

    'gun rights and control': {
        'strongly_favor': [
            'The author strongly supports gun rights and Second Amendment freedoms',
            'This comment advocates for unrestricted firearm ownership rights',
            'The author expresses strong pro-gun activist viewpoints'
        ],
        'moderately_favor': [
            'The author supports gun rights but accepts some safety regulations',
            'This comment leans pro-gun but allows reasonable restrictions',
            'The author favors gun ownership with common-sense limitations'
        ],
        'neutral': [
            'This comment provides neutral information about gun policy',
            'The author presents balanced views on firearm regulations',
            'This text discusses guns without taking a clear stance'
        ],
        'moderately_against': [
            'The author supports gun control but respects some ownership rights',
            'This comment leans toward restrictions but allows some gun rights',
            'The author moderately favors gun control with exceptions'
        ],
        'strongly_against': [
            'The author strongly supports gun control and firearm restrictions',
            'This comment advocates for strict limitations on gun ownership',
            'The author expresses strong gun control activist viewpoints'
        ]
    },

    'immigration': {
        'strongly_favor': [
            'The author strongly supports immigration and open border policies',
            'This comment advocates for expanded immigration and refugee rights',
            'The author expresses strong pro-immigration activist viewpoints'
        ],
        'moderately_favor': [
            'The author supports immigration with reasonable processing systems',
            'This comment leans pro-immigration but accepts some controls',
            'The author favors welcoming immigrants with proper procedures'
        ],
        'neutral': [
            'This comment provides neutral information about immigration policy',
            'The author presents balanced views on immigration reform',
            'This text discusses immigration without taking a clear stance'
        ],
        'moderately_against': [
            'The author has concerns about immigration but supports legal pathways',
            'This comment leans toward restrictions but allows controlled immigration',
            'The author moderately opposes unchecked immigration'
        ],
        'strongly_against': [
            'The author strongly opposes immigration and supports border restrictions',
            'This comment advocates for strict immigration controls and enforcement',
            'The author expresses strong anti-immigration viewpoints'
        ]
    },

    'healthcare': {
        'strongly_favor': [
            'The author strongly supports universal healthcare and single-payer systems',
            'This comment advocates for government-provided healthcare for all citizens',
            'The author expresses strong support for socialized medicine and healthcare as a right'
        ],
        'moderately_favor': [
            'The author supports expanded government healthcare with some private options',
            'This comment leans toward universal coverage but accepts mixed public-private systems',
            'The author favors government healthcare expansion with pragmatic compromises'
        ],
        'neutral': [
            'This comment provides neutral information about healthcare policy without taking sides',
            'The author presents balanced views on different healthcare systems',
            'This text discusses healthcare options without advocating for specific approaches'
        ],
        'moderately_against': [
            'The author prefers market-based healthcare but accepts some government safety nets',
            'This comment leans toward private healthcare while acknowledging need for limited government role',
            'The author moderately opposes universal healthcare but supports targeted government programs'
        ],
        'strongly_against': [
            'The author strongly opposes government healthcare and advocates for free-market solutions',
            'This comment advocates against socialized medicine and for private healthcare systems',
            'The author expresses strong opposition to universal healthcare and government involvement'
        ]
    },

    'climate change': {
        'strongly_favor': [
            'The author strongly supports aggressive climate action and environmental protection',
            'This comment advocates for immediate action on climate change',
            'The author expresses strong environmental activist viewpoints'
        ],
        'moderately_favor': [
            'The author supports climate action with balanced economic considerations',
            'This comment leans toward environmental protection with practical limits',
            'The author favors climate policies with gradual implementation'
        ],
        'neutral': [
            'This comment provides neutral information about climate policy',
            'The author presents balanced views on environmental issues',
            'This text discusses climate without taking a clear stance'
        ],
        'moderately_against': [
            'The author questions some climate policies but accepts environmental concerns',
            'This comment leans skeptical but allows for some climate action',
            'The author moderately opposes aggressive climate regulations'
        ],
        'strongly_against': [
            'The author strongly opposes climate regulations and questions climate science',
            'This comment advocates against environmental restrictions on business',
            'The author expresses strong climate skepticism and anti-regulation views'
        ]
    }
}

# General fallback hypotheses for unknown topics
GENERAL_STANCE_HYPOTHESES = {
    'strongly_favor': [
        'The author strongly supports the main position being discussed',
        'This comment advocates strongly for the topic being debated',
        'The author expresses strong support for the primary viewpoint'
    ],
    'moderately_favor': [
        'The author moderately supports the main position with some reservations',
        'This comment leans toward support but with qualifications',
        'The author generally favors the position but acknowledges concerns'
    ],
    'neutral': [
        'This comment provides neutral information without taking sides',
        'The author presents balanced views on the topic',
        'This text discusses the issue without clear position advocacy'
    ],
    'moderately_against': [
        'The author has concerns about the main position but shows some flexibility',
        'This comment leans against the primary viewpoint with some exceptions',
        'The author moderately opposes the position but acknowledges some merit'
    ],
    'strongly_against': [
        'The author strongly opposes the main position being discussed',
        'This comment advocates strongly against the topic being debated',
        'The author expresses strong opposition to the primary viewpoint'
    ]
}