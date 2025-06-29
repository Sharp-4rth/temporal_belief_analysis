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
            'The author strongly supports gun control and firearm restrictions',
            'This comment advocates for strict limitations on gun ownership',
            'The author expresses strong gun control activist viewpoints'
        ],
        'moderately_favor': [
            'The author supports gun control but respects some ownership rights',
            'This comment leans toward restrictions but allows some gun rights',
            'The author moderately favors gun control with exceptions'
        ],
        'neutral': [
            'This comment provides neutral information about gun policy',
            'The author presents balanced views on firearm regulations',
            'This text discusses guns without taking a clear stance'
        ],
        'moderately_against': [
            'The author supports gun rights but accepts some safety regulations',
            'This comment leans pro-gun but allows reasonable restrictions',
            'The author favors gun ownership with common-sense limitations'
        ],
        'strongly_against': [
            'The author strongly supports gun rights and Second Amendment freedoms',
            'This comment advocates for unrestricted firearm ownership rights',
            'The author expresses strong pro-gun activist viewpoints'
        ]
    },

    'immigration policy': {
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

    'healthcare policy': {
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

    'climate change and energy policy': {
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
    },

    'economic policy': {
        'strongly_favor': [
            'The author strongly supports progressive economic policies and wealth redistribution',
            'This comment advocates for extensive government intervention in the economy',
            'The author expresses strong support for socialist economic principles and workers rights'
        ],
        'moderately_favor': [
            'The author supports some government economic intervention with market elements',
            'This comment leans toward regulated capitalism with social safety nets',
            'The author favors mixed economic policies balancing government and market forces'
        ],
        'neutral': [
            'This comment provides neutral information about economic policy options',
            'The author presents balanced views on different economic approaches',
            'This text discusses economic issues without advocating specific policies'
        ],
        'moderately_against': [
            'The author prefers free market approaches but accepts some government regulation',
            'This comment leans toward capitalism while acknowledging need for limited intervention',
            'The author moderately opposes extensive government economic control'
        ],
        'strongly_against': [
            'The author strongly supports free market capitalism and opposes government intervention',
            'This comment advocates for minimal government economic regulation and maximum market freedom',
            'The author expresses strong libertarian or conservative economic viewpoints'
        ]
    },

    'foreign policy and defense': {
        'strongly_favor': [
            'The author strongly supports diplomatic solutions and opposes military intervention',
            'This comment advocates for peaceful foreign relations and non-intervention',
            'The author expresses strong anti-war and dovish foreign policy viewpoints'
        ],
        'moderately_favor': [
            'The author prefers diplomatic solutions but accepts some military necessity',
            'This comment leans toward non-intervention while acknowledging security needs',
            'The author favors peaceful engagement while supporting defensive capabilities'
        ],
        'neutral': [
            'This comment provides neutral information about foreign policy options',
            'The author presents balanced views on international relations',
            'This text discusses foreign policy without advocating specific approaches'
        ],
        'moderately_against': [
            'The author supports active foreign engagement with diplomatic preferences',
            'This comment leans toward international involvement but favors peaceful solutions',
            'The author moderately supports international leadership with restraint'
        ],
        'strongly_against': [
            'The author strongly supports an assertive foreign policy and strong military presence',
            'This comment advocates for military intervention and international leadership',
            'The author expresses strong support for hawkish foreign policy positions'
        ]
    },

    'civil rights and social issues': {
        'strongly_favor': [
            'The author strongly supports expanded civil rights and social justice initiatives',
            'This comment advocates for progressive social policies and equality measures',
            'The author expresses strong support for minority rights and social reform'
        ],
        'moderately_favor': [
            'The author supports civil rights with pragmatic implementation approaches',
            'This comment leans toward social progress but accepts gradual change',
            'The author favors civil rights advancement with reasonable compromises'
        ],
        'neutral': [
            'This comment provides neutral information about civil rights issues',
            'The author presents balanced views on social policy questions',
            'This text discusses civil rights without taking clear advocacy positions'
        ],
        'moderately_against': [
            'The author has concerns about some civil rights policies but supports basic equality',
            'This comment leans toward traditional approaches while accepting some social change',
            'The author moderately opposes rapid social change but supports fundamental rights'
        ],
        'strongly_against': [
            'The author strongly opposes expanded civil rights policies and social justice initiatives',
            'This comment advocates for traditional social structures and opposes progressive change',
            'The author expresses strong conservative or traditionalist social viewpoints'
        ]
    },

    'taxation and government spending': {
        'strongly_favor': [
            'The author strongly supports higher taxes and increased government spending',
            'This comment advocates for progressive taxation and expanded public programs',
            'The author expresses strong support for government-funded social services'
        ],
        'moderately_favor': [
            'The author supports reasonable tax increases for important public services',
            'This comment leans toward higher taxes but with fiscal responsibility',
            'The author favors balanced approaches to taxation and spending'
        ],
        'neutral': [
            'This comment provides neutral information about tax and spending policies',
            'The author presents balanced views on fiscal policy options',
            'This text discusses taxation without advocating specific approaches'
        ],
        'moderately_against': [
            'The author prefers lower taxes but accepts some government spending',
            'This comment leans toward fiscal conservatism while acknowledging public needs',
            'The author moderately opposes tax increases but supports essential services'
        ],
        'strongly_against': [
            'The author strongly opposes tax increases and supports reduced government spending',
            'This comment advocates for minimal taxation and limited government programs',
            'The author expresses strong fiscal conservative or libertarian viewpoints'
        ]
    },

    'education policy': {
        'strongly_favor': [
            'The author strongly supports increased public education funding and progressive education policies',
            'This comment advocates for public education expansion and teacher support',
            'The author expresses strong support for educational equity and access initiatives'
        ],
        'moderately_favor': [
            'The author supports public education improvements with balanced funding approaches',
            'This comment leans toward public education but accepts some choice options',
            'The author favors education reform with strong public school focus'
        ],
        'neutral': [
            'This comment provides neutral information about education policy options',
            'The author presents balanced views on different educational approaches',
            'This text discusses education without advocating specific policy positions'
        ],
        'moderately_against': [
            'The author supports some school choice while maintaining public education',
            'This comment leans toward market-based education but accepts public schools',
            'The author moderately supports education alternatives but funds public schools'
        ],
        'strongly_against': [
            'The author strongly supports school choice and opposes public education monopolies',
            'This comment advocates for privatization and market-based education solutions',
            'The author expresses strong support for vouchers and charter schools over public education'
        ]
    },

    'criminal justice and policing': {
        'strongly_favor': [
            'The author strongly supports police reform and criminal justice system changes',
            'This comment advocates for reducing police funding and addressing systemic issues',
            'The author expresses strong support for criminal justice reform and police accountability'
        ],
        'moderately_favor': [
            'The author supports police reform with balanced law enforcement needs',
            'This comment leans toward criminal justice changes but maintains public safety priorities',
            'The author favors police accountability while supporting reformed law enforcement'
        ],
        'neutral': [
            'This comment provides neutral information about criminal justice issues',
            'The author presents balanced views on policing and justice system questions',
            'This text discusses criminal justice without taking clear advocacy positions'
        ],
        'moderately_against': [
            'The author supports law enforcement but accepts some accountability measures',
            'This comment leans toward backing police while acknowledging need for some reforms',
            'The author moderately supports police while accepting minor improvements'
        ],
        'strongly_against': [
            'The author strongly supports law enforcement and opposes police reform initiatives',
            'This comment advocates for backing the police and maintaining current justice systems',
            'The author expresses strong support for law and order and opposition to police criticism'
        ]
    },

    'voting rights and elections': {
        'strongly_favor': [
            'The author strongly supports expanded voting access and making voting easier',
            'This comment advocates for removing barriers to voting and increasing electoral participation',
            'The author expresses strong support for voting rights expansion and democratic access'
        ],
        'moderately_favor': [
            'The author supports voting access improvements with reasonable security measures',
            'This comment leans toward expanded voting rights but accepts some verification requirements',
            'The author favors increased voting access while maintaining election integrity'
        ],
        'neutral': [
            'This comment provides neutral information about voting and election policies',
            'The author presents balanced views on electoral system questions',
            'This text discusses voting rights without advocating specific positions'
        ],
        'moderately_against': [
            'The author prioritizes election security while supporting basic voting access',
            'This comment leans toward voting safeguards but accepts fundamental voting rights',
            'The author moderately supports election integrity measures but allows broad access'
        ],
        'strongly_against': [
            'The author strongly supports voting restrictions and strict election security measures',
            'This comment advocates for limiting voting access to prevent fraud',
            'The author expresses strong support for voter ID requirements and election oversight'
        ]
    },

    'political figures and campaigns': {
        'strongly_favor': [
            'The author strongly supports liberal/progressive political figures and their campaigns',
            'This comment advocates enthusiastically for Democratic candidates and progressive policies',
            'The author expresses strong endorsement for left-leaning political figures'
        ],
        'moderately_favor': [
            'The author generally supports liberal political figures with some reservations',
            'This comment leans toward Democratic candidates while acknowledging some concerns',
            'The author favors progressive political figures but maintains some critical perspective'
        ],
        'neutral': [
            'This comment provides neutral information about political figures across the spectrum',
            'The author presents balanced views without clear partisan endorsement or opposition',
            'This text discusses political figures without taking ideological positions'
        ],
        'moderately_against': [
            'The author has concerns about conservative political figures but acknowledges some positives',
            'This comment leans against Republican candidates while recognizing some merit',
            'The author moderately opposes right-leaning political figures but maintains some respect'
        ],
        'strongly_against': [
            'The author strongly opposes conservative political figures and their campaign positions',
            'This comment advocates against Republican candidates and criticizes conservative policies',
            'The author expresses strong opposition to right-wing political figures'
        ]
    },

    'congressional politics': {
        'strongly_favor': [
            'The author strongly supports progressive congressional actions and liberal legislative approaches',
            'This comment advocates for Democratic congressional positions and left-leaning legislative strategy',
            'The author expresses strong approval of progressive congressional leadership and decisions'
        ],
        'moderately_favor': [
            'The author generally supports liberal congressional approaches with some concerns',
            'This comment leans toward progressive legislative positions but notes some issues',
            'The author favors Democratic congressional action while maintaining some criticism'
        ],
        'neutral': [
            'This comment provides neutral information about congressional activities across parties',
            'The author presents balanced views on legislative matters without clear partisan positions',
            'This text discusses congressional politics without ideological advocacy or opposition'
        ],
        'moderately_against': [
            'The author has concerns about conservative congressional actions but sees some merit',
            'This comment leans against Republican legislative approaches while acknowledging some value',
            'The author moderately opposes conservative congressional decisions but respects the process'
        ],
        'strongly_against': [
            'The author strongly opposes conservative congressional actions and right-wing legislative approaches',
            'This comment advocates against Republican congressional positions and criticizes conservative leadership',
            'The author expresses strong disapproval of right-wing congressional decisions and strategy'
        ]
    },

    'electoral politics': {
        'strongly_favor': [
            'The author strongly supports democratic reforms and progressive electoral changes',
            'This comment advocates for electoral reforms that increase democratic participation',
            'The author expresses strong support for progressive electoral democracy and voting expansion'
        ],
        'moderately_favor': [
            'The author generally supports democratic processes with some progressive reform concerns',
            'This comment leans toward electoral improvements that enhance democratic participation',
            'The author favors electoral democracy while seeking progressive improvements'
        ],
        'neutral': [
            'This comment provides neutral information about electoral processes and outcomes',
            'The author presents balanced views on democratic systems without clear ideological advocacy',
            'This text discusses electoral politics without taking partisan positions'
        ],
        'moderately_against': [
            'The author has concerns about progressive electoral changes but supports democratic principles',
            'This comment leans toward traditional electoral systems while maintaining democratic values',
            'The author moderately opposes electoral reforms but accepts democratic governance'
        ],
        'strongly_against': [
            'The author strongly opposes progressive electoral reforms and supports traditional systems',
            'This comment advocates against electoral changes that increase democratic participation',
            'The author expresses strong support for conservative electoral approaches'
        ]
    },

    'political parties and ideology': {
        'strongly_favor': [
            'The author strongly supports the Democratic Party and liberal ideological positions',
            'This comment advocates enthusiastically for progressive politics and left-leaning ideology',
            'The author expresses strong Democratic party loyalty and liberal ideological commitment'
        ],
        'moderately_favor': [
            'The author generally supports the Democratic Party with some independent thinking',
            'This comment leans toward liberal positions but maintains some ideological flexibility',
            'The author favors progressive ideology while accepting some moderate compromises'
        ],
        'neutral': [
            'This comment provides neutral information about political parties and ideologies',
            'The author presents balanced views without clear partisan or ideological bias',
            'This text discusses political parties without advocating specific partisan positions'
        ],
        'moderately_against': [
            'The author has concerns about the Republican Party but acknowledges some conservative merit',
            'This comment leans against conservative politics but accepts some right-leaning value',
            'The author moderately opposes conservative ideology but respects some traditional views'
        ],
        'strongly_against': [
            'The author strongly opposes the Republican Party and conservative ideological positions',
            'This comment advocates against conservative politics and right-wing ideology',
            'The author expresses strong opposition to Republican party politics and conservative thinking'
        ]
    },

    'media and political commentary': {
        'strongly_favor': [
            'The author strongly supports liberal media sources and progressive political commentary',
            'This comment advocates for the credibility of left-leaning media coverage',
            'The author expresses strong trust in progressive media outlets and liberal commentary'
        ],
        'moderately_favor': [
            'The author generally trusts liberal media sources while noting some concerns',
            'This comment leans toward supporting progressive media coverage but maintains some skepticism',
            'The author favors left-leaning political commentary while seeking additional perspectives'
        ],
        'neutral': [
            'This comment provides neutral assessment of media coverage across the political spectrum',
            'The author presents balanced views on media credibility without clear ideological bias',
            'This text discusses media and commentary without advocating specific partisan positions'
        ],
        'moderately_against': [
            'The author has concerns about conservative media bias but recognizes some value',
            'This comment leans toward skepticism of right-wing media while acknowledging some credibility',
            'The author moderately criticizes conservative political commentary but accepts some merit'
        ],
        'strongly_against': [
            'The author strongly opposes conservative media sources and rejects right-wing political commentary',
            'This comment advocates against conservative media credibility and right-wing commentary reliability',
            'The author expresses strong distrust of conservative media coverage and right-wing commentary bias'
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