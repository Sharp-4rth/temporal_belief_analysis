# Artificial vs Natural Discourse:
Validating Persuasion Predictors Beyond Debate Settings

---
## Summary
This project validates whether persuasion predictors from structured debates (e.g., Oxford Debates, r/CMV) generalize to natural online discussions. Using the r/PoliticalDiscussion corpus, a pipeline was built for topic and stance detection, stance timeline construction, change detection, and predictor analysis. Results show that some predictors (e.g., argument complexity, interplay, hedging) transfer with smaller effects, while others (e.g., politeness, evidence markers) behave differently, highlighting both overlaps and divergences between debate settings and real-world discourse.

---
## Pipeline Overview

1. **Topic Detection** – conversations are classified with a pretrained BART model into broad topical categories.  
2. **Stance Detection** – stance labels are assigned using BERTweet fine-tuned on SemEval-2016 (gold data) and further adapted on GPT-annotated Reddit data.  
3. **Timelines** – per-user stance trajectories are built across topics.  
4. **Change Detection** – belief shifts are identified using a persistence-based rule and CUSUM.  
5. **Window Extraction** – conversational windows (OP + reply paths) are collected around detected change points.  
6. **Predictor Analysis** – persuasion predictors (e.g., politeness, hedging, argument complexity, evidence markers, interplay) are measured and compared between changers and controls.    
---

## Installation

#### A Note On Notebooks
**Notebooks** are arranged chronologically. Earlier notebooks are kept to give a better picture of the development of the project even if they are outdated.

#### Memory Needs
To expore the data, run the study, or do anything of value with this project, you need to load the labeled `r/PoliticalDiscussion` corpus into memory. This takes up ~24GBs of memory, and for that reason, it's strongly advised you use `Google Colab` to explore the project unless of course there's other means of satisfying that hardware requirement.

### 1. Cloud

#### To reproduce the results:
1. Open `main.ipynb` notebook in `Google Colab` or other cloud environment
2. Execute the cells sequentially 

### 2. Local
If you want to explore/run parts of this project locally, 
Clone the repository:
```bash
  git clone https://git.cs.bham.ac.uk/projects-2024-25/lxt419.git
```
To install the dependencies, the requirements file is a good place to start:
```bash
  pip install -r requirements.txt
```
---
