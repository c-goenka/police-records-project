# California Police Records Classification

This project tackles the challenge of automatically classifying California police records into different document types.

## The Problem

Police records come in various forms—incident reports, arrest records, citations, administrative documents, etc. Being able to automatically categorize these helps with data organization and makes records more accessible for analysis.

## Methods Used

### 1. SetFit (Few-Shot Learning)
[SetFit](https://github.com/huggingface/setfit) is designed exactly for scenarios where you don't have tons of labeled examples. It uses sentence transformers to create embeddings and then fine-tunes them with contrastive learning. It can achieve strong performance with just a handful of examples per class.

### 2. Embeddings + Classical Classifiers
Generate embeddings using [sentence-transformers](https://www.sbert.net/) (specifically the `all-mpnet-base-v2` model), then train standard classifiers on top - I tested three:
- **Logistic Regression**
- **Random Forest**
- **SVM**

Models are well-understood, train quickly, and are easy to interpret compared to deep learning models.

### 3. Clustering Analysis
Unsupervised exploration using [K-means clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means) and [UMAP](https://umap-learn.readthedocs.io/) for visualization.

## Why These Methods?

- Works well with limited training data
- Can handle the natural language variation in official documents
- Doesn't require massive computational resources
- Gives interpretable results

Transformer-based embeddings capture semantic meaning really well, which is crucial when documents might express the same concept in different ways. Combining them with either few-shot learning (SetFit) or classical ML (logistic regression, etc.) gives a balance in performance and practicality.

## Setup

```bash
git clone https://github.com/c-goenka/police-records-project.git
cd police-records-project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Notebooks

- `eda.ipynb` - Exploratory data analysis and class distributions
- `setfit.ipynb` - Few-shot classification with SetFit
- `embeddings_classifier.ipynb` - Embedding generation + traditional classifiers
- `clustering_analysis.ipynb` - Unsupervised clustering and visualization
- `comparison_analysis.ipynb` - Performance comparison across methods
