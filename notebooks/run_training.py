"""
Script to run all notebook cells sequentially and fix any errors.
This executes the entire training pipeline from the notebook.
"""
import json
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter
from time import time

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    train_test_split, 
    GridSearchCV, 
    cross_val_score,
    StratifiedKFold,
    learning_curve
)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score
)
from sklearn.utils import shuffle
from scipy.sparse import hstack, csr_matrix
from scipy.stats import uniform

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import project utilities
try:
    from preprocessing.advanced_preprocessing import clean_text, preprocess_article
    from utils.text_utils import get_text_statistics
    print("✓ Project utilities imported successfully")
except ImportError as e:
    print(f"Error importing project utilities: {e}")
    sys.exit(1)

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
RANDOM_STATE = 42

print("=" * 70)
print("STARTING TRAINING PIPELINE")
print("=" * 70)
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Project root: {project_root}")

# ============================================================================
# 1. Load and Explore Dataset
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: LOADING DATASET")
print("=" * 70)

dataset_path = project_root / "data" / "combined" / "combined_dataset_clean.json"

if not dataset_path.exists():
    print(f"Error: Dataset not found at {dataset_path}")
    sys.exit(1)

with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"✓ Loaded {len(dataset):,} articles from {dataset_path}")

# Analyze dataset
labels = [article.get("label", "").lower() for article in dataset]
label_counts = Counter(labels)

print(f"\nLabel Distribution:")
for label, count in label_counts.items():
    percentage = (count / len(dataset)) * 100
    print(f"  {label.upper()}: {count:,} ({percentage:.2f}%)")

# ============================================================================
# 2. Advanced Preprocessing
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: PREPROCESSING DATA")
print("=" * 70)

processed_articles = []
empty_count = 0

for article in dataset:
    label = article.get("label", "").lower()
    if label not in ["ai", "human"]:
        continue
    
    content = article.get("content", "").strip()
    if not content:
        empty_count += 1
        continue
    
    cleaned_content = clean_text(content)
    
    if cleaned_content and len(cleaned_content) > 20:
        processed_articles.append({
            "content": cleaned_content,
            "label": label
        })

print(f"✓ Processed {len(processed_articles):,} articles")
print(f"  Removed {len(dataset) - len(processed_articles):,} invalid/empty articles")

# Balance dataset
ai_articles = [a for a in processed_articles if a["label"] == "ai"]
human_articles = [a for a in processed_articles if a["label"] == "human"]

print(f"\nBefore balancing:")
print(f"  AI articles: {len(ai_articles):,}")
print(f"  Human articles: {len(human_articles):,}")

min_size = min(len(ai_articles), len(human_articles))
ai_articles = ai_articles[:min_size]
human_articles = human_articles[:min_size]

balanced_dataset = ai_articles + human_articles
balanced_dataset = shuffle(balanced_dataset, random_state=RANDOM_STATE)

print(f"\nAfter balancing:")
print(f"  AI articles: {len(ai_articles):,}")
print(f"  Human articles: {len(human_articles):,}")
print(f"  Total: {len(balanced_dataset):,}")

texts = [article["content"] for article in balanced_dataset]
labels = [article["label"] for article in balanced_dataset]

# ============================================================================
# 3. Feature Engineering
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 70)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

print(f"Label encoding:")
print(f"  Classes: {label_encoder.classes_}")
print(f"  Distribution: AI={np.sum(y==1):,}, Human={np.sum(y==0):,}")

# Split data
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, y, 
    test_size=0.2, 
    random_state=RANDOM_STATE, 
    stratify=y
)

print(f"\nTrain/Test Split:")
print(f"  Training: {len(X_train_texts):,} samples")
print(f"  Testing: {len(X_test_texts):,} samples")

# Word-level features
print("\nCreating word-level features...")
word_vectorizer = TfidfVectorizer(
    max_features=25000,
    ngram_range=(1, 3),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
    analyzer='word',
    lowercase=True,
    strip_accents='unicode',
    token_pattern=r'(?u)\b\w+\b'
)

start_time = time()
X_train_word = word_vectorizer.fit_transform(X_train_texts)
X_test_word = word_vectorizer.transform(X_test_texts)
print(f"✓ Word features created in {time() - start_time:.2f} seconds")
print(f"  Training shape: {X_train_word.shape}")
print(f"  Vocabulary size: {len(word_vectorizer.vocabulary_):,}")

# Character-level features
print("\nCreating character-level features...")
char_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 6),
    max_features=35000,
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
    lowercase=True,
    strip_accents='unicode'
)

start_time = time()
X_train_char = char_vectorizer.fit_transform(X_train_texts)
X_test_char = char_vectorizer.transform(X_test_texts)
print(f"✓ Character features created in {time() - start_time:.2f} seconds")
print(f"  Training shape: {X_train_char.shape}")

# Combine features
X_train = hstack([X_train_word, X_train_char])
X_test = hstack([X_test_word, X_test_char])

print(f"\n✓ Combined feature matrix:")
print(f"  Training shape: {X_train.shape}")
print(f"  Test shape: {X_test.shape}")
print(f"  Total features: {X_train.shape[1]:,}")

# ============================================================================
# 4. Model Training
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: MODEL TRAINING")
print("=" * 70)

# Logistic Regression baseline
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE,
    C=1.0,
    solver='lbfgs',
    n_jobs=-1,
    class_weight='balanced'
)

start_time = time()
lr_model.fit(X_train, y_train)
lr_train_time = time() - start_time

y_pred_lr = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)

print(f"✓ Training completed in {lr_train_time:.2f} seconds")
print(f"  Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
print(f"  F1-Score: {lr_f1:.4f}")

# SVM with hyperparameter tuning
print("\n" + "-" * 70)
print("Hyperparameter Tuning for SVM")
print("-" * 70)

cv_folds = 5
skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

# Sample for grid search if dataset is large
if X_train.shape[0] > 10000:
    sample_size = 10000
    sample_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
    X_train_sample = X_train[sample_indices]
    y_train_sample = y_train[sample_indices]
    print(f"Using {sample_size:,} samples for grid search...")
else:
    X_train_sample = X_train
    y_train_sample = y_train

param_grid = {
    'C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
    'class_weight': [None, 'balanced']
}

print(f"Grid search parameters: {param_grid}")
print(f"CV folds: {cv_folds}")

svm_base = SVC(
    kernel='linear',
    probability=True,
    random_state=RANDOM_STATE,
    max_iter=10000
)

print("Running grid search (this may take several minutes)...")
start_time = time()

grid_search = GridSearchCV(
    svm_base,
    param_grid,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    refit=True
)

grid_search.fit(X_train_sample, y_train_sample)
grid_time = time() - start_time

print(f"\n✓ Grid search completed in {grid_time/60:.2f} minutes")
print(f"  Best parameters: {grid_search.best_params_}")
print(f"  Best CV score: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")

# Train final SVM
print("\nTraining final SVM model on full training set...")
best_params = grid_search.best_params_

final_svm = SVC(
    kernel='linear',
    C=best_params['C'],
    class_weight=best_params['class_weight'],
    probability=True,
    random_state=RANDOM_STATE,
    max_iter=10000
)

start_time = time()
final_svm.fit(X_train, y_train)
svm_train_time = time() - start_time

y_pred_svm = final_svm.predict(X_test)
y_proba_svm = final_svm.predict_proba(X_test)[:, 1]

svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm)

print(f"✓ Training completed in {svm_train_time:.2f} seconds")
print(f"  Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
print(f"  F1-Score: {svm_f1:.4f}")

# Additional models
print("\n" + "-" * 70)
print("Training additional models...")
print("-" * 70)

models = {}
results = {}

# Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=50,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0
)
start_time = time()
rf_model.fit(X_train, y_train)
rf_time = time() - start_time
y_pred_rf = rf_model.predict(X_test)
models['Random Forest'] = rf_model
results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'f1': f1_score(y_test, y_pred_rf),
    'time': rf_time
}
print(f"  Accuracy: {results['Random Forest']['accuracy']:.4f} ({results['Random Forest']['accuracy']*100:.2f}%)")

# Gradient Boosting
print("Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    verbose=0
)
start_time = time()
gb_model.fit(X_train, y_train)
gb_time = time() - start_time
y_pred_gb = gb_model.predict(X_test)
models['Gradient Boosting'] = gb_model
results['Gradient Boosting'] = {
    'accuracy': accuracy_score(y_test, y_pred_gb),
    'f1': f1_score(y_test, y_pred_gb),
    'time': gb_time
}
print(f"  Accuracy: {results['Gradient Boosting']['accuracy']:.4f} ({results['Gradient Boosting']['accuracy']*100:.2f}%)")

results['SVM'] = {'accuracy': svm_accuracy, 'f1': svm_f1, 'time': svm_train_time}
results['Logistic Regression'] = {'accuracy': lr_accuracy, 'f1': lr_f1, 'time': lr_train_time}

# ============================================================================
# 5. Evaluation
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: COMPREHENSIVE EVALUATION")
print("=" * 70)

precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred_svm, average=None, labels=[0, 1]
)

print(f"\nPer-Class Metrics:")
print(f"  Human (Class 0): Precision={precision[0]:.4f}, Recall={recall[0]:.4f}, F1={f1[0]:.4f}")
print(f"  AI (Class 1): Precision={precision[1]:.4f}, Recall={recall[1]:.4f}, F1={f1[1]:.4f}")

cm = confusion_matrix(y_test, y_pred_svm)
print(f"\nConfusion Matrix:")
print(f"  TN (Human→Human): {cm[0][0]:,}")
print(f"  FP (Human→AI):   {cm[0][1]:,}")
print(f"  FN (AI→Human):   {cm[1][0]:,}")
print(f"  TP (AI→AI):      {cm[1][1]:,}")

# Cross-validation
print("\nRunning cross-validation...")
cv_scores = cross_val_score(
    final_svm, 
    X_train, 
    y_train, 
    cv=skf, 
    scoring='accuracy',
    n_jobs=-1
)

print(f"Cross-Validation Results:")
print(f"  Mean: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
print(f"  Std: {cv_scores.std():.4f}")

# ============================================================================
# 6. Save Models
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: SAVING MODELS")
print("=" * 70)

model_save_dir = project_root / "models" / "saved_models"
model_save_dir.mkdir(parents=True, exist_ok=True)

# Save vectorizers
with open(model_save_dir / "word_vectorizer.pkl", "wb") as f:
    pickle.dump(word_vectorizer, f)
print("✓ Saved word_vectorizer.pkl")

with open(model_save_dir / "char_vectorizer.pkl", "wb") as f:
    pickle.dump(char_vectorizer, f)
print("✓ Saved char_vectorizer.pkl")

# Save models
with open(model_save_dir / "svm_model.pkl", "wb") as f:
    pickle.dump(final_svm, f)
print("✓ Saved svm_model.pkl")

with open(model_save_dir / "logreg_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)
print("✓ Saved logreg_model.pkl")

# Save label encoder
with open(model_save_dir / "label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("✓ Saved label_encoder.pkl")

# ============================================================================
# 7. Final Summary
# ============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"\n📊 Dataset:")
print(f"  Total samples: {len(balanced_dataset):,}")
print(f"  Training: {len(X_train_texts):,}, Test: {len(X_test_texts):,}")

print(f"\n🔧 Features:")
print(f"  Word: {X_train_word.shape[1]:,}, Char: {X_train_char.shape[1]:,}")
print(f"  Total: {X_train.shape[1]:,}")

print(f"\n🎯 Model Performance:")
for model_name, metrics in results.items():
    print(f"  {model_name}: {metrics['accuracy']*100:.2f}% accuracy")

print(f"\n✅ Cross-Validation:")
print(f"  Mean: {cv_scores.mean()*100:.2f}%")

if svm_accuracy >= 0.99:
    print(f"\n🎉 SUCCESS! Achieved 99%+ accuracy: {svm_accuracy*100:.2f}%")
else:
    print(f"\n⚠️  Current accuracy: {svm_accuracy*100:.2f}% (Target: 99%+)")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)


