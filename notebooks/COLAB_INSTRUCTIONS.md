# How to Run the Notebook in Google Colab

## Quick Start Guide

### Step 1: Open the Notebook in Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **"File" → "Upload notebook"**
3. Upload `notebooks/advanced_training_colab.ipynb`
   - OR click **"File" → "New notebook"** and copy all cells from the Colab notebook

### Step 2: Prepare Your Dataset

Make sure you have `combined_dataset_clean.json` ready on your computer.

### Step 3: Run the Notebook

1. **Run the first cell** - This installs required packages
2. **Run the second cell** - This imports all libraries
3. **Run the third cell** - This defines utility functions
4. **Run the upload cell** - This will show a file upload button
   - Click "Choose Files" and select your `combined_dataset_clean.json`
   - Wait for upload to complete
5. **Run all remaining cells** - Click "Runtime" → "Run all" or run cells one by one

### Step 4: Download Models

After training completes, the last cell will automatically download a zip file containing all trained models:
- `word_vectorizer.pkl`
- `char_vectorizer.pkl`
- `svm_model.pkl`
- `logreg_model.pkl`
- `label_encoder.pkl`

## What's Different in Colab Version?

✅ **All utilities included inline** - No need for project folder structure  
✅ **File upload widget** - Easy dataset upload  
✅ **Automatic package installation** - Everything installs automatically  
✅ **Model download** - Models automatically zipped and downloaded  
✅ **Optimized for Colab** - Works perfectly in Google's environment  

## Tips

- **Use GPU/TPU**: Go to "Runtime" → "Change runtime type" → Select GPU for faster training
- **Save your work**: Colab sessions expire, so download models immediately after training
- **Large datasets**: If your dataset is very large (>100MB), consider using Google Drive instead of direct upload

## Troubleshooting

- **Upload fails**: Try uploading to Google Drive first, then mount Drive in Colab
- **Out of memory**: Reduce dataset size or use a smaller sample for grid search
- **Session disconnects**: Enable "Runtime" → "Change runtime type" → "High-RAM" if available

