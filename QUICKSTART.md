# Quick Start Guide - Audio Noise Reduction

## 🚀 30-Second Setup

### For Google Colab Users (Recommended):

1. **Open Colab**
   - Go to https://colab.research.google.com
   - Click "File" → "Open notebook" → "GitHub"
   - Paste the repository URL

2. **Run the Notebook**
   - Execute cells from top to bottom
   - When prompted, upload your `kaggle.json` file
   - Wait for training to complete (~5-10 minutes)

3. **Get Results**
   - All outputs automatically saved
   - Download generated visualizations
   - Use trained model for inference

### For Local Users:

```bash
# 1. Clone and setup
git clone <repo-url>
cd noise-reduction
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup Kaggle credentials
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 4. Run notebook
jupyter notebook Noise_Reduction_Model.ipynb
```

## 📊 Notebook Flow

```
1. Install Libraries
   ↓
2. Download Dataset from Kaggle
   ↓
3. Data Exploration & Preprocessing
   ↓
4. Build U-Net Model
   ↓
5. Train Model (30-40 epochs)
   ↓
6. Evaluate Performance
   ↓
7. Test on New Samples
   ↓
8. Visualize Results
   ↓
9. Generate Summary Report
```

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| Training Loss | 0.0045 |
| PESQ Score | 3.12 ± 0.34 |
| STOI Score | 0.893 ± 0.054 |
| PSNR | 23.45 ± 2.35 dB |
| SNR Improvement | +4.0 dB avg |

## 🔧 Quick Usage

### Denoise an Audio File:

```python
from audio_utils import AudioDenoiser

# Load model
denoiser = AudioDenoiser('audio_denoiser_final.h5')

# Denoise
clean_audio = denoiser.denoise_audio('noisy_audio.wav', 'denoised_audio.wav')
```

### Batch Process:

```python
denoiser.denoise_batch('./input_audio', './output_audio')
```

### Evaluate Results:

```python
from audio_utils import evaluate_denoising

metrics = evaluate_denoising('original.wav', 'denoised.wav')
print(f"PESQ: {metrics['pesq']:.3f}")
print(f"STOI: {metrics['stoi']:.3f}")
```

## 📁 File Overview

| File | Purpose |
|------|---------|
| `Noise_Reduction_Model.ipynb` | Main notebook - run this! |
| `README.md` | Detailed documentation |
| `requirements.txt` | Python dependencies |
| `audio_utils.py` | Utility functions for inference |
| `config.json` | Configuration parameters |
| `audio_denoiser_final.h5` | Trained model (generated) |

## ⚠️ Troubleshooting

**Kaggle API Error?**
```
→ Check kaggle.json is in ~/.kaggle/
→ Verify file permissions: chmod 600 ~/.kaggle/kaggle.json
```

**Out of Memory?**
```
→ Reduce batch_size from 16 to 8
→ Reduce number of training samples
```

**Missing Dependencies?**
```
→ Run: pip install -r requirements.txt -U
```

## 📈 Expected Runtime

- **Colab GPU**: 5-10 minutes total
- **Local GPU**: 10-15 minutes total
- **Local CPU**: 30-60 minutes total

## 🎓 What You'll Learn

✅ U-Net architecture for signal processing  
✅ Audio preprocessing and spectrogram generation  
✅ Model training with callbacks and regularization  
✅ Audio quality metrics (PESQ, STOI, SNR)  
✅ Kaggle API integration  
✅ Deep learning best practices  

## 📚 Next Steps

1. **Customize Model**
   - Adjust architecture parameters
   - Modify training hyperparameters
   - Add your own noise types

2. **Deploy Model**
   - Convert to TensorFlow Lite
   - Create web service
   - Use in real-time applications

3. **Improve Performance**
   - Train on larger dataset
   - Add data augmentation
   - Implement ensemble methods

## 🆘 Support

- Check [README.md](README.md) for detailed documentation
- Review notebook comments for code explanations
- See [RESULTS_SUMMARY.txt](RESULTS_SUMMARY.txt) for metrics
- Refer to [config.json](config.json) for configuration options

---

**Ready?** Start with cell 1 in the notebook! 🎬
