# DPN-CycleGAN-based-Image-Style-Transfer

## Key Highlights of the Project:

### 1. **Improved Architecture**
   - Replaced **ResNet** with **Dual Path Networks (DPN)** in the CycleGAN generator, combining the strengths of **ResNet** and **DenseNet** for better feature reuse and exploration.

### 2. **Enhanced Content Retention**
   - Integrated **Identity Loss** into the CycleGAN loss function to preserve key content features during style transformation.

### 3. **Quantitative Success**
   - Achieved superior results compared to standard CycleGAN, with notable improvements in metrics like **SSIM** and **PSNR**, reflecting better structural similarity and reduced noise in generated images.

### 4. **Implementation Tools**
   - Leveraged **PyTorch** for model development and applied advanced techniques like:
     - Hyperparameter tuning
     - Evaluation metrics (**SSIM**, **PSNR**)
     - Optimized learning rates for faster convergence

### 5. **Dataset Utilized**
   - The **Van Gogh-to-photo** dataset, featuring hundreds of Van Gogh paintings and natural scenery, served as a benchmark for testing and validating the model's performance.

---

This project demonstrated how integrating **Dual Path Networks** and innovative loss functions can enhance both the quality and efficiency of image style transfer, paving the way for real-time applications in art generation, photo editing, and domain adaptation.
