# 🔊 Noise Addition & Removal in Digital Images

## 📘 **Digital Image Processing (DIP) College Lab Project**

---

## 📋 **Project Objective**

To implement **noise models** (Salt & Pepper, Gaussian, Uniform) and **denoising filters** (Mean, Median, Gaussian, Bilateral) with **PSNR evaluation** for image quality assessment.

---

## 🔬 **Experiments Implemented**

| **S.No** | **Noise Type** | **Denoising Methods** | **PSNR Analysis** |
|----------|----------------|----------------------|-------------------|
| 1 | **Salt & Pepper (5%)** | Mean(5x5), Median(5x5), Gaussian(5x5), Bilateral | ✅ Comparison table |
| 2 | **Gaussian (σ=25)** | Mean(5x5), Median(5x5), Gaussian(5x5), Bilateral | ✅ Comparison table |
| 3 | **Uniform (-50,+50)** | Visualization only | - |
| 4 | **Kernel Size Study** | Median: 3x3,5x5,7x7,9x9<br>Mean: 3x3,5x5,7x7,9x9 | Visual comparison |

---

## 🛠️ **Technical Stack**

Python 3.x

OpenCV (cv2) - Noise addition & filtering

NumPy - Array operations & PSNR

Matplotlib - Visualization

text

---

## 📊 **Key Findings**

| **Filter** | **Salt & Pepper** | **Gaussian** | **Edge Preservation** |
|------------|-------------------|--------------|----------------------|
| **Mean** | Poor ❌ | Good ✅ | Blurs edges |
| **Median** | **Excellent** ⭐ | Fair | Good |
| **Gaussian** | Fair | **Excellent** ⭐ | Moderate blur |
| **Bilateral** | Good | Good | **Best** ⭐ |

---

## 🚀 **Execution Steps**

1. Install dependencies
pip install opencv-python numpy matplotlib
2. Place input image (Test Flower.jpg)
3. Run main script
python main.py


**Input:** `Test Flower.jpg`  
**Output:** 7 processed images + **PSNR values** + Multiple visualizations

---

## 📈 **PSNR Results (Typical Values)**

**Salt & Pepper Noise Removal:**

Median Filter: ~28-32 dB (BEST)

Mean Filter: ~22-25 dB

Bilateral: ~25-28 dB




**Gaussian Noise Removal:**

Gaussian Filter: ~30-35 dB (BEST)

Mean Filter: ~28-32 dB

Median Filter: ~25-28 dB

---

## 🎓 **Key Learning Outcomes**

1. **Noise Models**: Salt&Pepper (impulse), Gaussian (additive), Uniform (random)
2. **Filter Characteristics**:
   - **Median**: Best for Salt & Pepper (removes outliers)
   - **Gaussian/Mean**: Best for Gaussian noise (smoothens)
   - **Bilateral**: Edge-preserving denoising
3. **PSNR Metric**:  [Higher = Better]
4. **Kernel Size Trade-off**: Larger = More smoothing, Less detail


---

## 💡 **Real-World Applications**

- **Photography**: Noise reduction in low-light images
- **Medical Imaging**: Denoising MRI/CT scans
- **Satellite Imagery**: Atmospheric noise removal
- **Computer Vision**: Preprocessing for object detection

**✨ Comprehensive DIP noise analysis with quantitative PSNR evaluation!**
