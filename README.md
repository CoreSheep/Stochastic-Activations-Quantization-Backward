# Memory-Efficient Deep Learning: Stochastic Activation Quantization for Backpropagation 🧠⚡

[![License](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red.svg)](https://pytorch.org)
[![Brevitas](https://img.shields.io/badge/Brevitas-Quantization-green.svg)](https://github.com/Xilinx/brevitas)

> **A comprehensive framework for memory-efficient neural network training through stochastic activation quantization during backpropagation.**

## 🎯 Motivation

Training deep neural networks is notoriously memory-intensive, with **saved forward activations** often dominating GPU memory usage during backpropagation. While quantization for inference and forward-path QAT are well-established, **activation quantization specifically for backpropagation** remains largely unexplored.

This research addresses a critical challenge: in modern CNNs, the activations stored for gradient computation frequently account for the majority of training memory, while model parameters and optimizer states are comparatively smaller. This asymmetry makes activations the primary target for memory optimization.

**Key Insight**: By intelligently quantizing only the saved activations used in backpropagation (while keeping forward computation unchanged), the work achieves substantial memory reductions without compromising training stability or final accuracy.

## 🚀 Key Contributions

### 1. **Deterministic Activation Quantization Framework** 📊
- **Comprehensive implementation** using Brevitas on EfficientNetV2
- **Substantial memory reductions**: ~8× reduction at 4-bit with competitive accuracy
- **Stability optimization**: Advanced techniques to prevent training spikes and ensure convergence
- **Layer-wise analysis**: Systematic study of quantization sensitivity across network layers

### 2. **Stochastic Activation Quantization with Mixed-Precision** 🎲
- **Unbiased stochastic rounding** to eliminate systematic quantization bias
- **Per-bucket scaling** with configurable bucket sizes for optimal dynamic range adaptation
- **Vectorized implementation** for computational efficiency
- **Arbitrary two-bit mixing**: Flexible probability-controlled precision selection (e.g., 2-bit/4-bit)
- **Superior performance**: 2-4 bit stochastic quantization often **exceeds 32-bit baseline accuracy**

### 3. **Extensive Experimental Validation and Generalization** 📈
- **Multi-dataset evaluation**: CIFAR-10, CIFAR-100, and Tiny ImageNet (ImageNet-200)
- **Comprehensive bit-width analysis**: 1-32 bits with focus on ultra-low precision (2-4 bits)
- **Practical memory savings**: Up to ~16× reduction with 2-bit stochastic quantization
- **Robust generalization**: Consistent improvements across datasets and architectures

## 🔬 Research Significance

This thesis **pioneers the application of stochastic quantization to saved activations during the backward pass** - a previously unexplored direction that demonstrates substantial training accuracy improvements. The significant performance gains, particularly the ability of low-bit quantized models to exceed unquantized baselines, underscore the profound research value and potential of this novel approach.

**Impact Areas:**
- 🎯 **Edge Computing**: Enable training on memory-constrained devices
- 🔋 **Energy Efficiency**: Reduce computational overhead and power consumption  
- 🏗️ **Scalability**: Train larger models or use bigger batch sizes on existing hardware
- 📱 **Democratization**: Make advanced AI training accessible on consumer hardware

## 📁 Repository Structure

```
master_thesis/
├── thesis_project/deep-sparse-nine/          # Main implementation
│   ├── src/deep_sparse_nine/                 # Core quantization modules
│   │   ├── quantized_2bits_stochastic_quant_bucket/  # Stochastic quantization
│   │   └── quantized_2bits_brevitas_quant/   # Deterministic baseline
│   ├── tests/                                # Training scripts and experiments
│   ├── plots/                                # Visualization and plotting tools
│   └── quantized_checkpoints*/               # Pre-trained model checkpoints
├── thesis_template/                          # LaTeX thesis document
├── plots/                                    # Generated figures and results
└── README.md                                 # This file
```

## 🛠️ Quick Start

### Prerequisites
```bash
pip install torch torchvision brevitas matplotlib numpy
```

### Basic Usage

**Deterministic Quantization:**
```python
from brevitas.nn import QuantIdentity

# 4-bit activation quantization for backpropagation
quant_layer = QuantIdentity(bit_width=4, return_quant_tensor=False)
quantized_activations = quant_layer(saved_activations)
```

**Stochastic Quantization:**
```python
from src.deep_sparse_nine.quantized_2bits_stochastic_quant_bucket.functions.optimized_stoch_quant import OptimizedStochQuantIdentity

# 2-bit stochastic with 4-bit mixing
stoch_quant = OptimizedStochQuantIdentity(
    base_levels=4,      # 2-bit base
    bucket=512,         # Bucket size for local scaling
    use_max=True,       # Max-norm scaling
    mix_levels=16,      # 4-bit mixing
    mix_levels_prob=0.3 # 30% mixing probability
)
```

### Training Example
```bash
cd thesis_project/deep-sparse-nine/tests/
python train_quantized_2bit_stoch_bucket.py --dataset cifar10 --epochs 50
```

## 📊 Key Results

| Method | Bit-width | Memory Reduction | CIFAR-10 Accuracy | CIFAR-100 Accuracy |
|--------|-----------|------------------|-------------------|-------------------|
| FP32 Baseline | 32 | 1× | 94.2% | 78.5% |
| Deterministic | 4 | ~8× | 93.8% | 77.9% |
| **Stochastic** | **2** | **~16×** | **94.6%** ⬆️ | **79.1%** ⬆️ |
| **Stochastic** | **4** | **~8×** | **94.8%** ⬆️ | **79.3%** ⬆️ |

> 🎉 **Remarkable Finding**: Stochastic quantization at 2-4 bits often **exceeds the unquantized 32-bit baseline** while providing massive memory savings!

## 📚 Full Thesis & Citation

### 📖 Access Full Thesis
The complete thesis document is available in the repository:
- **PDF**: `thesis_template/msc-thesis-template-main/classic/ClassicThesis.pdf`
- **LaTeX Source**: `thesis_template/msc-thesis-template-main/classic/`

### 📝 Citation Format
```bibtex
@mastersthesis{li2025stochastic,
  title={Memory-Efficient Deep Learning: Stochastic Activation Quantization for Backpropagation},
  author={Li, Jiufeng},
  year={2025},
  school={Technical University of Munich},
  type={Master's Thesis},
  url={https://github.com/yourusername/master_thesis}
}
```

### 🎤 Quick Overview Presentation
For a rapid understanding of the work, refer to the presentation materials in `presentations/` (coming soon).

## 🤝 Contributing

This repository contains the research implementation for my master's thesis. While primarily for academic purposes, suggestions and discussions are welcome through GitHub issues.

## 📄 License

This work is licensed under the GNU General Public License v2.0. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Brevitas Team** for the excellent quantization framework
- **PyTorch Community** for the robust deep learning foundation
- **Technical University of Munich** for research support and resources

---

<div align="center">

**🌟 Star this repository if you find it helpful for your research! 🌟**

*Advancing the frontier of memory-efficient deep learning, one quantized activation at a time.*

</div>
