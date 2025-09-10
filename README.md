# Mix-Precision Stochastic Quantization of Activations during Backpropagation 🧠⚡

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

## 📁 Thesis Structure

```
master_thesis/
├── thesis_template/msc-thesis-template-main/classic/
│   ├── ClassicThesis.tex                     # Main thesis document
│   ├── Chapters/                             # Thesis chapters
│   │   ├── 01-introduction.tex               # Introduction and motivation
│   │   ├── 02-background.tex                 # Background and related work
│   │   ├── 03-sota_and_rw.tex               # State-of-the-art review
│   │   ├── 04-contribution1.tex              # Deterministic quantization framework
│   │   ├── 05-contribution2.tex              # Stochastic quantization approach
│   │   ├── 06-contribution3.tex              # Generalization experiments
│   │   └── 07-discussion.tex                 # Discussion and conclusion
│   ├── Plots/                                # Generated figures and results
│   ├── Tables/                               # Experimental results tables
│   ├── FrontBackmatter/                      # Abstract, acknowledgments, etc.
│   └── Bibliography.bib                      # References
├── plots/                                    # Additional visualization materials
└── README.md                                 # This file
```

## 🛠️ Implementation Overview

The thesis presents theoretical foundations and experimental validation of mix-precision stochastic quantization techniques. Key implementation concepts include:

### Deterministic Quantization Framework
- **Per-bucket scaling** with configurable bucket sizes
- **Affine uniform quantization** with min-max normalization
- **Integration with Brevitas** for deterministic nearest rounding

### Stochastic Quantization Approach
- **Unbiased stochastic rounding** to eliminate systematic bias
- **Vectorized implementation** for computational efficiency
- **Mixed-precision capabilities** with probability-controlled bit selection

### Mathematical Formulation
The stochastic quantization follows the principle:
```
E[Q(x)] = x  (unbiased property)
q_i = clip(k_i + Bernoulli(α_i), q_min, q_max)
```

Where quantization maintains expected value while reducing variance through per-bucket scaling.

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
@mastersthesis{li2025mixprecision,
  title={Mix-Precision Stochastic Quantization of Activations during Backpropagation},
  author={Li, Jiufeng},
  year={2025},
  school={Heidelberg University},
  type={Master's Thesis},
  url={https://github.com/CoreSheep/Stochastic-Activations-Quantization-Backward}
}
```

### 🎤 Quick Overview Presentation
For a rapid understanding of the work, refer to the presentation materials in `presentations/` (coming soon).

## 🤝 Contributing

This repository contains the thesis documentation for academic research on mix-precision stochastic quantization. For academic discussions and questions about the research, please feel free to open issues or contact the author.

## 📄 License

This work is licensed under the GNU General Public License v2.0. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Brevitas Team** for the excellent quantization framework
- **PyTorch Community** for the robust deep learning foundation
- **Hardware and Articial Intelligence (HAWAII)
Lab** for research support and resources

---

<div align="center">

**🌟 Star this repository if you find it helpful for your research! 🌟**

*Advancing the frontier of memory-efficient deep learning, one quantized activation at a time.*

</div>
