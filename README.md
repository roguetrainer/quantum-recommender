# Quantum Recommender

A comprehensive exploration of quantum mechanics applied to recommender systems, featuring implementations, interactive tutorials, and academic research survey.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📚 Overview

This repository demonstrates how the mathematical formalism of quantum mechanics can be applied to build novel recommender systems. It includes:

- **Working implementations** of quantum-inspired algorithms
- **Interactive Jupyter notebook** for hands-on experimentation
- **Academic survey** of the field's research landscape
- **Visualizations** of quantum states and interference effects

The project bridges quantum computing, machine learning, and cognitive science to explore new approaches to the recommendation problem.

## 🎯 Key Features

### Quantum-Inspired Algorithms

- **Complex Hilbert Space Embeddings**: Users and items as complex-valued quantum state vectors
- **Quantum Interference**: Multiple recommendation paths that interfere constructively/destructively
- **Density Matrix Formalism**: Mixed quantum states for modeling preference uncertainty
- **Born Rule Predictions**: Converting quantum probability amplitudes to ratings

### Practical Implementations

- Fully functional Python code with NumPy/SciPy
- Three distinct prediction methods (direct, interference, density matrix)
- Training via gradient descent with quantum state normalization
- Comprehensive visualizations of quantum states

## 📂 Repository Structure

```
quantum-recommender/
├── README.md                          # This file
├── quantum_recommender.py             # Core implementation (standalone script)
├── quantum_recommender.ipynb          # Interactive Jupyter notebook
├── quantum_recommender_survey.md      # Academic survey article (~6,500 words)
└── visualizations/
    └── quantum_states_visualization.png
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy scipy matplotlib
```

For the Jupyter notebook:
```bash
pip install jupyter pandas
```

### Running the Demo

**Option 1: Standalone Script**
```bash
python quantum_recommender.py
```

This will run a complete demonstration including:
- Training a quantum-inspired recommender on synthetic data
- Demonstrating quantum interference effects
- Showing density matrix predictions
- Visualizing quantum states in 2D projections

**Option 2: Interactive Notebook**
```bash
jupyter notebook quantum_recommender.ipynb
```

Explore the notebook to:
- Understand the theory step-by-step
- Run interactive examples
- Experiment with parameters
- Create custom visualizations

## 🔬 What's Inside

### 1. Quantum-Inspired Recommender (`quantum_recommender.py`)

A complete implementation featuring:

```python
class QuantumInspiredRecommender:
    """
    Recommender system using quantum mechanics concepts:
    - Complex Hilbert space embeddings
    - Quantum interference effects  
    - Density matrix representations
    """
```

**Key Methods:**
- `quantum_inner_product()`: Computes ⟨ψ|φ⟩ in complex space
- `predict_with_interference()`: Uses quantum interference from context items
- `create_density_matrix()`: Builds mixed state from rating history
- `train()`: Gradient descent with quantum normalization

### 2. Interactive Notebook (`quantum_recommender.ipynb`)

11 comprehensive sections covering:

1. **Introduction** - Background and motivation
2. **Class Definition** - Complete implementation with explanations
3. **Initialization** - Setting up the quantum system
4. **Data Generation** - Creating synthetic training data
5. **Model Training** - Optimizing quantum embeddings
6. **Quantum Interference Demo** - Interactive exploration
7. **Density Matrix Predictions** - Mixed state approach
8. **State Visualization** - Complex plane and 2D projections
9. **Method Comparison** - Side-by-side evaluation
10. **Interactive Exploration** - Custom prediction function
11. **Summary & Experiments** - Key takeaways and suggestions

### 3. Academic Survey (`quantum_recommender_survey.md`)

A comprehensive review of the research field covering:

- **Quantum Algorithms**: Kerenidis-Prakash algorithm, Tang's dequantization
- **Quantum Annealing**: D-Wave applications for feature selection, carousel optimization
- **Quantum-Inspired Methods**: Complex embeddings, quantum cognition, density matrices
- **Quantum Neural Networks**: Recent QNN architectures for recommendations
- **Theoretical Foundations**: Quantum probability, geometric interpretations
- **Empirical Results**: Benchmarks and performance comparisons
- **Future Directions**: Open challenges and opportunities

## 💡 Concepts Explained

### Complex Hilbert Space Embeddings

Users and items are represented as complex vectors (quantum states):
- Each vector has unit norm: ||ψ|| = 1
- Complex numbers encode both magnitude and phase
- Inner products give quantum similarity: ⟨ψ|φ⟩

### Quantum Interference

Recommendations can be influenced by "context items" that create interference:
```
Total Amplitude = Direct Amplitude + Σ(Context Path Amplitudes)
Probability = |Total Amplitude|²
```

This captures how considering related items affects final predictions.

### Density Matrices

User preferences as mixed quantum states:
```
ρ = Σᵢ wᵢ |ψᵢ⟩⟨ψᵢ|
```

Where weights wᵢ represent the importance of different preference states.

## 📊 Example Results

From the demonstration:

```
Quantum Interference Effects:
- Direct path only:           1.701 stars
- With context [1, 2]:        4.373 stars  
- With context [10, 15]:      4.360 stars
- With context [1, 2, 10]:    5.000 stars
```

Different context items produce different predictions due to quantum interference!

## 🎓 Educational Use

This repository is ideal for:

- **Quantum Computing Students**: Learn practical applications of quantum formalism
- **Machine Learning Researchers**: Explore novel recommendation approaches
- **Course Projects**: Quantum computing, recommender systems, or applied mathematics courses
- **Research Exploration**: Starting point for quantum-inspired ML research

## 📖 Background Reading

The survey article provides extensive references, but key papers include:

- **Kerenidis & Prakash (2016)**: "Quantum Recommendation Systems" - Original quantum algorithm
- **Tang (2018)**: "A quantum-inspired classical algorithm for recommendation systems" - Classical dequantization
- **Busemeyer & Bruza (2012)**: "Quantum Models of Cognition and Decision" - Quantum cognition foundations
- **Ferrari Dacrema et al. (2021-2024)**: Series on quantum annealing for recommendations
- **Pilato & Vella (2023)**: "A Survey on Quantum Computing for Recommendation Systems"

## 🔬 Technical Details

### Mathematical Framework

The core mathematics draws from quantum mechanics:

**State Representation:**
```
|u⟩ = Σᵢ aᵢ|i⟩, where aᵢ ∈ ℂ and Σᵢ|aᵢ|² = 1
```

**Quantum Inner Product:**
```
⟨u|v⟩ = Σᵢ ūᵢvᵢ
```

**Born Rule:**
```
P(outcome) = |⟨outcome|state⟩|²
```

### Implementation Details

- **Embeddings**: Complex-valued NumPy arrays (n_entities × embedding_dim)
- **Normalization**: States maintained at unit norm throughout training
- **Training**: Custom gradient descent respecting quantum constraints
- **Sampling**: Uses quantum probability distributions via Born rule

## 🛠️ Extending the Code

Ideas for extensions:

1. **Real Datasets**: Load MovieLens, Amazon Reviews, or other recommendation datasets
2. **Advanced Interference**: Implement higher-order interference terms
3. **Quantum Entanglement**: Model user-user or item-item correlations
4. **Measurement Operators**: Add projection operators for preference categories
5. **Benchmarking**: Compare against classical matrix factorization
6. **Quantum Hardware**: Adapt for actual quantum processors (Qiskit, Cirq)

Example extension:
```python
def predict_with_entanglement(self, user1_idx, user2_idx, item_idx):
    """
    Model entangled preferences between two users
    """
    # Create entangled state
    state = self.user_embeddings[user1_idx] ⊗ self.user_embeddings[user2_idx]
    # Project onto item space
    # ... implement entangled recommendation logic
```

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- **Algorithms**: New quantum-inspired recommendation methods
- **Visualizations**: Better ways to visualize quantum states
- **Datasets**: Integration with standard benchmarks
- **Documentation**: Improved explanations and tutorials
- **Performance**: Optimizations and scalability improvements

Please open an issue to discuss major changes before submitting a pull request.

## ⚖️ License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

This work synthesizes ideas from multiple research communities:

- **Quantum Computing**: For the mathematical formalism
- **Recommender Systems**: For the application domain  
- **Cognitive Science**: For quantum models of decision-making
- **Information Retrieval**: For quantum-inspired IR methods

Special recognition to researchers who pioneered this interdisciplinary field, including Iordanis Kerenidis, Anupam Prakash, Ewin Tang, Peter Bruza, Jerome Busemeyer, Maurizio Ferrari Dacrema, and many others cited in the survey.

## 📬 Contact & Citations

If you use this code or survey in your research, please cite:

```bibtex
@software{quantum_recommender_2025,
  title = {Quantum Recommender: Quantum Mechanics for Recommendation Systems},
  author = {Ian Buckley},
  year = {2025},
  url = {https://github.com/[username]/quantum-recommender}
}
```

## 🔮 Future Work

Potential research directions:

- **Quantum Annealing Integration**: Connect to D-Wave systems for optimization
- **Quantum Neural Networks**: Implement variational quantum circuits
- **Cognitive Validation**: Empirical studies of quantum-like user behavior
- **Hybrid Systems**: Combine quantum and classical recommendation engines
- **Explainability**: Use quantum geometry for interpretable recommendations

## 📚 Additional Resources

### Papers and Books
- arXiv preprints on quantum machine learning
- "The Geometry of Information Retrieval" by van Rijsbergen
- RecSys conference proceedings (quantum sessions)
- QuantumCLEF evaluation lab

### Software
- [PennyLane](https://pennylane.ai/): Quantum machine learning framework
- [Qiskit](https://qiskit.org/): IBM's quantum computing toolkit
- [D-Wave Ocean](https://ocean.dwavesys.com/): Quantum annealing platform

### Courses
- edX: Quantum Computing courses
- Coursera: Recommender Systems specializations
- MIT OpenCourseWare: Quantum computing lectures

---

**Built with ❤️ for the quantum × machine learning community**

*"The universe is not only queerer than we suppose, but queerer than we can suppose." - J.B.S. Haldane*

*"Quantum mechanics is very impressive. But an inner voice tells me that it is not yet the real thing." - Albert Einstein*

Perhaps quantum mechanics can tell us something about how we choose what to watch on Netflix... 🎬🔮
