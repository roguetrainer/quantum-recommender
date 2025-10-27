"""
Quantum-Inspired Recommender System
Demonstrates several quantum mechanics concepts applied to collaborative filtering
"""

import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class QuantumInspiredRecommender:
    """
    A recommender system using quantum-inspired methods:
    1. Complex Hilbert space embeddings
    2. Quantum interference effects
    3. Density matrix representation of user preferences
    """
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 10):
        """
        Initialize the quantum-inspired recommender
        
        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Dimension of the Hilbert space (complex vector space)
        """
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # Complex embeddings in Hilbert space (quantum states)
        # Users and items are represented as complex vectors
        self.user_embeddings = self._initialize_complex_embeddings(n_users, embedding_dim)
        self.item_embeddings = self._initialize_complex_embeddings(n_items, embedding_dim)
        
        # Density matrices for users (mixed quantum states)
        self.user_density_matrices = None
        
    def _initialize_complex_embeddings(self, n: int, dim: int) -> np.ndarray:
        """
        Initialize complex-valued embeddings (quantum state vectors)
        Each embedding is normalized to unit length (like quantum state vectors)
        """
        # Random complex numbers
        real_part = np.random.randn(n, dim) * 0.1
        imag_part = np.random.randn(n, dim) * 0.1
        embeddings = real_part + 1j * imag_part
        
        # Normalize to unit vectors (quantum normalization)
        norms = np.sqrt(np.sum(np.abs(embeddings)**2, axis=1, keepdims=True))
        return embeddings / norms
    
    def quantum_inner_product(self, vec1: np.ndarray, vec2: np.ndarray) -> complex:
        """
        Compute quantum inner product (bra-ket notation: <vec1|vec2>)
        This is the complex conjugate of vec1 dotted with vec2
        """
        return np.dot(np.conj(vec1), vec2)
    
    def compute_probability_amplitude(self, user_idx: int, item_idx: int) -> complex:
        """
        Compute probability amplitude for user-item interaction
        In quantum mechanics, probability = |amplitude|^2
        """
        user_state = self.user_embeddings[user_idx]
        item_state = self.item_embeddings[item_idx]
        
        # Quantum inner product gives the probability amplitude
        amplitude = self.quantum_inner_product(user_state, item_state)
        return amplitude
    
    def predict_with_interference(self, user_idx: int, item_idx: int, 
                                   context_items: List[int] = None) -> float:
        """
        Predict rating using quantum interference effects
        
        Multiple paths (through context items) can interfere constructively 
        or destructively, similar to the double-slit experiment
        """
        # Direct amplitude
        direct_amplitude = self.compute_probability_amplitude(user_idx, item_idx)
        
        if context_items is None or len(context_items) == 0:
            # No interference, just direct path
            probability = np.abs(direct_amplitude)**2
            return float(probability * 5)  # Scale to 0-5 rating
        
        # Compute interference from multiple paths
        total_amplitude = direct_amplitude
        
        for context_item in context_items:
            # Path through context item (two-step process)
            amp1 = self.compute_probability_amplitude(user_idx, context_item)
            amp2 = self.quantum_inner_product(
                self.item_embeddings[context_item], 
                self.item_embeddings[item_idx]
            )
            # Multiply amplitudes for sequential events
            total_amplitude += amp1 * amp2 * 0.3  # Weight factor
        
        # Born rule: probability = |amplitude|^2
        probability = np.abs(total_amplitude)**2
        
        # Scale to rating (0-5)
        return float(np.clip(probability * 5, 0, 5))
    
    def create_density_matrix(self, user_idx: int, rated_items: Dict[int, float]) -> np.ndarray:
        """
        Create a density matrix representation of user preferences
        
        Density matrices represent mixed quantum states and can capture
        uncertainty in user preferences
        """
        # Start with zero matrix
        density_matrix = np.zeros((self.embedding_dim, self.embedding_dim), dtype=complex)
        
        # Add contribution from each rated item
        total_weight = 0
        for item_idx, rating in rated_items.items():
            item_state = self.item_embeddings[item_idx]
            # Weight by rating (normalized)
            weight = rating / 5.0
            # Outer product: |ψ⟩⟨ψ| (projection operator)
            density_matrix += weight * np.outer(item_state, np.conj(item_state))
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            density_matrix /= total_weight
        
        return density_matrix
    
    def predict_with_density_matrix(self, user_idx: int, item_idx: int,
                                     rated_items: Dict[int, float]) -> float:
        """
        Predict rating using density matrix formalism
        
        This captures mixed states and uncertainty in user preferences
        """
        # Create density matrix from user's rating history
        rho = self.create_density_matrix(user_idx, rated_items)
        
        # Item state
        item_state = self.item_embeddings[item_idx]
        
        # Expectation value: Tr(ρ|ψ⟩⟨ψ|)
        projection = np.outer(item_state, np.conj(item_state))
        expectation = np.trace(rho @ projection)
        
        # Scale to rating
        return float(np.clip(np.real(expectation) * 5, 0, 5))
    
    def train(self, ratings: List[Tuple[int, int, float]], 
              epochs: int = 100, learning_rate: float = 0.01):
        """
        Train the quantum-inspired embeddings using gradient descent
        
        Args:
            ratings: List of (user_idx, item_idx, rating) tuples
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """
        print("Training quantum-inspired recommender...")
        
        for epoch in range(epochs):
            total_loss = 0
            
            for user_idx, item_idx, true_rating in ratings:
                # Forward pass
                amplitude = self.compute_probability_amplitude(user_idx, item_idx)
                predicted_rating = np.abs(amplitude)**2 * 5
                
                # Compute loss (MSE)
                error = predicted_rating - true_rating
                total_loss += error**2
                
                # Gradient descent update (simplified)
                # In quantum mechanics, we update to minimize prediction error
                user_state = self.user_embeddings[user_idx]
                item_state = self.item_embeddings[item_idx]
                
                # Gradient approximation
                gradient_scale = 2 * error * np.abs(amplitude) * 5
                
                # Update embeddings (maintaining normalization)
                self.user_embeddings[user_idx] -= learning_rate * gradient_scale * np.conj(item_state)
                self.item_embeddings[item_idx] -= learning_rate * gradient_scale * np.conj(user_state)
                
                # Re-normalize (quantum states must be unit vectors)
                self.user_embeddings[user_idx] /= np.linalg.norm(self.user_embeddings[user_idx])
                self.item_embeddings[item_idx] /= np.linalg.norm(self.item_embeddings[item_idx])
            
            if epoch % 20 == 0:
                avg_loss = total_loss / len(ratings)
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    def visualize_quantum_states(self, user_indices: List[int] = None, 
                                  item_indices: List[int] = None):
        """
        Visualize the quantum states in 2D (projection of complex embeddings)
        """
        if user_indices is None:
            user_indices = range(min(10, self.n_users))
        if item_indices is None:
            item_indices = range(min(10, self.n_items))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Project to 2D by taking first two dimensions
        user_real = np.real(self.user_embeddings[user_indices, :2])
        user_imag = np.imag(self.user_embeddings[user_indices, :2])
        
        item_real = np.real(self.item_embeddings[item_indices, :2])
        item_imag = np.imag(self.item_embeddings[item_indices, :2])
        
        # Plot users
        ax1.scatter(user_real[:, 0], user_real[:, 1], c='blue', 
                   label='Real part', alpha=0.6, s=100)
        ax1.scatter(user_imag[:, 0], user_imag[:, 1], c='red', 
                   label='Imaginary part', alpha=0.6, s=100)
        ax1.set_title('User Quantum States (First 2 Dimensions)')
        ax1.set_xlabel('Dimension 1')
        ax1.set_ylabel('Dimension 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot items
        ax2.scatter(item_real[:, 0], item_real[:, 1], c='green', 
                   label='Real part', alpha=0.6, s=100)
        ax2.scatter(item_imag[:, 0], item_imag[:, 1], c='orange', 
                   label='Imaginary part', alpha=0.6, s=100)
        ax2.set_title('Item Quantum States (First 2 Dimensions)')
        ax2.set_xlabel('Dimension 1')
        ax2.set_ylabel('Dimension 2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/quantum_states_visualization.png', dpi=150)
        print("Visualization saved!")
        
    def demonstrate_interference(self, user_idx: int, target_item: int, 
                                 context_items_list: List[List[int]]):
        """
        Demonstrate quantum interference effects with different context items
        """
        results = []
        
        # No interference (direct path only)
        no_interference = self.predict_with_interference(user_idx, target_item, [])
        results.append(('Direct path only', no_interference))
        
        # With different context items
        for i, context_items in enumerate(context_items_list):
            rating = self.predict_with_interference(user_idx, target_item, context_items)
            results.append((f'With context {i+1}', rating))
        
        return results


def demo():
    """
    Demonstrate the quantum-inspired recommender system
    """
    print("=" * 70)
    print("QUANTUM-INSPIRED RECOMMENDER SYSTEM DEMO")
    print("=" * 70)
    print()
    
    # Create synthetic dataset
    n_users = 20
    n_items = 30
    
    print(f"Setting up system with {n_users} users and {n_items} items")
    print(f"Using complex Hilbert space embeddings (quantum states)")
    print()
    
    # Initialize recommender
    recommender = QuantumInspiredRecommender(n_users, n_items, embedding_dim=8)
    
    # Generate synthetic ratings
    np.random.seed(42)
    ratings = []
    for _ in range(200):
        user = np.random.randint(0, n_users)
        item = np.random.randint(0, n_items)
        # Synthetic rating based on user-item interaction
        rating = np.random.uniform(1, 5)
        ratings.append((user, item, rating))
    
    # Train the model
    recommender.train(ratings, epochs=100, learning_rate=0.05)
    print()
    
    # Demonstrate quantum interference
    print("=" * 70)
    print("DEMONSTRATION 1: Quantum Interference Effects")
    print("=" * 70)
    print()
    print("In quantum mechanics, particles can take multiple paths, and these")
    print("paths interfere with each other (constructive or destructive interference).")
    print("We apply this to recommendations: considering multiple 'paths' through")
    print("context items affects the final prediction.")
    print()
    
    user_idx = 0
    target_item = 5
    
    # Different context scenarios
    context_scenarios = [
        [1, 2],      # Context set 1
        [10, 15],    # Context set 2
        [1, 2, 10],  # Larger context set
    ]
    
    results = recommender.demonstrate_interference(user_idx, target_item, context_scenarios)
    
    for description, rating in results:
        print(f"{description:25s}: Predicted rating = {rating:.3f}")
    
    print()
    print("Notice how different context items produce different predictions due")
    print("to constructive/destructive interference of probability amplitudes!")
    print()
    
    # Demonstrate density matrix approach
    print("=" * 70)
    print("DEMONSTRATION 2: Density Matrix Predictions")
    print("=" * 70)
    print()
    print("Density matrices represent 'mixed quantum states' - useful for")
    print("capturing uncertainty in user preferences based on rating history.")
    print()
    
    # Create a user's rating history
    user_history = {
        2: 5.0,   # User loved item 2
        7: 4.5,   # User liked item 7
        12: 2.0,  # User disliked item 12
    }
    
    print(f"User {user_idx}'s rating history:")
    for item, rating in user_history.items():
        print(f"  Item {item}: {rating:.1f} stars")
    print()
    
    # Predict for new items
    print("Predictions for unrated items (using density matrix):")
    test_items = [3, 8, 15, 20]
    for item in test_items:
        pred = recommender.predict_with_density_matrix(user_idx, item, user_history)
        print(f"  Item {item}: {pred:.3f} stars")
    
    print()
    
    # Visualize quantum states
    print("=" * 70)
    print("DEMONSTRATION 3: Visualizing Quantum States")
    print("=" * 70)
    print()
    print("Creating visualization of user and item quantum states...")
    print("(Complex embeddings projected to 2D)")
    print()
    
    recommender.visualize_quantum_states()
    
    # Show some quantum properties
    print("=" * 70)
    print("QUANTUM PROPERTIES")
    print("=" * 70)
    print()
    
    # Show that states are normalized (like quantum states)
    user_norms = np.linalg.norm(recommender.user_embeddings, axis=1)
    item_norms = np.linalg.norm(recommender.item_embeddings, axis=1)
    
    print(f"User state norms (should be ≈1): min={user_norms.min():.4f}, max={user_norms.max():.4f}")
    print(f"Item state norms (should be ≈1): min={item_norms.min():.4f}, max={item_norms.max():.4f}")
    print()
    
    # Show complex nature
    print("Complex embeddings allow for:")
    print("  • Phase information (angle in complex plane)")
    print("  • Interference effects (amplitude addition)")
    print("  • Richer representation than real-valued embeddings")
    print()
    
    # Compare predictions
    print("=" * 70)
    print("COMPARISON: Different Quantum-Inspired Methods")
    print("=" * 70)
    print()
    
    test_user = 5
    test_item = 10
    
    pred1 = recommender.predict_with_interference(test_user, test_item, [])
    pred2 = recommender.predict_with_interference(test_user, test_item, [2, 7])
    pred3 = recommender.predict_with_density_matrix(test_user, test_item, user_history)
    
    print(f"For User {test_user}, Item {test_item}:")
    print(f"  Direct quantum prediction:       {pred1:.3f}")
    print(f"  With interference (context):     {pred2:.3f}")
    print(f"  Density matrix approach:         {pred3:.3f}")
    print()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("This quantum-inspired recommender system demonstrates:")
    print()
    print("1. HILBERT SPACE EMBEDDINGS")
    print("   Users and items as complex vectors in quantum state space")
    print()
    print("2. QUANTUM INTERFERENCE")
    print("   Multiple recommendation paths interfere constructively/destructively")
    print()
    print("3. DENSITY MATRICES")
    print("   Mixed states capture uncertainty in preferences")
    print()
    print("4. BORN RULE")
    print("   Probability = |amplitude|² converts quantum amplitudes to predictions")
    print()
    print("Advantages over classical methods:")
    print("  • Richer representations (complex vs real numbers)")
    print("  • Natural handling of uncertainty and superposition")
    print("  • Interference effects capture subtle item relationships")
    print("  • Theoretically grounded in quantum probability theory")
    print()
    print("=" * 70)


if __name__ == "__main__":
    demo()
