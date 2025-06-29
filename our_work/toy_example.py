import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import pandas as pd
# import ace_tools as tools

# Set seed for reproducibility
np.random.seed(42)

# === Synthetic Model and Concept Setup ===

# True model: prediction is linear combination of 3 features
true_w = np.array([0.6, -0.4, 0.9])  # model weights
# Normalize weights to unit norm
true_w /= np.linalg.norm(true_w)
def model(x):
    return np.dot(x, true_w)

# Define concept distribution: Gaussian in embedding space
mu_c = np.array([0.5, 0.3, -0.2])
Sigma_c = np.array([
    [0.1, 0.05, 0.0],
    [0.05, 0.2, 0.0],
    [0.0, 0.0, 0.15]
])

# Generate N concept samples
N = 10000
concept_samples = np.random.multivariate_normal(mu_c, Sigma_c, size=N)

# Generate fixed image feature vector
h = np.array([1.2, -0.5, 0.7])
# Normalize h to unit norm
h /= np.linalg.norm(h)

# Define SKIT-like test statistic: here, simple proxy = correlation between y and ⟨c, h⟩
# True label is from the model
x_samples = np.random.normal(size=(N, 3))
y = np.array([model(x) for x in x_samples])

# For each concept sample, calculate e(c) as a simple score (e.g. |corr(y, ⟨c, h⟩)|)
scores = []
for c in concept_samples:
    z = np.dot(h, c)
    concept_signal = np.dot(x_samples, c)  # ⟨c, x⟩ as surrogate "concept activation"
    corr = np.corrcoef(y, concept_signal)[0, 1]
    e_val = np.abs(corr)
    scores.append(e_val)

scores = np.array(scores)
threshold = 0.2
delta = 0.05

# Estimate probability that e(c) > threshold
hat_p = np.mean(scores > threshold)
epsilon_N = np.sqrt(np.log(2 / delta) / (2 * N))
lower_bound = hat_p - epsilon_N

# Prepare DataFrame to display
df = pd.DataFrame({
    "e(c)": scores,
    "Important (e(c) > τ)": scores > threshold
})

# tools.display_dataframe_to_user("Concept Importance Scores", df)

print(f"Estimated probability that e(c) > {threshold}: {hat_p:.4f}")
print(f"Lower bound with confidence {1 - delta}: {lower_bound:.4f}")
print(f"Estimated epsilon_N: {epsilon_N:.4f}")

# Plot the distribution of scores
plt.figure(figsize=(10, 6))
sns.histplot(scores, bins=30, kde=True)
threshold_percentile = np.mean(scores > threshold)
plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold}, percentile = {threshold_percentile:.2f}')
# Add a line for 90 percentile
percentile_90 = np.percentile(scores, 90)
plt.axvline(percentile_90, color='blue', linestyle='--', label=f'90th Percentile = {percentile_90:.4f}')
plt.title('Distribution of Concept Importance Scores')
plt.xlabel('e(c)')
plt.ylabel('Frequency')
plt.legend()
plt.savefig("our_work/figures/concept_importance_scores.png")

# Print the percentage of concepts above the threshold
percentage_above_threshold = np.mean(scores > threshold) * 100
print(f"Percentage of concepts above the threshold: {percentage_above_threshold:.2f}%")

# === Synthetic Model and Concept Setup ===

def calculate_e_histogram(mu_c, Sigma_c, true_w, h, N=10000):
    """
    Calculate histogram of e-values for concepts sampled from given distribution.
    
    Parameters:
    - mu_c: mean vector of concept distribution
    - Sigma_c: covariance matrix of concept distribution  
    - true_w: true model weights
    - h: fixed image feature vector
    - N: number of concept samples
    
    Returns:
    - scores: array of e-values
    """
    # Generate concept samples from the given distribution
    concept_samples = np.random.multivariate_normal(mu_c, Sigma_c, size=N)
    
    # For the fixed image h, compute model output and concept activations
    model_output = np.dot(h, true_w)  # scalar
    # model_output /= np.abs(model_output)  # normalize to avoid division by zero
    # For each concept, compute activation for h
    concept_activations = np.dot(concept_samples, true_w)  # shape (N,)
    # concept_activations /= np.abs(concept_activations)  # normalize

    # The correlation between model_output (scalar) and concept_activations (vector) is not defined.
    # Instead, we can define e(c) as the absolute value of the product between model_output and concept_activation,
    # or just the absolute value of concept_activation if you want to measure alignment.
    # Here, we use the absolute value of the product as a proxy for importance.
    scores = np.abs(model_output * concept_activations)
    
    return scores


def create_scaled_sigmas(base_sigma, scales):
    """
    Create a list of scaled covariance matrices.

    Parameters:
    - base_sigma: base covariance matrix (numpy array)
    - scales: list or array of scaling factors

    Returns:
    - List of scaled covariance matrices
    """
    return [scale * base_sigma for scale in scales]

# --- Try Multiple Sigmas ---

mu_c = h  # Use the fixed image feature vector as the mean for concept distribution
# rotate mu_c slightly to avoid singularity
scales = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
sigmas = create_scaled_sigmas(Sigma_c, scales)

scores_list_important = []
for sigma in sigmas:
    scores = calculate_e_histogram(mu_c, sigma, true_w, h, N)
    scores_list_important.append(scores)

# Plot histograms for each sigma
plt.figure(figsize=(12, 8))
for i, scores in enumerate(scores_list_important):
    sns.histplot(scores, bins=30, kde=True, label=f'Sigma Scale: {scales[i]}', stat='density')
plt.title('Distribution of e(c) for Different Sigma Scales')
plt.xlabel('e(c)')
plt.ylabel('Density')
plt.legend()
plt.savefig("our_work/figures/e_c_important_distribution_scaled_sigmas.png")

# --- Try with no important concept ---
# Find mu_c such that e(c) is not important
# Here we can use a mean that is orthogonal to h, or simply a zero vector
mu_c = np.zeros_like(h)  # Use a zero vector as the mean for concept distribution
# rotate mu_c slightly to avoid singularity
scales = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
sigmas = create_scaled_sigmas(Sigma_c, scales)

scores_list_not_important = []
for sigma in sigmas:
    scores = calculate_e_histogram(mu_c, sigma, true_w, h, N)
    scores_list_not_important.append(scores)

# Plot histograms for each sigma
plt.figure(figsize=(12, 8))
for i, scores in enumerate(scores_list_not_important):
    sns.histplot(scores, bins=30, kde=True, label=f'Sigma Scale: {scales[i]}', stat='density')
plt.title('Distribution of e(c) for Different Sigma Scales')
plt.xlabel('e(c)')
plt.ylabel('Density')
plt.legend()
plt.savefig("our_work/figures/e_c_not_important_distribution_scaled_sigmas.png")

# --- Define threshold and calculate probabilities ---
threshold = 0.6
def calculate_probabilities(scores_list, threshold):
    """
    Calculate the probability of scores exceeding the threshold for each list of scores.
    
    Parameters:
    - scores_list: list of numpy arrays containing scores
    - threshold: threshold value to compare against
    
    Returns:
    - List of probabilities for each score list
    """
    probabilities = [np.count_nonzero(scores > threshold) / len(scores) for scores in scores_list]
    return probabilities

# Calculate probabilities for important concepts
probabilities_important = calculate_probabilities(scores_list_important, threshold)
# Calculate probabilities for not important concepts
probabilities_not_important = calculate_probabilities(scores_list_not_important, threshold)

# Print probabilities
print("Probabilities of e(c) > threshold for important concepts:")
print(probabilities_important)
print("Probabilities of e(c) > threshold for not important concepts:")
print(probabilities_not_important)

# Make it into binary by demanding 90 percent confidence
def calculate_binary_decision(probabilities, delta=0.1):
    return [1 if prob > 0.9 else 0 for prob in probabilities]
binary_decision_important = calculate_binary_decision(probabilities_important)
binary_decision_not_important = calculate_binary_decision(probabilities_not_important)
# Print binary decisions
print("Binary decisions for important concepts (1 if > 90% confidence):")
print(binary_decision_important)
print("Binary decisions for not important concepts (1 if > 90% confidence):")
print(binary_decision_not_important)

# Plot binary decisions
plt.figure(figsize=(12, 6))
plt.bar(range(len(scales)), binary_decision_important, alpha=0.5, label='Important Concepts', color='blue')
plt.bar(range(len(scales)), binary_decision_not_important, alpha=0.5, label='Not Important Concepts', color='red')
plt.xticks(range(len(scales)), [f'Scale: {scale}' for scale in scales])
plt.title('Binary Decisions for Different Sigma Scales')
plt.xlabel('Sigma Scale')
plt.ylabel('Binary Decision (1 if > 90% confidence)')
plt.legend()
plt.savefig("our_work/figures/binary_decisions_scaled_sigmas.png")