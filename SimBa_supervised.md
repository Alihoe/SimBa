# Supervised Re-Ranking

Challenges:
- Training data has binary labels
- A meaningful relevance score should be produced

3 Options:
1. Pairwise Learning to Rank Algorithm
2. Binary Classifier with prediction-independent relevance score
    - problem: training data does not influence ranking
3. Binary Classifier with classifier's confidence score as relevance score
    - problem: artificially increase recall

## Pairwise Learning to Rank Algorithm