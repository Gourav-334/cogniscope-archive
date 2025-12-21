import numpy as np
from typing import Dict, List, Tuple

# Centroids for each of the 16 MBTI types
# Order: [Dopamine, Serotonin, Testosterone/Androgenicity, Estrogen/Oxytocin]
CENTROIDS = {
    "ISTJ": [2, 5, 3, 2],
    "ISFJ": [2, 5, 2, 3],
    "INFJ": [3, 3, 2, 5],
    "INTJ": [3, 2, 4, 3],
    "ISTP": [4, 2, 4, 2],
    "ISFP": [3, 3, 2, 4],
    "INFP": [3, 2, 1, 5],
    "INTP": [4, 2, 4, 2],
    "ESTP": [5, 1, 5, 1],
    "ESFP": [5, 2, 3, 3],
    "ENFP": [5, 2, 2, 4],
    "ENTP": [5, 1, 4, 2],
    "ESTJ": [3, 4, 5, 1],
    "ESFJ": [2, 5, 2, 4],
    "ENFJ": [4, 3, 2, 5],
    "ENTJ": [4, 2, 5, 2],
}

class MBTINearestCentroidClassifier:
    def __init__(self, centroids: Dict[str, List[float]] = CENTROIDS):
        self.types = list(centroids.keys())
        self.centroids = np.array(list(centroids.values()))  # Shape: (16, 4)
        self.type_labels = self.types

    def predict(self, user_scores: List[float]) -> str:
        user_array = np.array(user_scores).reshape(1, -1)
        distances = np.sqrt(((self.centroids - user_array) ** 2).sum(axis=1))
        closest_idx = np.argmin(distances)
        return self.type_labels[closest_idx]

    def predict_with_ranking(self, user_scores: List[float]) -> Tuple[str, Dict[str, float]]:
        user_array = np.array(user_scores).reshape(1, -1)
        distances = np.sqrt(((self.centroids - user_array) ** 2).sum(axis=1))
        distance_dict = {mbti_type: float(dist) for mbti_type, dist in zip(self.type_labels, distances)}
        sorted_distances = dict(sorted(distance_dict.items(), key=lambda x: x[1]))
        closest_type = min(distance_dict, key=distance_dict.get)
        return closest_type, sorted_distances

def get_user_input() -> List[float]:
    print("=== MBTI Prediction Based on Neurochemical Self-Assessment ===\n")
    print("Rate yourself on a scale of 1 to 5 for each trait:\n1 = Very low\n2 = Slightly low\n3 = Balanced\n4 = Slightly high\n5 = Very high\n")
    
    questions = [
        "Dopamine-related traits (novelty-seeking, excitement, energy, curiosity, impulsivity)",
        "Serotonin-related traits (calmness, caution, respect for rules/authority, organization, tradition)",
        "Testosterone/Androgenicity-related traits (assertiveness, competitiveness, analytical thinking, directness)",
        "Estrogen/Oxytocin-related traits (empathy, emotional sensitivity, holistic thinking, people-focused, bonding)"
    ]
    
    scores = []
    for i, question in enumerate(questions, 1):
        while True:
            try:
                value = float(input(f"{i}. {question}:\n   Your rating (1–5): "))
                if 1 <= value <= 5:
                    scores.append(value)
                    break
                else:
                    print("   Please enter a number between 1 and 5.")
            except ValueError:
                print("   Invalid input. Please enter a number between 1 and 5.")
    
    return scores

def main():
    classifier = MBTINearestCentroidClassifier()
    
    print("\nLet's determine your possible MBTI type!\n")
    user_scores = get_user_input()
    
    print("\nYour reported levels:")
    labels = ["Dopamine", "Serotonin", "Testosterone/Androgenicity", "Estrogen/Oxytocin"]
    for label, score in zip(labels, user_scores):
        print(f"  {label}: {score}")
    
    best_type, all_distances = classifier.predict_with_ranking(user_scores)
    
    print(f"\n🎯 Predicted MBTI type: {best_type}")
    print("\nTop 5 closest matches:")
    for i, (mbti_type, dist) in enumerate(list(all_distances.items())[:5]):
        print(f"  {i+1}. {mbti_type} — similarity distance: {dist:.2f}")
    
    print("\nNote: This is a fun, speculative tool based on loose temperament-neurochemistry analogies.")
    print("Actual personality is far more complex than four chemicals or 16 types! 😊")

if __name__ == "__main__":
    main()