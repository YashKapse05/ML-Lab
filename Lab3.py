# Machine Learning Algorithms
# Find-S Algorithm and Candidate Elimination Algorithm

import pandas as pd
import numpy as np

# -----------------------------
# Load Dataset
# -----------------------------
# Replace 'trainingdata.csv' with your dataset filename

data = pd.read_csv("enjoy (1).csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# ==========================================================
# FIND-S ALGORITHM
# ==========================================================

def find_s(X, y):
    hypothesis = None
    print("========== FIND-S ALGORITHM ==========")
    print("Initial Hypothesis:", hypothesis)

    for i in range(len(X)):
        if y[i] == "Yes":
            if hypothesis is None:
                hypothesis = X[i].copy()
            else:
                for j in range(len(hypothesis)):
                    if hypothesis[j] != X[i][j]:
                        hypothesis[j] = "?"

            print(f"\nAfter Training Example {i+1}")
            print(hypothesis)

    return hypothesis


final_hypothesis = find_s(X, y)

print("\nFinal Hypothesis:")
print(final_hypothesis)


# ==========================================================
# CANDIDATE ELIMINATION ALGORITHM
# ==========================================================

def candidate_elimination(concepts, target):

    specific = concepts[0].copy()
    general = [["?" for _ in range(len(specific))]]

    print("\n========== CANDIDATE ELIMINATION ==========")
    print("Initial Specific Hypothesis (S):", specific)
    print("Initial General Hypothesis (G):", general)

    for i, h in enumerate(concepts):

        if target[i] == "Yes":

            for x in range(len(specific)):
                if h[x] != specific[x]:
                    specific[x] = "?"
                    general[0][x] = "?"

        else:

            for x in range(len(specific)):
                if h[x] != specific[x]:
                    general[0][x] = specific[x]
                else:
                    general[0][x] = "?"

        print(f"\nAfter Training Example {i+1}")
        print("Specific Hypothesis (S):", specific)
        print("General Hypothesis (G):", general)

    return specific, general


specific, general = candidate_elimination(X, y)

print("\nFinal Specific Hypothesis:")
print(specific)

print("\nFinal General Hypothesis:")
for g in general:
    print(g)
