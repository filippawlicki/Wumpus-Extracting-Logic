import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("../datasets/fol_3pit_random_map_dataset.csv")
X = df[["stench", "breeze", "glitter", "bump", "scream", "hasgold", "on_entrance"]]
y = df["action"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(max_depth=36, min_impurity_decrease=0.001)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save logic extraction rules
unique_classes = sorted(y.unique())
class_names_map = {
    0: "Move Forward",
    1: "Turn",
    2: "Grab",
    3: "Climb",
    4: "Shoot"
}
class_names = [class_names_map[i] for i in unique_classes]

tree_rules = export_text(clf, feature_names=list(X.columns), class_names=class_names)

with open("decision_tree_rules_random_map_3pit_fol.txt", "w") as f:
    f.write(tree_rules)

print("Decision tree rules saved to decision_tree_rules.txt")