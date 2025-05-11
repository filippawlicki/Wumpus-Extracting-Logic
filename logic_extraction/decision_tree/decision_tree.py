import pandas as pd
import pydot
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import copy

def compute_parents(tree):
    parents = [-1] * tree.node_count  # Initialize parent array with -1
    for i in range(tree.node_count):
        left_child = tree.children_left[i]
        right_child = tree.children_right[i]
        if left_child != -1:  # If the node has a left child
            parents[left_child] = i
        if right_child != -1:  # If the node has a right child
            parents[right_child] = i
    return parents

def prune_tree(org_tree: DecisionTreeClassifier):
    tree = copy.deepcopy(org_tree.tree_)
    parents = compute_parents(tree)  # Compute parent nodes

    # Prune the tree by removing leafs if they are the same class as the parent node
    for node in range(tree.node_count):
        if tree.children_left[node] == tree.children_right[node]:  # Leaf node
            parent = parents[node]
            if parent != -1 and tree.value[node].argmax() == tree.value[parent].argmax():
                # Prune the leaf node
                tree.children_left[parent] = -1
                tree.children_right[parent] = -1
                tree.value[parent] = tree.value[node]

    org_tree = copy.deepcopy(org_tree)
    org_tree.tree_ = tree

    return org_tree

# Load dataset
df = pd.read_csv("../datasets/fol_3pit_random_map_dataset.csv")
X = df[["stench", "breeze", "glitter", "bump", "scream", "hasgold", "on_entrance"]]
y = df["action"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(max_depth=36)
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

pruned_tree = prune_tree(clf)

dot_data = export_graphviz(
    pruned_tree,
    feature_names=list(X.columns),
    class_names=class_names,
    filled=True,
    label="none",  # Exclude gini, samples, and value
    impurity=False  # Exclude gini
)
(graph,) = pydot.graph_from_dot_data(dot_data)
graph.write_png("decision_tree.png")

print("Decision tree rules saved to decision_tree.png")