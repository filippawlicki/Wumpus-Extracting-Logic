import pandas as pd
import pydot
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import copy

def prune_tree(org_tree: DecisionTreeClassifier):
    tree = copy.deepcopy(org_tree.tree_)

    nodes = range(0, tree.node_count)
    ls = tree.children_left
    rs = tree.children_right
    classes = [[list(e).index(max(e)) for e in v] for v in tree.value]

    leaves = [(ls[i] == rs[i]) for i in nodes]

    LEAF = -1
    for i in reversed(nodes):
        if leaves[i]:
            continue
        if leaves[ls[i]] and leaves[rs[i]] and classes[ls[i]] == classes[rs[i]]:
            ls[i] = rs[i] = LEAF
            leaves[i] = True

    pruned_tree = copy.deepcopy(org_tree)
    pruned_tree.tree_ = tree

    return pruned_tree

# Load dataset
df = pd.read_csv("../datasets/dqn_3pit_random_map_dataset.csv")
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