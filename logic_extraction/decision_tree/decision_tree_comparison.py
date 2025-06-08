import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
from itertools import product
from decision_tree import prune_tree

# Load datasets
df1 = pd.read_csv("../datasets/fol_3pit_random_map_dataset.csv")
X1 = df1[["stench", "breeze", "glitter", "bump", "scream", "hasgold", "on_entrance"]]
y1 = df1["action"]

# Split data and train two models with different seeds
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

df2 = pd.read_csv("../datasets/dqn_3pit_random_map_dataset.csv")
X2 = df2[["stench", "breeze", "glitter", "bump", "scream", "hasgold", "on_entrance"]]
y2 = df2["action"]

# Split data and train two models with different seeds
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

clf1 = DecisionTreeClassifier(max_depth=36, min_samples_leaf=10)
clf1.fit(X_train1, y_train1)

clf2 = DecisionTreeClassifier(max_depth=36, min_samples_leaf=10)
clf2.fit(X_train2, y_train2)

# Prune the trees
clf1 = prune_tree(clf1)
clf2 = prune_tree(clf2)

class_names = {
  0: "Move Forward",
  1: "Turn",
  2: "Grab",
  3: "Climb",
  4: "Shoot"
}

# Helper function to extract rules from a decision tree
def extract_rules(tree: DecisionTreeClassifier, feature_names):
  tree_ = tree.tree_
  feature_name = [
    feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
    for i in tree_.feature
  ]

  paths = []

  nodes = range(0, tree_.node_count)
  ls = tree_.children_left
  rs = tree_.children_right
  classes = [[list(e).index(max(e)) for e in v] for v in tree_.value]

  leaves = [(ls[i] == rs[i]) for i in nodes]

  LEAF = -1
  for i in reversed(nodes):
    if leaves[i]:
      continue
    if leaves[ls[i]] and leaves[rs[i]] and classes[ls[i]] == classes[rs[i]]:
      ls[i] = rs[i] = LEAF
      leaves[i] = True

  def recurse(node, path):
    if node == LEAF:  # Skip pruned nodes
      return
    if not leaves[node]:  # If not a leaf
      name = feature_name[node]
      threshold = tree_.threshold[node]
      recurse(ls[node], path + [(name, "<=", threshold)])
      recurse(rs[node], path + [(name, ">", threshold)])
    else:  # If a leaf
      action = class_names[classes[node][0]]
      paths.append((frozenset(path), action))

  recurse(0, [])
  return paths

# Extract rules from both trees
rules_1 = extract_rules(clf1, X1.columns)
rules_2 = extract_rules(clf2, X2.columns)

# Compute Jaccard similarity only for rules with the same action
def jaccard_similarity_with_action(rule_a, action_a, rule_b, action_b):
    if action_a != action_b:
        return 0  # Different actions, similarity is 0
    return len(rule_a & rule_b) / len(rule_a | rule_b)

# Find rule pairs with Jaccard similarity > 2/3 and the same action
similar_rules = []
for (rule_a, action_a), (rule_b, action_b) in product(rules_1, rules_2):
    sim = jaccard_similarity_with_action(rule_a, action_a, rule_b, action_b)
    if sim >= 0.6:
        similar_rules.append((rule_a, rule_b, sim, action_a))

# Format rules for readability
def readable_rule_with_action(rule, action):
    return f"({' AND '.join([f'{f} {op} {round(val, 2)}' for f, op, val in rule])}) -> Action: {action}"

readable_similar_rules = [
    (readable_rule_with_action(a, action), readable_rule_with_action(b, action), round(sim, 2))
    for a, b, sim, action in similar_rules
]

# Calculate max depth and average depth of the tree
def calculate_tree_depths(tree: DecisionTreeClassifier):
    tree_ = tree.tree_
    depths = []

    def recurse(node, depth):
        if tree_.children_left[node] == tree_.children_right[node]:  # Leaf node
            depths.append(depth)
        else:
            if tree_.children_left[node] != -1:
                recurse(tree_.children_left[node], depth + 1)
            if tree_.children_right[node] != -1:
                recurse(tree_.children_right[node], depth + 1)

    recurse(0, 0)
    max_depth = max(depths)
    avg_depth = sum(depths) / len(depths)
    return max_depth, avg_depth

# Print similar rules
for rule_a, rule_b, sim in readable_similar_rules:
    print(f"Rule A: {rule_a}\nRule B: {rule_b}\nSimilarity: {sim}")
    print("-" * 80)

print(f"Total similar rules found: {len(readable_similar_rules)}")
print("Total rules in Model 1:", len(rules_1))
print("Total rules in Model 2:", len(rules_2))

# Calculate depths for both trees
max_depth_1, avg_depth_1 = calculate_tree_depths(clf1)
max_depth_2, avg_depth_2 = calculate_tree_depths(clf2)

# Print the results
print(f"Model 1 - Max Depth: {max_depth_1}, Average Depth: {avg_depth_1:.2f}")
print(f"Model 2 - Max Depth: {max_depth_2}, Average Depth: {avg_depth_2:.2f}")
