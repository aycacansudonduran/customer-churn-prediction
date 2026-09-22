"""
Sıfırdan (from-scratch) NumPy ile Lojistik Regresyon ve Karar Ağacı / Random Forest.
Harici bir ML kütüphanesi (scikit-learn vb.) kullanılmamıştır — amaç, algoritmaların
iç mantığını (gradyan inişi, Gini impurity, bootstrap agregasyonu) göstermektir.
"""
import numpy as np


# ----------------------------------------------------------------------
# Lojistik Regresyon (Gradyan İnişi ile)
# ----------------------------------------------------------------------
class LogisticRegressionScratch:
    def __init__(self, lr=0.1, n_iter=2000, l2=0.01):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iter):
            z = X @ self.weights + self.bias
            preds = self._sigmoid(z)
            error = preds - y

            grad_w = (X.T @ error) / n_samples + (self.l2 / n_samples) * self.weights
            grad_b = error.mean()

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

            eps = 1e-9
            loss = -np.mean(y * np.log(preds + eps) + (1 - y) * np.log(1 - preds + eps))
            self.loss_history.append(loss)
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ----------------------------------------------------------------------
# Karar Ağacı (CART, Gini impurity)
# ----------------------------------------------------------------------
class _TreeNode:
    __slots__ = ("feature_idx", "threshold", "left", "right", "value")

    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def _gini(y):
    if len(y) == 0:
        return 0.0
    p = y.mean()
    return 1 - p ** 2 - (1 - p) ** 2


class DecisionTreeScratch:
    def __init__(self, max_depth=6, min_samples_split=20, n_thresholds=12, max_features=None, rng=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_thresholds = n_thresholds
        self.max_features = max_features
        self.rng = rng or np.random.default_rng()
        self.root = None

    def fit(self, X, y):
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        n_samples, n_features = X.shape
        if (depth >= self.max_depth or n_samples < self.min_samples_split
                or len(np.unique(y)) == 1):
            return _TreeNode(value=y.mean() if n_samples else 0.0)

        feature_indices = np.arange(n_features)
        if self.max_features:
            feature_indices = self.rng.choice(n_features, size=self.max_features, replace=False)

        best_gain, best_feat, best_thr = -1, None, None
        parent_gini = _gini(y)

        for feat_idx in feature_indices:
            col = X[:, feat_idx]
            quantiles = np.linspace(0.1, 0.9, self.n_thresholds)
            thresholds = np.unique(np.quantile(col, quantiles))
            for thr in thresholds:
                left_mask = col <= thr
                right_mask = ~left_mask
                n_left, n_right = left_mask.sum(), right_mask.sum()
                if n_left == 0 or n_right == 0:
                    continue
                gini_left = _gini(y[left_mask])
                gini_right = _gini(y[right_mask])
                weighted_gini = (n_left * gini_left + n_right * gini_right) / n_samples
                gain = parent_gini - weighted_gini
                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, feat_idx, thr

        if best_feat is None or best_gain <= 1e-7:
            return _TreeNode(value=y.mean())

        left_mask = X[:, best_feat] <= best_thr
        left_node = self._build(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return _TreeNode(feature_idx=best_feat, threshold=best_thr, left=left_node, right=right_node)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict_proba(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ----------------------------------------------------------------------
# Random Forest (Bootstrap agregasyonu + rastgele öznitelik alt kümesi)
# ----------------------------------------------------------------------
class RandomForestScratch:
    def __init__(self, n_trees=25, max_depth=6, min_samples_split=20, max_features=None, seed=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.seed = seed
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        max_feat = self.max_features or max(1, int(np.sqrt(n_features)))
        rng = np.random.default_rng(self.seed)
        self.trees = []
        for i in range(self.n_trees):
            boot_idx = rng.integers(0, n_samples, size=n_samples)
            X_boot, y_boot = X[boot_idx], y[boot_idx]
            tree = DecisionTreeScratch(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                max_features=max_feat, rng=np.random.default_rng(self.seed + i),
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        preds = np.array([tree.predict_proba(X) for tree in self.trees])
        return preds.mean(axis=0)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importances(self, n_features):
        """Basit permütasyon-benzeri önem: her ağaçtaki split sayısına göre ağırlıklandırma."""
        counts = np.zeros(n_features)

        def walk(node):
            if node.value is not None:
                return
            counts[node.feature_idx] += 1
            walk(node.left)
            walk(node.right)

        for tree in self.trees:
            walk(tree.root)
        total = counts.sum()
        return counts / total if total > 0 else counts


# ----------------------------------------------------------------------
# Değerlendirme metrikleri (sıfırdan)
# ----------------------------------------------------------------------
def confusion_matrix(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return np.array([[tn, fp], [fn, tp]])


def classification_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    accuracy = (tp + tn) / cm.sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm}


def roc_curve_scratch(y_true, y_scores, n_thresholds=100):
    thresholds = np.linspace(0, 1, n_thresholds)
    tpr_list, fpr_list = [], []
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    for thr in thresholds:
        y_pred = (y_scores >= thr).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        tpr_list.append(tp / n_pos if n_pos > 0 else 0)
        fpr_list.append(fp / n_neg if n_neg > 0 else 0)
    fpr = np.array(fpr_list)[::-1]
    tpr = np.array(tpr_list)[::-1]
    auc = np.trapz(tpr, fpr)
    return fpr, tpr, auc
