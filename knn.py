from data_loader import load_normalized_data


class KNN:
	def __init__(self, n_neighbors=5):
		self.n_neighbors = n_neighbors
		self.X_train = None
		self.y_train = None

	# ------------------------------------------------------------------ #
	#  Fit                                                                 #
	# ------------------------------------------------------------------ #

	def fit(self, X, y):
		"""Mémorise les données d'entraînement."""
		self.X_train = [list(x) for x in X]
		self.y_train = list(y)

	# ------------------------------------------------------------------ #
	#  Predict                                                             #
	# ------------------------------------------------------------------ #

	def _euclidean_distance(self, x1, x2):
		return sum((a - b) ** 2 for a, b in zip(x1, x2)) ** 0.5

	def _predict_one(self, x):
		distances = sorted(
			[(self._euclidean_distance(x, x_train), label)
			 for x_train, label in zip(self.X_train, self.y_train)],
			key=lambda t: t[0],
		)
		k_labels = [label for _, label in distances[:self.n_neighbors]]
		counts = {}
		for label in k_labels:
			counts[label] = counts.get(label, 0) + 1
		return max(counts, key=counts.get)

	def predict(self, X):
		"""Prédit la classe de chaque exemple de X."""
		return [self._predict_one(list(x)) for x in X]

	# ------------------------------------------------------------------ #
	#  Evaluate (cross-validation stratifiée + F1-weighted)               #
	# ------------------------------------------------------------------ #

	def _f1_weighted(self, y_true, y_pred):
		classes = list(set(y_true))
		total = len(y_true)
		f1_weighted = 0.0
		for c in classes:
			tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
			fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
			fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
			precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
			recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
			f1 = (2 * precision * recall / (precision + recall)
				  if (precision + recall) > 0 else 0.0)
			support = sum(1 for yt in y_true if yt == c)
			f1_weighted += f1 * support / total
		return f1_weighted

	def _stratified_kfold_indices(self, y, n_splits):
		"""Retourne n_splits listes d'indices (test sets)."""
		class_indices = {}
		for i, label in enumerate(y):
			if label not in class_indices:
				class_indices[label] = []
			class_indices[label].append(i)

		folds = [[] for _ in range(n_splits)]
		for indices in class_indices.values():
			chunk_size = len(indices) // n_splits
			remainder  = len(indices) % n_splits
			start = 0
			for fold_idx in range(n_splits):
				end = start + chunk_size + (1 if fold_idx < remainder else 0)
				folds[fold_idx].extend(indices[start:end])
				start = end
		return folds

	def evaluate(self, X, y, n_splits=10, scoring="f1_weighted"):
		"""Validation croisée stratifiée — retourne la liste des scores par fold."""
		X_list = [list(x) for x in X]
		y_list = list(y)
		folds   = self._stratified_kfold_indices(y_list, n_splits)
		scores  = []

		for fold_idx in range(n_splits):
			test_set   = set(folds[fold_idx])
			train_idx  = [i for i in range(len(X_list)) if i not in test_set]
			test_idx   = list(test_set)

			X_train = [X_list[i] for i in train_idx]
			y_train = [y_list[i] for i in train_idx]
			X_test  = [X_list[i] for i in test_idx]
			y_test  = [y_list[i] for i in test_idx]

			self.fit(X_train, y_train)
			y_pred = self.predict(X_test)
			score  = self._f1_weighted(y_test, y_pred)
			scores.append(score)

		mean_score = sum(scores) / len(scores)
		std_score  = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5

		print(f"N neighbors : {self.n_neighbors}")
		print(f"{scoring} moyen : {mean_score:.4f} +/- {std_score:.4f}")
		print("-" * 20)
		return scores

	# ------------------------------------------------------------------ #
	#  Grid Search                                                         #
	# ------------------------------------------------------------------ #

	def grid_search(self, X, y, n_neighbors_range=range(1, 20, 2),
					n_splits=10, scoring="f1_weighted"):
		"""Cherche le k optimal par grid search artisanal."""
		best_score = -float("inf")
		best_k     = None

		for k in n_neighbors_range:
			self.n_neighbors = k
			scores     = self.evaluate(X, y, n_splits=n_splits, scoring=scoring)
			mean_score = sum(scores) / len(scores)

			if mean_score > best_score:
				best_score = mean_score
				best_k     = k

		self.n_neighbors = best_k
		print("=" * 100)
		print(f"Optimal k : {best_k}  |  {scoring} : {best_score:.4f}")
		print("=" * 100)
		return best_k, best_score


if __name__ == "__main__":
	X_normalized, Y, _ = load_normalized_data(file_path="bienetre.csv")

	knn = KNN(n_neighbors=5)

	# Grid search sur k impairs de 1 à 19
	best_k, best_score = knn.grid_search(
		X_normalized, Y,
		n_neighbors_range=range(1, 20, 2),
	)

	# Évaluation finale avec le meilleur k
	print(f"\nÉvaluation finale avec k={best_k}")
	knn.evaluate(X_normalized, Y)
