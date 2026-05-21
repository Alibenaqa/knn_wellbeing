from data_loader import load_normalized_data
from benchmark_model import find_optimal_model

if __name__ == "__main__":
	X_normalized, Y, standard_scaler_object = load_normalized_data(file_path="bienetre.csv")

	best_model_name, best_params, best_score = find_optimal_model(
		X_normalized=X_normalized,
		Y=Y,
		n_splits=10,
		scoring="f1_weighted",
	)
