import pandas
from sklearn.preprocessing import StandardScaler


def load_data(file_path, target_col="target"):
	if ".csv" in file_path:
		df = pandas.read_csv(file_path)
	elif ".xlsx" in file_path:
		df = pandas.read_excel(file_path)
	else:
		raise Exception("The file path must be a csv or excel file !")

	if target_col not in df.columns:
		raise Exception("The target column does not exist")

	print(df[target_col].value_counts())
	X = df.drop(columns=[target_col])
	Y = df[target_col]
	return X, Y


def normalize(X):
	standard_scaler_object = StandardScaler()
	X_normalized = standard_scaler_object.fit_transform(X)
	return standard_scaler_object, X_normalized


def load_normalized_data(file_path, target_col="target"):
	print("Loading data phase")
	X, Y = load_data(file_path, target_col=target_col)
	standard_scaler_object, X_normalized = normalize(X)
	print("Success : Loading data")
	print("-" * 20)
	return X_normalized, Y, standard_scaler_object


if __name__ == "__main__":
	X_normalized, Y, standard_scaler_object = load_normalized_data(file_path="bienetre.csv")
