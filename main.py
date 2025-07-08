import pandas as pd
from header import DATA_PATH


def load_data(file_path):
	"""
	Load data from a CSV file into a pandas DataFrame.
	
	Parameters:
	file_path (str): The path to the CSV file.
	
	Returns:
	pd.DataFrame: The loaded data as a DataFrame.
	"""
	try:
		data = pd.read_csv(file_path)
		return data
	except Exception as e:
		print(f"Error loading data: {e}")
		return None


def main():
	data = load_data(DATA_PATH)
	if data is not None:
		print("Data loaded successfully:")
		print(data.head())
		print(data.describe())
	else:
		print("Failed to load data.")

if __name__ == "__main__":
	main()