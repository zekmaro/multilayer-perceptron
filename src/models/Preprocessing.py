from typing import List, Tuple
import pandas as pd
import numpy as np


class Preprocessing:
	"""Class for preprocessing data in machine learning tasks."""
	def __init__(self, file_path: str):
		"""
		Initialize the Preprocessing class with the file path.
	
		Args:
			file_path (str): Path to the data file.

		Attributes:
			file_path (str): Path to the data file.
			X (pd.DataFrame): Features data.
			y (pd.Series): Target variable.
		"""
		self.file_path = file_path
		self.df = None
		self.X = None
		self.y = None
	

	def load_data(self, header: bool = True):
		"""
		Load data from the specified file path into a pandas DataFrame.
		
		Raises:
			FileNotFoundError: If the file does not exist.
			ValueError: If the data is not in the expected format.
		"""
		try:
			data = pd.read_csv(self.file_path, header=0 if header else None)
			self.df = data.copy()
			self.X = data
		except FileNotFoundError as e:
			raise FileNotFoundError(f"File not found: {e}")
		except ValueError as e:
			raise ValueError(f"Data format error: {e}")
		except Exception as e:
			raise Exception(f"An error occurred while loading data: {e}")


	def name_columns(self, columns: List[str]):
		"""
		Name the columns of the DataFrame.
		
		Args:
			columns (List[str]): List of column names to set.
		
		Raises:
			ValueError: If the number of columns does not match the DataFrame.
		"""
		if len(columns) != self.X.shape[1]:
			raise ValueError("Number of columns does not match the DataFrame shape.")
		self.X.columns = columns
		self.df.columns = columns
	

	def extract_X_y(
		self, target_column: str, drop_columns: List[str] = None):
		"""
		Extract features and target variable from the DataFrame.
		
		Args:
			target_column (str): Name of the target column.
			drop_columns (List[str]): List of columns to drop from features.
		
		Raises:
			KeyError: If the target column or drop columns are not found in the DataFrame.
		"""
		if target_column not in self.X.columns:
			raise KeyError(f"Target column '{target_column}' not found in DataFrame.")
		if drop_columns:
			for col in drop_columns:
				if col not in self.X.columns:
					raise KeyError(f"Column '{col}' not found in DataFrame.")
			self.X = self.X.drop(columns=drop_columns)
		self.y = self.X[target_column]
		self.X = self.X.drop(columns=[target_column])
	

	def encode_target(self, mapping: dict):
		"""
		Encode the target variable using a mapping dictionary.
		
		Args:
			mapping (dict): Dictionary mapping original values to encoded values.
		
		Raises:
			ValueError: If the target variable is not set or if the mapping is invalid.
		"""
		if self.y is None:
			raise ValueError("Target variable (y) is not set.")
		self.y = self.y.map(mapping)
		if self.y.isnull().any():
			raise ValueError("Invalid mapping, resulting in NaN values in target variable.")


	def normalize_features(self):
		"""
		Normalize the features data using z-score normalization.
		
		Raises:
			ValueError: If the features data is not set.
		"""
		if self.X is None:
			raise ValueError("Features data (X) is not set.")
		self.X = (self.X - self.X.mean()) / self.X.std()


	def split_data(
		self,
		split_ratio: float = 0.8,
	) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
		"""
		Split the data into training and testing sets.

		Args:
			split_ratio (float): Proportion of data to use for training.
			shuffle (bool): Whether to shuffle the data before splitting.
		
		Returns:
			tuple: Training and testing sets (X_train, X_test, y_train, y_test).
		"""
		indices = np.arange(len(self.X))
		np.random.shuffle(indices)
		split_index = int(len(self.X) * split_ratio)
		X_train = self.X.iloc[indices[:split_index]]
		y_train = self.y.iloc[indices[:split_index]]
		X_test = self.X.iloc[indices[split_index:]]
		y_test = self.y.iloc[indices[split_index:]]
		return X_train, y_train, X_test, y_test


	def check_nulls(self, df):
		return df.isnull().values.any()


	def preprocess(self):
		pass
