from src.models.Preprocessing import Preprocessing
import matplotlib.pyplot as plt
from src.header import (
	DATA_PATH,
    COLUMNS,
    LABEL_MAPPING
)
import seaborn as sns
import pandas as pd
import numpy as np


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


def plot_features_histogram(df, feature_cols):
    for col in feature_cols:
        plt.figure(figsize=(6, 4))
        plt.hist(df[col], bins=30, edgecolor='black')
        plt.title(f"Histogram for {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def normalize_features(features_data):
    mean = features_data.mean(axis=0)
    std = features_data.std(axis= 0)
    norm_data = (features_data - mean) / std
    return norm_data


def plot_correlation_matrix(corr_matrix, feature_names):
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title("Correlation Matrix of Features")
    plt.xticks(ticks=np.arange(len(feature_names)), labels=feature_names, rotation=45)
    plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)
    plt.tight_layout()
    plt.show()


def plot_pairplot(df, feature_cols):
    sns.pairplot(df[feature_cols + ["diagnosis"]], hue="diagnosis", plot_kws={'alpha': 0.5})
    plt.suptitle("Pairplot of Features", y=1.02)
    plt.tight_layout()
    plt.savefig("pairplot_features.png")


def main():
    try:
        processor = Preprocessing(DATA_PATH)
        processor.load_data(header=True)
        processor.name_columns(COLUMNS)
        processor.extract_X_y(target_column="diagnosis", drop_columns=["id"])
        processor.encode_target(LABEL_MAPPING)
        processor.normalize_features()
        X_train, y_train, X_test, y_test = processor.split_data(split_ratio=0.8)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return


if __name__ == "__main__":
    main()
