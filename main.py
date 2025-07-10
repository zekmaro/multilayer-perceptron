import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from header import DATA_PATH, AMOUNT_COLUMNS


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
    df = load_data(DATA_PATH)
    if df is not None:
        print("Data loaded successfully:")
        df.columns = ["id", "diagnosis"] + [f"feature_{i}" for i in range(0, AMOUNT_COLUMNS - 2)]
        print(df.head(10))
        print(df.describe())
        print(df.info())
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        # plot_features_histogram(df, feature_cols)
        norm_data = normalize_features(df[feature_cols].to_numpy())
        corr = np.corrcoef(norm_data, rowvar=False)
        plot_correlation_matrix(corr, feature_cols)
        plot_pairplot(df, feature_cols)
        
    else:
        print("Failed to load data.")

if __name__ == "__main__":
    main()
