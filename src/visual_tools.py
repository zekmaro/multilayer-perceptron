import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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


def plot_correlation_matrix(corr_matrix, feature_names):
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title("Correlation Matrix of Features")
    plt.xticks(ticks=np.arange(len(feature_names)), labels=feature_names, rotation=45)
    plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)
    plt.tight_layout()
    plt.show()


def plot_pairplot(df, feature_cols, target_col, filename):
    """
    Create a pairplot for selected features and a target column.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        feature_cols (List[str]): List of feature column names.
        target_col (str): Name of the target column.
        filename (str): File path to save the plot.

    Raises:
        ValueError: If the target column is not in the DataFrame.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # Avoid adding the target twice
    selected_cols = [col for col in feature_cols if col != target_col]
    plot_df = df[selected_cols + [target_col]].copy()

    sns.pairplot(plot_df, hue=target_col, plot_kws={'alpha': 0.5})
    plt.suptitle("Pairplot of Selected Features", y=1.02)
    plt.tight_layout()
    plt.savefig(filename)
