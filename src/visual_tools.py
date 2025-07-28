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


def plot_correlation_matrix(corr_matrix):
    n = len(corr_matrix)
    cell_size = 0.5  # size per cell in inches; increase this to make it bigger
    plt.figure(figsize=(cell_size * n, cell_size * n))
    sns.heatmap(corr_matrix, vmin=-1, vmax=1, annot=True, annot_kws={"size": 10})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
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


def plot_boxplot_melted(df, features_cols, target_col):
    """
    Create box plots for selected features grouped by a target column.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        features_cols (List[str]): List of feature column names.
        target_col (str): Name of the target column.
    """
    df_melted = df.melt(id_vars='diagnosis', value_vars=features_cols, var_name='feature', value_name='value')
    sns.boxplot(x='feature', y='value', hue='diagnosis', data=df_melted)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_violinplot_melted(df, features_cols, target_col):
    """
    Create violin plots for selected features grouped by a target column.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        features_cols (List[str]): List of feature column names.
        target_col (str): Name of the target column.
    """
    
    df_melted = df.melt(id_vars='diagnosis', value_vars=features_cols, var_name='feature', value_name='value')
    sns.violinplot(x='feature', y='value', hue='diagnosis', data=df_melted, split=True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
