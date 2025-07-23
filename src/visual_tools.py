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


def plot_pairplot(df, feature_cols):
    sns.pairplot(df[feature_cols + ["diagnosis"]], hue="diagnosis", plot_kws={'alpha': 0.5})
    plt.suptitle("Pairplot of Features", y=1.02)
    plt.tight_layout()
    plt.savefig("pairplot_features.png")
