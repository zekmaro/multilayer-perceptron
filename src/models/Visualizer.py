import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Visualizer:
    def __init__(self, df):
        pass

    def plot_features_histogram(self, df, feature_cols):
        for col in feature_cols:
            plt.figure(figsize=(6, 4))
            plt.hist(df[col], bins=30, edgecolor='black')
            plt.title(f"Histogram for {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            plt.show()


    def plot_correlation_matrix(self, corr_matrix):
        n = len(corr_matrix)
        cell_size = 0.5  # size per cell in inches; increase this to make it bigger
        plt.figure(figsize=(cell_size * n, cell_size * n))
        sns.heatmap(corr_matrix, vmin=-1, vmax=1, annot=True, annot_kws={"size": 10})
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("correlation_matrix.png")


    def plot_pairplot(self, data, target_col, title, filename):
        """Create a pairplot for selected features and a target column."""
        sns.pairplot(data, hue=target_col, diag_kind='kde', palette='rocket')
        plt.suptitle(title, y=1.02)
        plt.tight_layout()
        plt.savefig(filename)


    def plot_boxplot_melted(self, df, features_cols, target_col):
        """
        Create box plots for selected features grouped by a target column.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            features_cols (List[str]): List of feature column names.
            target_col (str): Name of the target column.
        """
        df_melted = df.melt(id_vars=target_col, value_vars=features_cols, var_name='feature', value_name='value')
        sns.boxplot(x='feature', y='value', hue=target_col, data=df_melted)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("boxplot_melted.png")


    def plot_violinplot_melted(self, df, features_cols, target_col):
        """
        Create violin plots for selected features grouped by a target column.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            features_cols (List[str]): List of feature column names.
            target_col (str): Name of the target column.
        """

        df_melted = df.melt(id_vars=target_col, value_vars=features_cols, var_name='feature', value_name='value')
        sns.violinplot(x='feature', y='value', hue=target_col, data=df_melted, split=True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


    def density_plot(self, df, feature, target='diagnosis'):
        """
        Create a density plot for a feature grouped by the target column.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            feature (str): Name of the feature to plot.
            target (str): Name of the target column.
        """
        sns.kdeplot(data=df, x=feature, hue=target, fill=True, common_norm=False)
        plt.title(f"Density Plot of {feature} by {target}")
        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def count_plot(self, df, feature, target='diagnosis'):
        """
        Create a count plot for a feature grouped by the target column.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            feature (str): Name of the feature to plot.
            target (str): Name of the target column.
        """
        sns.countplot(data=df, x=feature, hue=target)
        plt.title(f"Count Plot of {feature} by {target}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_loss_history(self, loss_history):
        """
        Plot the loss history over epochs.

        Args:
            loss_history (list): List of loss values recorded during training.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history, label='Loss', color='blue')
        plt.title('Loss History')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_accuracy_history(self, accuracy_history):
        """
        Plot the accuracy history over epochs.

        Args:
            accuracy_history (list): List of accuracy values recorded during training.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(accuracy_history, label='Accuracy', color='green')
        plt.title('Accuracy History')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_value_distribution(self, df, feature, target):
        classes = df[target].unique()
        plt.figure(figsize=(8, 6))
        for c in classes:
            subset = df[df[target] == c]
            plt.hist(subset[feature], bins=30, alpha=0.5, label=f"{c} ({len(subset)})")
        plt.title(f"Distribution of {feature} by {target}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    

    def plot_histograms(self, df):
        """
        Create histograms for all features in the DataFrame.
        """
        plt.figure(figsize=(15, 10))
        df.hist(bins=30, color='orange', edgecolor='black', figsize=(15, 10))
        plt.tight_layout()
        plt.show()