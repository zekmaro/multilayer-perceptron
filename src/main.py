from src.visual_tools import plot_features_histogram, plot_correlation_matrix, plot_pairplot
from src.models.Preprocessing import Preprocessing
from src.header import (
	DATA_PATH,
    COLUMNS,
    LABEL_MAPPING
)
import matplotlib.pyplot as plt
import pandas as pd


def separation_score(df, feature, target='diagnosis'):
    classes = df[target].unique()
    means = [df[df[target] == c][feature].mean() for c in classes]
    stds = [df[df[target] == c][feature].std() for c in classes]
    return abs(means[0] - means[1]) / (stds[0] + stds[1] + 1e-6)  # small epsilon to avoid div by 0


def plot_value_distribution(df, feature, target='diagnosis'):
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

def main():
    try:
        processor = Preprocessing(DATA_PATH)
        processor.load_data(header=True)
        processor.name_columns(COLUMNS)
        processor.extract_X_y(target_column="diagnosis", drop_columns=["id"])
        processor.encode_target(LABEL_MAPPING)
        print(processor.check_nulls(processor.df))
        df = processor.df.copy()
        print(df.nunique())
        plot_value_distribution(df, 'diagnosis')
        df['diagnosis'] = processor.y
        plot_correlation_matrix(df.corr())

        print(df.describe())
        processor.normalize_features()
        X_train, y_train, X_test, y_test = processor.split_data()

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return


if __name__ == "__main__":
    main()
