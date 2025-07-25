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


def main():
    try:
        processor = Preprocessing(DATA_PATH)
        processor.load_data(header=True)
        processor.name_columns(COLUMNS)
        processor.extract_X_y(target_column="diagnosis", drop_columns=["id"])
        processor.encode_target(LABEL_MAPPING)

        df = processor.df.copy()
        df['diagnosis'] = processor.y

        best_features, redundant_features = processor.select_relevant_features(df)
        print(f"Best features: {best_features}")
        print(f"Redundant features: {redundant_features}")
        df = df.drop(columns=redundant_features + ["id"], errors='ignore')
        # plot_pairplot(df, df.columns.to_list(), "diagnosis", "plot.png")
        processor.X = df.drop(columns=["diagnosis"])
        processor.y = df["diagnosis"]

        scores = {}
        for feature in df.columns:
            if feature != 'diagnosis':
                scores[feature] = separation_score(df, feature)
                
        # Sort by score
        import pprint
        pprint.pprint(sorted(scores.items(), key=lambda x: x[1], reverse=True))

        processor.normalize_features()
        X_train, y_train, X_test, y_test = processor.split_data()

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return


if __name__ == "__main__":
    main()
