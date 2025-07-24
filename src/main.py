from src.visual_tools import plot_features_histogram, plot_correlation_matrix, plot_pairplot
from src.models.Preprocessing import Preprocessing
from src.header import (
	DATA_PATH,
    COLUMNS,
    LABEL_MAPPING
)
import matplotlib.pyplot as plt
import pandas as pd

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
        processor.normalize_features()
        X_train, y_train, X_test, y_test = processor.split_data()

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return


if __name__ == "__main__":
    main()
