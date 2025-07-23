from src.models.Preprocessing import Preprocessing
from src.header import (
	DATA_PATH,
    COLUMNS,
    LABEL_MAPPING
)


def main():
    try:
        processor = Preprocessing(DATA_PATH)
        processor.load_data(header=True)
        processor.name_columns(COLUMNS)
        processor.extract_X_y(target_column="diagnosis", drop_columns=["id"])
        processor.encode_target(LABEL_MAPPING)
        X_train, y_train, X_test, y_test = processor.split_data(split_ratio=0.8)
        print(X_train.shape)
        print(X_train.head())
        print(X_train.describe())
        print(X_train.info())
        processor.normalize_features()

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return


if __name__ == "__main__":
    main()
