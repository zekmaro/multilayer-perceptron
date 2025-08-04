from src.models.Preprocessing import Preprocessing
from src.header import COLUMNS, DROP_COLUMNS, LABEL_MAPPING, DATA_PATH
import numpy as np

def main():
    processor = Preprocessing(DATA_PATH)
    processor.load_data(header=True)
    processor.name_columns(COLUMNS)
    processor.extract_X_y(target_column="diagnosis", drop_columns=DROP_COLUMNS)
    processor.encode_target(LABEL_MAPPING)
    processor.normalize_features()

    X_train, y_train, X_test, y_test = processor.split_data()

    # Save splits
    np.save("saved/X_train.npy", X_train)
    np.save("saved/y_train.npy", y_train)
    np.save("saved/X_test.npy", X_test)
    np.save("saved/y_test.npy", y_test)

    print("Data successfully split and saved.")

if __name__ == "__main__":
    main()
