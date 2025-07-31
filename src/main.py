from src.models.Preprocessing import Preprocessing
from src.models.DenseLayer import DenseLayer
from src.models.Visualizer import Visualizer
from src.models.Model import Model
from src.header import (
    DROP_CONCAVE_POINTS,
    GROUPED_FEATURES,
    DROP_PERIMETER,
    DROP_CONCAVITY,
    LABEL_MAPPING,
    MEAN_FEATURES,
    DROP_COLUMNS,
    CORR_GROUPS,
    DROP_WORST,
	DATA_PATH,
    DROP_AREA,
    COLUMNS,
)
import pandas as pd


def load_and_prepare_data(processor, target_column) -> None:
    """
    Load and prepare the dataset for analysis.
    """
    processor.load_data(header=True)
    processor.name_columns(COLUMNS)
    processor.extract_X_y(target_column=target_column, drop_columns=DROP_COLUMNS)
    processor.encode_target(LABEL_MAPPING)
    processor.check_nulls(processor.df)
    processor.check_uniqueness(processor.df)
    print(processor.df.describe().T)


def explore_dataset(df, visualizer):
    """Explore the dataset using various visualizations."""
    visualizer.plot_correlation_matrix(df.corr())
    visualizer.plot_histograms(df)
    df_mean = pd.DataFrame(df, columns=MEAN_FEATURES + ["diagnosis"])
    visualizer.plot_pairplot(df_mean, "diagnosis", "Pairplot of Mean Features", 'plots/mean_features_pairplot.png')


def main():
    processor = Preprocessing(DATA_PATH)
    load_and_prepare_data(processor, target_column="diagnosis")

    df = processor.df.copy()
    df['diagnosis'] = processor.y

    visualizer = Visualizer(processor.df)
    explore_dataset(df, visualizer)

    processor.X = df.drop(columns=['diagnosis', 'id'])
    processor.y = df['diagnosis']

    processor.normalize_features()
    X_train, y_train, X_test, y_test = processor.split_data()
    print(X_train.shape)
    print(y_train.shape)

    model = Model()
    layers = [
        DenseLayer(units=16, activation_name='relu', input_dim=X_train.shape[1]),
        DenseLayer(units=8, activation_name='relu'),
        DenseLayer(units=2, activation_name='softmax'),
    ]
    network = model.create_network(layers)
    model.fit(network, X_train, y_train.to_numpy(), epochs=100, batch_size=32, learning_rate=0.001)

    accuracy = model.get_model_accuracy(network, X_test, y_test.to_numpy())
    print(f"Model accuracy: {accuracy:.2f}")

    visualizer.plot_loss_history(model.loss_history)
    visualizer.plot_accuracy_history(model.accuracy_history)


if __name__ == "__main__":
    main()
