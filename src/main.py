from src.models.Preprocessing import Preprocessing
from src.models.Model import Model
from src.models.DenseLayer import DenseLayer
from src.models.Visualizer import Visualizer
from src.header import (
	DATA_PATH,
    COLUMNS,
    LABEL_MAPPING,
    MEAN_FEATURES,
    CORR_GROUPS,
    GROUPED_FEATURES,
    DROP_WORST,
    DROP_PERIMETER,
    DROP_AREA,
    DROP_CONCAVITY,
    DROP_CONCAVE_POINTS
)
import matplotlib.pyplot as plt
import pandas as pd
import pprint
import seaborn as sns
import numpy as np


def separation_score(df, feature, target='diagnosis'):
    classes = df[target].unique()
    means = [df[df[target] == c][feature].mean() for c in classes]
    stds = [df[df[target] == c][feature].std() for c in classes]
    return abs(means[0] - means[1]) / (stds[0] + stds[1] + 1e-6)  # small epsilon to avoid div by 0


def get_model_accuracy(model, network, X_test, y_test):
    y_pred = model.predict(network, X_test)
    pred_classes = np.argmax(y_pred, axis=1)
    accuracy = np.mean(pred_classes == y_test)
    return accuracy


def main():
    processor = Preprocessing(DATA_PATH)
    visualizer = Visualizer(processor.df)
    processor.load_data(header=True)
    processor.name_columns(COLUMNS)
    processor.extract_X_y(target_column="diagnosis", drop_columns=["id"])
    processor.encode_target(LABEL_MAPPING)
    processor.check_nulls(processor.df)
    df = processor.df.copy()
    processor.check_uniqueness(df)
    df['diagnosis'] = processor.y
    print(df.describe().T)

    visualizer.plot_pairplot(df, 'diagnosis', "Pairplot of Features", 'pairplot.png')
    visualizer.plot_value_distribution('diagnosis')

    corr_matrix = df.drop(columns=['diagnosis']).corr()
    visualizer.plot_correlation_matrix(corr_matrix)

    visualizer.plot_histograms()

    df_mean = pd.DataFrame(df, columns=MEAN_FEATURES + ['diagnosis'])
    visualizer.plot_pairplot(df_mean, 'diagnosis', "Pairplot of Mean Features", 'mean_features_pairplot.png')

    for group in GROUPED_FEATURES:
        visualizer.plot_boxplot_melted(group, 'diagnosis')
        visualizer.plot_violinplot_melted(group, 'diagnosis')

    processor.X = df.drop(columns=['diagnosis', 'id'])
    processor.y = df['diagnosis']

    processor.normalize_features()
    X_train, y_train, X_test, y_test = processor.split_data()
    print(X_train.shape)
    print(y_train.shape)

    layers = [
        DenseLayer(units=16, activation_name='relu', input_dim=X_train.shape[1]),
        DenseLayer(units=8, activation_name='relu'),
        DenseLayer(units=2, activation_name='softmax'),
    ]
    model = Model()
    network = model.create_network(layers)
    model.fit(network, X_train, y_train.to_numpy(), epochs=100, batch_size=32, learning_rate=0.001)

    accuracy = get_model_accuracy(model, network, X_test, y_test.to_numpy())
    print(f"Model accuracy: {accuracy:.2f}")

    visualizer.plot_loss_history(model.loss_history)
    visualizer.plot_accuracy_history(model.accuracy_history)


if __name__ == "__main__":
    main()
