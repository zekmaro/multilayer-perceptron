from src.visual_tools import plot_features_histogram, plot_correlation_matrix, plot_pairplot, plot_boxplot_melted, plot_violinplot_melted, density_plot
from src.models.Preprocessing import Preprocessing
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
        df['diagnosis'] = processor.y
        print(df.describe().T)
        # plot_pairplot(df, COLUMNS, 'diagnosis', 'pairplot.png')

        # plot_value_distribution(df, 'diagnosis')

        corr_matrix = df.drop(columns=['diagnosis']).corr()

        # plot_correlation_matrix(corr_matrix)

        # df.hist(figsize=(15, 10), color='orange')
        # plt.tight_layout()
        # plt.savefig("histograms.png")
    
        # corr_groups = processor.group_correlated_features(corr_matrix)
        # print("Correlated feature groups:")
        # pprint.pprint(corr_groups)

        # df_mean = pd.DataFrame(df, columns=MEAN_FEATURES + ['diagnosis'])
        # sns.pairplot(df_mean, hue="diagnosis", diag_kind='kde',palette = 'rocket')
        # plt.suptitle("Pairplot of Mean Features", y=1.02)
        # plt.tight_layout()
        # plt.savefig("mean_features_pairplot.png")

        # for group in GROUPED_FEATURES:
            # plot_boxplot_melted(df, group, 'diagnosis')
            # plot_violinplot_melted(df, group, 'diagnosis')

        df = df.drop(columns=DROP_WORST + DROP_PERIMETER + DROP_AREA + DROP_CONCAVITY + DROP_CONCAVE_POINTS)
        print("Data after dropping worst features and others:")
        print(df.columns)

        # plot_correlation_matrix(df.drop(columns=['diagnosis', 'id']).corr())

        X = df.drop(columns=['diagnosis', 'id'])
        y = df['diagnosis']

        processor.normalize_features()
        X_train, y_train, X_test, y_test = processor.split_data()

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return


if __name__ == "__main__":
    main()
