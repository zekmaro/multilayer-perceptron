DATA_PATH = "data/data.csv"

COLUMNS = [
    "id", "diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

LABEL_MAPPING = {
	"M": 1,  # Malignant
	"B": 0   # Benign
}

CORR_GROUPS = [
    ["radius_mean", "perimeter_mean", "area_mean"],
    ['radius_se', 'perimeter_se', 'area_se'],
    ['compactness_mean', 'concavity_mean', 'compactness_worst'],
    ['smoothness_mean', 'smoothness_worst'],
    ['texture_mean', 'texture_worst'],
    ['fractal_dimension_mean', 'fractal_dimension_worst'],
    ['concave_points_mean', 'concave_points_worst'],
    ['symmetry_mean', 'symmetry_worst']
]

GROUPED_FEATURES = [
    ["radius_mean", "radius_se", "radius_worst"],
    ["texture_mean", "texture_se", "texture_worst"],
    ["perimeter_mean", "perimeter_se", "perimeter_worst"],
    ["area_mean", "area_se", "area_worst"],
    ["smoothness_mean", "smoothness_se", "smoothness_worst"],
    ["compactness_mean", "compactness_se", "compactness_worst"],
    ["concavity_mean", "concavity_se", "concavity_worst"],
    ["concave_points_mean", "concave_points_se", "concave_points_worst"]
]

MEAN_FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean"
]
