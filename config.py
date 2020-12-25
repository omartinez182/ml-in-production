

TRAINING_DATA_FILE = 'https://raw.githubusercontent.com/pkmklong/Breast-Cancer-Wisconsin-Diagnostic-DataSet/master/data.csv'
PIPELINE_NAME = 'breast_cancer_classification'

TARGET = 'diagnosis'

FEATURES = ['id','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']

DROP_FEATURES = 'id'

NUMERICAL_VARS_WITH_NA = []

CATEGORICAL_VARS_WITH_NA = []

TEMPORAL_VARS = []

NUMERICAL_LOG_VARS = ['concavity_mean', 'concave points_worst', 'compactness_worst','area_mean','perimeter_mean','radius_mean']

CATEGORICAL_VARS = []

TEMPORAL_REFERENCE = []