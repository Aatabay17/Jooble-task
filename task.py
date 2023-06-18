import pandas as pd

# Load the training data
test_data = pd.read_csv('test.csv')

# Select the columns to be standardized
columns_to_standardize = ['feature_type_1_1', 'feature_type_1_2', 'feature_type_1_3', 'feature_type_1_4', 'feature_type_1_5', 'feature_type_1_6', 'feature_type_1_7', 'feature_type_1_8', 'feature_type_1_9']

# For adding any additional columns(factors)
# columns_to_standardize += ['new_feature_type_column',...]

# Calculate the mean and standard deviation for the selected columns
mean_values = test_data[columns_to_standardize].mean()
std_values = test_data[columns_to_standardize].std()

# Perform z-score normalization on the selected columns
for column in columns_to_standardize:
    new_column = f'features_type_1_stand_{column.split("_")[-1]}'
    test_data[new_column] = (test_data[column] - mean_values[column]) / std_values[column]


test_data['max_feature_type_1_index'] = test_data[columns_to_standardize].idxmax(axis=1).str.split('_').str[-1].astype(int)

columns_to_write = ['id_job'] + [f'features_type_1_stand_{i}' for i in range(1, 10)] + ['max_feature_type_1_index']

# Write the selected columns to a CSV file
test_data[columns_to_write].to_csv('test_transformed.csv', index=False)

