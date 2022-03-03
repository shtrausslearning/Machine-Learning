''' FEATURES WITH STRINGS '''
# Data that contain strings

''' LABEL ENCODING '''
import pandas as pd

# Create series with male and female values
non_categorical_series = pd.Series(['male', 'female', 'male', 'female'])
# Convert the text series to a categorical series
categorical_series = non_categorical_series.astype('category')
# Print the numeric codes for each value
print(categorical_series.cat.codes)
# Print the category names
print(categorical_series.cat.categories)

''' ONE HOT ENCODING '''
import pandas as pd

# Create series with male and female values
non_categorical_series = pd.Series(['male', 'female', 'male', 'female'])
# Create dummy or one-hot encoded variables
print(pd.get_dummies(non_categorical_series))
