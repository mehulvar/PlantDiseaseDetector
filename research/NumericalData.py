import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('../data/plant_health_data.csv')

# Display basic information about the dataset
print("Shape of the dataset:", df.shape)
display(df.head())
print("\nDataset Information:")
print(df.info())
print("\nStatistical Summary:")
display(df.describe().T)

# Display the number of unique values in each column
print("\nUnique Values in Each Column:")
print(df.nunique())

# Separate numerical and categorical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
non_numerical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Display the lists of numerical and categorical columns
print("\nNumerical Columns:", numerical_columns)
print("Categorical Columns:", non_numerical_columns)

# Display unique values for each categorical column
for col in non_numerical_columns:
    print(f"\nColumn: {col}")
    print(f"Unique Values: {df[col].unique()}")

column_name = 'Plant_Health_Status'
plt.figure(figsize=(10, 4))

# First subplot: Count plot
plt.subplot(1, 2, 1)
sns.countplot(y=column_name, data=df, palette='muted')  
plt.title(f'Distribution of {column_name}')

ax = plt.gca()
for p in ax.patches:
    ax.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2), 
                ha='center', va='center', xytext=(10, 0), textcoords='offset points')

sns.despine(left=True, bottom=True)

# Second subplot: Pie chart
plt.subplot(1, 2, 2)
df[column_name].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('muted'), startangle=90, explode=[0.05]*df[column_name].nunique())
plt.title(f'Percentage Distribution of {column_name}')
plt.ylabel('')  

plt.tight_layout()
plt.show()


# Find the earliest and latest timestamps
start_date = pd.to_datetime(df['Timestamp']).min()
end_date = pd.to_datetime(df['Timestamp']).max()

print("Start Date:", start_date)
print("End Date:", end_date)

# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Check time differences between consecutive entries
time_diffs = df['Timestamp'].diff().value_counts()

print("Most common time differences:")
print(time_diffs.head(10))

# Function to perform univariate analysis for numeric columns
def univariate_analysis(data, columns):
    plt.figure(figsize=(16, 24))  # Adjusted to accommodate more plots
    
    muted_colors = sns.color_palette("muted", len(columns))
    
    for i, column in enumerate(columns):
        plt.subplot(4, 3, i + 1)  
        sns.histplot(data[column], kde=True, bins=10, color=muted_colors[i])
        plt.title(f'{column.replace("_", " ")} Distribution with KDE')
        plt.xlabel(column.replace('_', ' '))
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

columns_to_analyze = [
    'Plant_ID', 'Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature', 
    'Humidity', 'Light_Intensity', 'Soil_pH', 'Nitrogen_Level', 
    'Phosphorus_Level', 'Potassium_Level', 'Chlorophyll_Content', 
    'Electrochemical_Signal'
]

# Perform univariate analysis
univariate_analysis(df, columns_to_analyze)

# Select only numerical columns
numerical_df = df.select_dtypes(include=[np.number])

# Calculate skewness and kurtosis
skewness = numerical_df.skew()
kurtosis = numerical_df.kurt()

display("Skewness:", skewness)
print("\n")
display("Kurtosis:", kurtosis)

# Function to perform univariate analysis for numeric columns with boxplots and statistics
def univariate_analysis(data, column, title):
    plt.figure(figsize=(10, 2))
    
    color = sns.color_palette("muted")[columns_to_analyze.index(column) % len(sns.color_palette("muted"))]
    
    sns.boxplot(x=data[column], color=color)
    plt.title(f'{title} Boxplot')
    
    plt.tight_layout()
    plt.show()

    print(f'\nSummary Statistics for {title}:\n', data[column].describe())

columns_to_analyze = [
    'Plant_ID', 'Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature', 
    'Humidity', 'Light_Intensity', 'Soil_pH', 'Nitrogen_Level', 
    'Phosphorus_Level', 'Potassium_Level', 'Chlorophyll_Content', 
    'Electrochemical_Signal'
]

# Iterate through columns and perform univariate analysis
for column in columns_to_analyze:
    univariate_analysis(df, column, column.replace('_', ' '))

# Convert the Timestamp column to datetime format if not already
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Create a 'Week' column to group by week
df['Week'] = df['Timestamp'].dt.to_period('W').apply(lambda r: r.start_time)

# Aggregate Plant Health Status weekly for each Plant ID
weekly_health_status = (
    df.groupby(['Plant_ID', 'Week', 'Plant_Health_Status'])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

weekly_health_status.columns.name = None
weekly_health_status = weekly_health_status.rename(columns={
    'High Stress': 'High_Stress_Count',
    'Moderate Stress': 'Moderate_Stress_Count',
    'Healthy': 'Healthy_Count'
})

weekly_health_status

# Compare Plant_Health_Status with Soil Properties
# Define soil properties
soil_properties = [
    'Soil_Moisture', 'Soil_Temperature', 'Soil_pH', 
    'Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level'
]

# Create subplots to visualize the relationship between Plant_Health_Status and soil properties
plt.figure(figsize=(16, 20))
for i, feature in enumerate(soil_properties):
    plt.subplot(3, 2, i + 1)
    sns.boxplot(x='Plant_Health_Status', y=feature, data=df, palette='muted')
    plt.title(f'{feature} vs Plant Health Status')
    plt.xlabel('Plant Health Status')
    plt.ylabel(feature.replace('_', ' '))
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Compare Plant_Health_Status with Environmental Conditions
# Define environmental condition features
environmental_conditions = [
    'Ambient_Temperature', 'Humidity', 'Light_Intensity'
]

# Create subplots to visualize the relationship between Plant_Health_Status and environmental conditions
plt.figure(figsize=(16, 12))
for i, feature in enumerate(environmental_conditions):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='Plant_Health_Status', y=feature, data=df, palette='muted')
    plt.title(f'{feature} vs Plant Health Status')
    plt.xlabel('Plant Health Status')
    plt.ylabel(feature.replace('_', ' '))
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Compare Plant_Health_Status with Plant Health Indicators

# Define plant health indicator features
health_indicators = [
    'Chlorophyll_Content', 'Electrochemical_Signal'
]

# Create subplots to visualize the relationship between Plant_Health_Status and plant health indicators
plt.figure(figsize=(16, 8))
for i, feature in enumerate(health_indicators):
    plt.subplot(1, 2, i + 1)
    sns.boxplot(x='Plant_Health_Status', y=feature, data=df, palette='muted')
    plt.title(f'{feature} vs Plant Health Status')
    plt.xlabel('Plant Health Status')
    plt.ylabel(feature.replace('_', ' '))
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Compare Plant_Health_Status with Plant Health Indicators vs Soil Properties using scatter plots

# Define soil properties and plant health indicators
soil_properties = [
    'Soil_Moisture', 'Soil_Temperature', 'Soil_pH', 
    'Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level'
]

health_indicators = [
    'Chlorophyll_Content', 'Electrochemical_Signal'
]

# Create scatter plots for each combination of plant health indicators and soil properties
plt.figure(figsize=(30, 15))
plot_index = 1

for health_indicator in health_indicators:
    for soil_property in soil_properties:
        plt.subplot(len(health_indicators), len(soil_properties), plot_index)
        sns.scatterplot(
            x=soil_property, 
            y=health_indicator, 
            hue='Plant_Health_Status', 
            data=df, 
            palette='muted', 
            alpha=0.7
        )
        plt.title(f'{health_indicator} vs {soil_property}')
        plt.xlabel(soil_property.replace('_', ' '))
        plt.ylabel(health_indicator.replace('_', ' '))
        plt.legend(title='Health Status', loc='upper right')
        plot_index += 1

plt.tight_layout()
plt.show()

# Compare Plant_Health_Status with Plant Health Indicators vs Environmental Conditions using scatter plots

# Define environmental conditions and plant health indicators
environmental_conditions = [
    'Ambient_Temperature', 'Humidity', 'Light_Intensity'
]

health_indicators = [
    'Chlorophyll_Content', 'Electrochemical_Signal'
]

# Create scatter plots for each combination of plant health indicators and environmental conditions
plt.figure(figsize=(20, 12))
plot_index = 1

for health_indicator in health_indicators:
    for env_condition in environmental_conditions:
        plt.subplot(len(health_indicators), len(environmental_conditions), plot_index)
        sns.scatterplot(
            x=env_condition, 
            y=health_indicator, 
            hue='Plant_Health_Status', 
            data=df, 
            palette='muted', 
            alpha=0.7
        )
        plt.title(f'{health_indicator} vs {env_condition}')
        plt.xlabel(env_condition.replace('_', ' '))
        plt.ylabel(health_indicator.replace('_', ' '))
        plt.legend(title='Health Status', loc='upper right')
        plot_index += 1

plt.tight_layout()
plt.show()

# Explore relationships between Plant_Health_Status, Plant Health Indicators, and Environmental Conditions

# Group data by Plant_Health_Status and calculate mean values
status_analysis_mean = df.groupby('Plant_Health_Status')[
    ['Chlorophyll_Content', 'Electrochemical_Signal', 'Ambient_Temperature', 'Humidity', 'Light_Intensity']
].mean()

# Group data by Plant_Health_Status and calculate standard deviation
status_analysis_std = df.groupby('Plant_Health_Status')[
    ['Chlorophyll_Content', 'Electrochemical_Signal', 'Ambient_Temperature', 'Humidity', 'Light_Intensity']
].std()

# Displaying the mean values table
print("----- Mean Values by Plant Health Status -----")
display(status_analysis_mean)

# Displaying the standard deviation table
print("\n----- Standard Deviation by Plant Health Status -----")
display(status_analysis_std)


# Explore relationships between Plant_Health_Status and Soil Properties

# Group data by Plant_Health_Status and calculate mean values for soil properties
soil_properties = [
    'Soil_Moisture', 'Soil_Temperature', 'Soil_pH', 
    'Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level'
]

soil_mean = df.groupby('Plant_Health_Status')[soil_properties].mean()
soil_std = df.groupby('Plant_Health_Status')[soil_properties].std()

# Displaying the mean values table
print("----- Mean Soil Properties by Plant Health Status -----")
display(soil_mean)

# Displaying the standard deviation table
print("\n----- Standard Deviation of Soil Properties by Plant Health Status -----")
display(soil_std)

