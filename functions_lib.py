'''
UDF-library

-------------------
Categories:
1. Visualisation
2. Statistic
3. Processing
-------------------

Table of contents:

Visualisation:
plot_aggregation_histogram
two_categorical_one_numeric
two_numeric_one_categorical
numeric_distribution
plot_scatterplots_by_group
two_numeric_relationship
categorical_numeric_relationship
categorical_categorical_relationship
categorical_distribution
plot_correlation_heatmap

Statistic:
check_target_overlap
top_n_popular_values

Processing:
binary_encode_columns
categorize_columns
convert_ordinal_to_category
compare_variable_types
'''


#############################################################################
# Import Libraries
#############################################################################

# Data processing
import pandas as pd
import numpy as np
import pandasql as ps

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


#############################################################################
# Visualisation
#############################################################################

def plot_aggregation_histogram(df, x, y, agg_funcs=['mean']):
    """
    Plot histograms and KDEs for aggregated data grouped by a categorical variable.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the data to be analyzed.
    x : str
        The name of the column in df that contains the categorical variable for grouping.
    y : str
        The name of the column in df that contains the numeric variable to be aggregated.
    agg_funcs : list of str, optional
        A list of aggregation functions to apply to the numeric variable. 
        Options include: 'mean', 'std', 'median', 'max', 'min', 'count', 'nunique', 'sum'.
        Default is ['mean'].

    Returns
    -------
    None
        Displays a plot showing the distribution of aggregated data with histograms and KDEs.
    """
    # Define a dictionary for aggregation functions
    agg_funcs_dict = {
        'mean': 'mean',
        'std': 'std',
        'median': 'median',
        'max': 'max',
        'min': 'min',
        'count': 'count',
        'nunique': 'nunique',
        'sum': 'sum' 
    }
    
    # Check if the provided aggregation functions are valid
    for func in agg_funcs:
        if func not in agg_funcs_dict:
            raise ValueError(f"Invalid aggregation function: {func}. Choose from: {list(agg_funcs_dict.keys())}")

    # Generate colors for the plots
    colors = sns.color_palette("husl", len(agg_funcs))
    
    # Set up the figure size and grid layout
    plt.figure(figsize=(8, 5))
    
    for i, agg_func in enumerate(agg_funcs):
        # Group by the categorical variable and apply the selected aggregation function
        aggregated_data = df.groupby(x)[y].agg(agg_funcs_dict[agg_func]).reset_index()
        
        # Plot the histogram with normalization (density=True)
        sns.histplot(
            aggregated_data[y], 
            bins=10, 
            color=colors[i], 
            alpha=0.5, 
            kde=False, 
            stat='density',  # Normalize the histogram
            label=f'{agg_func.capitalize()} Histogram'
        )
        
        # Plot the KDE curve
        sns.kdeplot(
            aggregated_data[y], 
            color=colors[i], 
            label=f'{agg_func.capitalize()} KDE', 
            linewidth=2
        )

    # Customize axes and title
    plt.xlabel(y, fontsize=12)  # Label for the x-axis
    plt.ylabel('Density', fontsize=12)  # Label for the y-axis
    plt.title(f'Histograms and KDEs of {y} by {x}', fontsize=14)  # Title of the plot
    plt.legend(fontsize=10)  # Add a legend with appropriate font size
    plt.grid(True, linestyle='--', alpha=0.6)  # Add a grid with dashed lines
    
    # Show the plot
    plt.tight_layout()  # Adjust layout to prevent overlapping elements
    plt.show()
    
    

def two_categorical_one_numeric(data, x, y, hue):
    """
    Create grouped box plots with two categorical variables and one numeric variable.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the data with at least three columns: x, y, and hue.
    x : str
        The name of the column in data that contains the x-axis variable.
    y : str
        The name of the column in data that contains the y-axis variable.
    hue : str
        The name of the column in data that determines the grouping within the box plots.

    Returns
    -------
    None
        Displays the grouped box plot.
    """
    
    # Set the style of the graph
    sns.set(style="whitegrid")
    
    # Create a grouped boxplot with a more pleasant color palette
    ax = sns.boxplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        palette="pastel",  # Nice color palette
        fliersize=4,       # Size of outliers
        linewidth=1.2      # Width of box lines
    )
    
    # Add gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Set the title and axis labels
    plt.title(f'Box plot of {y} grouped by {x} and colored by {hue}', fontsize=14)
    plt.xlabel(x, fontsize=12)
    plt.ylabel(y, fontsize=12)
    
    # Remove the top and right borders
    sns.despine(left=True, bottom=False)
    
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


def two_numeric_one_categorical(data, x, y, hue):
    """
    Create a scatter plot with two numeric variables and one categorical variable for visual cues.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the data with at least three columns: x, y, and hue.
    x : str
        The name of the column in data that contains the x-axis variable.
    y : str
        The name of the column in data that contains the y-axis variable.
    hue : str
        The name of the column in data that determines the color or shape of the points.

    Returns
    -------
    None
        Displays the scatter plot.
    """
    
    # Set the style of the graph
    sns.set(style="whitegrid")
    
    # Create a scatter plot with a more pleasant color palette
    ax = sns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        palette="Set2",  # Nice color palette
        s=50,           # Marker size
        edgecolor="w",  # White outline for markers
        linewidth=0.8   # Line width for markers
    )
    
    # Add gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Set the title and axis labels
    plt.title(f'Scatter plot of {y} vs {x} colored by {hue}', fontsize=14)
    plt.xlabel(x, fontsize=12)
    plt.ylabel(y, fontsize=12)
    
    # Remove the top and right borders
    sns.despine(left=True, bottom=False)
    
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


def numeric_distribution(series, variable_type='continuous'):
    """
    Plot the distribution for a numeric series with a histogram in the background.

    Parameters
    ----------
    series : pd.Series
        A numeric Series for which to plot the distribution.
    variable_type : str, optional
        The type of variable: 'continuous' for continuous variables and 'discrete' for discrete variables.
        Default is 'continuous'.

    Returns
    -------
    None
        Displays a plot of the distribution of the numeric series.
    """
    # Calculate basic statistics
    mean_value = series.mean()
    median_value = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    # Calculate outlier thresholds
    outlier_threshold_upper = q3 + 1.5 * iqr
    outlier_threshold_lower = q1 - 1.5 * iqr
    has_upper_outliers = (series > outlier_threshold_upper).any()
    has_lower_outliers = (series < outlier_threshold_lower).any()
    
    # Determine the number of bins based on variable type
    if variable_type == 'discrete':
        bins = len(series.unique())  # For discrete data, set bins to the number of unique values
    else:
        bins = 'auto'  # For continuous data, use default 'auto' bins
    
    # Create the plot
    plt.figure(figsize=(6, 4))
    plt.hist(
        series, 
        bins=bins, 
        color='lightgray', 
        alpha=0.8, 
        density=True, 
        label='Histogram'
    )
    
    # Add vertical lines for mean, median, and quartiles
    plt.axvline(mean_value, color='green', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(median_value, color='red', linestyle='dotted', linewidth=2, label='Median')
    plt.axvline(q1, color='blue', linestyle='--', linewidth=1, label='Q1')
    plt.axvline(q3, color='blue', linestyle='--', linewidth=1, label='Q3')
    
    # Add vertical lines for outlier thresholds if they exist
    if has_upper_outliers:
        plt.axvline(outlier_threshold_upper, color='orange', linestyle='--', linewidth=1, label='Upper Outlier')
    if has_lower_outliers:
        plt.axvline(outlier_threshold_lower, color='orange', linestyle='--', linewidth=1, label='Lower Outlier')
    
    # Customize the plot
    plt.title(f'Distribution of {series.name}', fontsize=14)
    plt.xlabel(series.name, fontsize=12)
    plt.ylabel('Density' if variable_type == 'continuous' else 'Frequency', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(axis='y')
    
    # Show the plot
    plt.show()


def plot_scatterplots_by_group(df, x, y, groupby, plots_per_row=3):
    """
    Create scatter plots for a continuous variable against another continuous variable, 
    grouped by a categorical variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the data with one categorical variable and two continuous variables.
    x : str
        The name of the column in df that contains the first continuous variable (x-axis).
    y : str
        The name of the column in df that contains the second continuous variable (y-axis).
    groupby : str
        The name of the column in df that contains the categorical variable used for grouping.
    plots_per_row : int, optional
        The number of scatter plots to display in each row. Default is 3.
    
    Returns
    -------
    None
        Displays scatter plots for each group defined by the categorical variable.
    """
    
    # Set up the plot dimensions based on the number of unique groups
    plt.figure(figsize=(15, 10))
    num_vals = df[groupby].nunique()  # Number of unique groups
    num_rows = (num_vals + plots_per_row - 1) // plots_per_row  # Calculate required rows

    # Iterate over each unique group and create a scatter plot
    for i, group_value in enumerate(df[groupby].unique()):
        plt.subplot(num_rows, plots_per_row, i + 1)  # Create subplot

        # Filter data for the current group
        group_data = df[df[groupby] == group_value]

        # Create scatter plot for the current group
        sns.scatterplot(data=group_data, x=x, y=y)

        # Add trend line without confidence interval shading
        sns.regplot(data=group_data, x=x, y=y, scatter=False, color='red', ci=None)

        # Set title and labels for the plot
        plt.title(f'Relation between {x} and {y} for {group_value}')
        plt.xlabel(x)
        plt.ylabel(y)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Display the plots
    plt.show()


def two_numeric_relationship(df, x, y, agg_func='mean'):
    """
    Plot the relationship between two continuous variables using a scatter plot.
    
    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the data with two continuous variables.
    x : str
        The name of the column in df that contains the first continuous variable.
    y : str
        The name of the column in df that contains the second continuous variable.
    agg_func : str or None
        The aggregation function to use ('mean', 'median', 'std', 'max', 'min', 'count', 'nunique') or None.
    
    Returns
    -------
    None
        Displays a plot showing the relationship between the two continuous variables.
    """
    
    # Check if the two columns are the same
    if x == y:
        print(f"Skipping plot for {x} and {y} because they are the same.")
        return
    
    # Define a dictionary for aggregation functions
    agg_funcs = {
        'mean': 'mean',
        'std': 'std',
        'median': 'median',
        'max': 'max',
        'min': 'min',
        'count': 'count',
        'nunique': 'nunique'
    }
    
    # Check if the provided aggregation function is valid
    if agg_func is not None and agg_func not in agg_funcs:
        raise ValueError(f"Invalid aggregation function: {agg_func}. Choose from {list(agg_funcs.keys()) + [None]}.")
    
    # If no aggregation is specified, simply plot using the raw data
    if agg_func is None:
        plt.scatter(df[x], df[y], label='Raw Data', color='skyblue', alpha=0.7)
        
        # Add a trend line using linear regression
        z = np.polyfit(df[x], df[y], 1)  # Linear regression
        p = np.poly1d(z)
        plt.plot(df[x], p(df[x]), linestyle='--', color='orange', label='Trend Line')
        
        plt.title(f'Relationship between {x} and {y}', fontsize=12)
    
    else:
        # Group by the first continuous variable and apply the aggregation function to the second
        try:
            aggregated_data = df.groupby(x)[y].agg(agg_funcs[agg_func]).reset_index()
            
            # Plot scatter of aggregated data
            plt.scatter(aggregated_data[x], aggregated_data[y],
                        label=f'{agg_func.capitalize()} of {y}', color='skyblue', alpha=0.7)
            
            # Add a trend line using linear regression
            z = np.polyfit(aggregated_data[x], aggregated_data[y], 1)  # Linear regression
            p = np.poly1d(z)
            plt.plot(aggregated_data[x], p(aggregated_data[x]), linestyle='--',
                     color='orange', label='Trend Line')
            
            plt.title(f'{agg_func.capitalize()} of {y} by {x}', fontsize=12)
        
        except ValueError as e:
            print(f"Error while processing {x} and {y}: {e}")
            return

    plt.xlabel(x, fontsize=10)
    plt.ylabel(y, fontsize=10)
    plt.legend(fontsize=10)
    plt.grid()
    
    # Show the plot
    plt.show()



def categorical_numeric_relationship(data, x, y):
    """
    Plot the distribution of a continuous variable grouped by a categorical variable using a box plot.
    
    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the data with one categorical variable and one continuous variable.
    x : str
        The name of the column in data that contains the categorical variable.
    y : str
        The name of the column in data that contains the continuous variable for which to plot the distributions.
    
    Returns
    -------
    None
        Displays a box plot comparing the distributions of the continuous variable for each category.
    """
    
    # Set the style of the graph
    sns.set(style="whitegrid")
    
    # Create a boxplot with a more pleasant color palette and disable the legend
    ax = sns.boxplot(
        data=data,
        x=x,
        y=y,
        palette="Set3",
        fliersize=4,
        linewidth=1.2
    )
    
    # Add gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Set the title and axis labels in English
    plt.title(f'Comparison of distributions of {y} by {x}', fontsize=14)
    plt.xlabel(x, fontsize=12)
    plt.ylabel(y, fontsize=12)
    
    # Remove the top and right borders
    sns.despine(left=True, bottom=False)
    
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


def categorical_categorical_relationship(df, categorical1, categorical2, max_bars_num=10):
    """
    Plot the proportions of a categorical variable grouped by another categorical variable.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the data with two columns: one for the categorical variable (categorical1)
        and one for the other categorical variable (categorical2).
    categorical1 : str
        The name of the column in df that contains the first categorical variable.
    categorical2 : str
        The name of the column in df that contains the second categorical variable for which to plot the distributions.
    max_bars_num : int, optional
        The maximum number of unique values to display (default is 10).

    Returns
    -------
    None
        Displays a bar plot comparing the proportions of the second categorical variable for each category of the first.
    """
    
    # Counting unique values for the second categorical variable
    value_counts = df[categorical2].value_counts()

    # Limiting to max_bars_num values    
    if len(value_counts) > max_bars_num:
        top_values = value_counts.head(max_bars_num).index
        df = df[df[categorical2].isin(top_values)]

    # Counting frequency for each category by first categorical variable
    count_data = df.groupby([categorical2, categorical1]).size().unstack(fill_value=0)

    # Counting the total number of rows for each class of the first categorical variable
    total_counts = df[categorical1].value_counts()

    # Calculating proportions (density) considering class imbalance
    density_data = count_data.div(total_counts, axis=1) * 100

    # Sorting by descending proportion for the first category in categorical1
    density_data = density_data.sort_values(by=density_data.columns[0], ascending=False)

    # Creating the plot
    ax = density_data.plot(kind='bar', colormap='Set3')  # Use a nice color palette

    # Setting up the plot
    plt.title(f'Comparison of distributions of {categorical2} for each category of {categorical1} (in %)', fontsize=14)
    plt.xlabel(categorical2, fontsize=12)
    plt.ylabel('Density (%)', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend(title=categorical1, fontsize=10)
    plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    # Remove the top and right borders
    sns.despine(left=True, bottom=False)

    # Show the plot
    plt.tight_layout()
    plt.show()
        
    
def categorical_distribution(
    series: pd.Series, 
    max_bars_num: int = 10, 
    bars_order: str = None
):
    """
    Plot the proportions of a categorical variable in a Series.
    
    Parameters
    ----------
    series : pd.Series
        A categorical Series for which to plot the distribution.
    max_bars_num : int, optional
        The maximum number of unique values to display (default is 10).
    bars_order : str, optional
        The order of bars: 
            - None: sort by category name alphabetically (ascending)
            - 'asc': sort by frequency ascending
            - 'desc': sort by frequency descending (default behavior if omitted)
    
    Returns
    -------
    None
        Displays a bar plot comparing the proportions of the categorical variable.
    """
    # Calculate value counts
    value_counts = series.value_counts()
    
    # Sort based on bars_order
    if bars_order == 'asc':
        sorted_counts = value_counts.sort_values(ascending=True)
    elif bars_order is None:
        sorted_counts = value_counts.sort_index()
    else:  # 'desc' or any other value defaults to frequency descending
        sorted_counts = value_counts
    
    # Limit to max_bars_num values
    if len(sorted_counts) > max_bars_num:
        top_values = sorted_counts.head(max_bars_num).index
        series = series[series.isin(top_values)]
    
    # Recalculate counts for filtered data
    filtered_counts = series.value_counts()
    
    # Calculate proportions (density)
    total_counts = value_counts.sum()  # Total counts for all categories
    density_data = (filtered_counts / total_counts) * 100
    
    # Sort density_data based on bars_order
    if bars_order == 'asc':
        sorted_density = density_data.sort_values()
    elif bars_order is None:
        sorted_density = density_data.sort_index()
    else:
        sorted_density = density_data.sort_values(ascending=False)
    
    # Create plot
    ax = sorted_density.plot(
        kind='bar', 
        color='#4CAF50', 
        width=0.8,
        edgecolor='black',
        alpha=0.8
    )
    
    # Formatting
    plt.title(f'Distribution of {series.name} (in %)', fontsize=14)
    plt.xlabel(series.name, fontsize=12)
    plt.ylabel('Density (%)', fontsize=12)
    plt.xticks(rotation=45, fontsize=10, ha='right')
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Add percentages on bars based on total counts
    for p in ax.patches:
        category_label = p.get_x() + p.get_width() / 2  # Получаем центр столбца
        category_name = sorted_density.index[int(category_label)]  # Получаем имя категории по индексу
        total_percentage = (value_counts[category_name] / total_counts) * 100  # Получаем реальный процент для всех значений
        
        ax.annotate(
            f'{total_percentage:.1f}%', 
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', 
            fontsize=10, 
            color='black'
        )
    
    # Final adjustments
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, corr_threshold = 0.2):
    """
    Visualize the correlation matrix of features in a DataFrame using a heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the features for which to calculate the correlation matrix.
    corr_threshold : float (default is 0.2).
        The threshold for displaying correlations in the heatmap. Only correlations 
        with an absolute value greater than this threshold will be highlighted.

    Returns
    -------
    None
    """
    
    # Calculate the correlation matrix
    corr_matrix = df.corr()
    
    # Create a mask for values exceeding the correlation threshold
    mask = np.abs(corr_matrix) > corr_threshold
    
    # Set up the figure size for the heatmap
    plt.figure(figsize=(5, 4))
    
    # Create the heatmap with the specified mask
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", mask=~mask,
                cmap='coolwarm', square=True, cbar_kws={"shrink": .8},
                linewidths=.5, linecolor='black')

    # Set the title of the heatmap
    plt.title(f'Correlation Matrix (Threshold: {corr_threshold})')
    
    # Display the heatmap
    plt.show()
    
    return None
           
                
#############################################################################
# Statistic
#############################################################################

def check_target_overlap(df, target, unique_sample_id):
    """
    Check if there are overlapping targets in unique samples.    
    This function groups the DataFrame by the unique sample ID and counts the distinct 
    target values for each sample. It then returns a list of unique sample IDs that 
    have more than one distinct target.    
    
    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing at least two columns: one for the target variable and 
        another for the unique sample identifier.        
        
    target : str
        The name of the target variable column (default is 'Conversion').
        
    unique_sample_id : str
        The name of the unique sample identifier column (default is 'CheckoutViewID').        
        
    Returns
    -------
    list
        A list of unique sample IDs that have more than one distinct target value.
    """    
    # Group by unique identifier and count the number of unique targets
    overlap_counts = df.groupby(unique_sample_id)[target].nunique()
    
    # Get the list of unique identifiers with more than one target
    overlapping_samples = overlap_counts[overlap_counts > 1].index.tolist()
    
    return overlapping_samples


def top_n_popular_values(df, n):
    """ 
    Calculate the top N popular values for all columns in a DataFrame,
    along with their counts and percentages of total rows.
    
    Parameters 
    ----------
    df : pd.DataFrame
        A DataFrame containing the data.
    n : int
        The number of top values to return for each column.
    """
    # Create lists to store results
    results = []
    
    # Iterate over all columns in the DataFrame
    for column in df.columns:
        # Get value counts for the column
        value_counts = df[column].value_counts()
        
        # Determine the number of rows in the DataFrame
        total_rows = len(df)
        
        # Get the top N values (or all if less than N)
        top_values = value_counts.head(n)
        
        # Iterate over the top values to collect counts and percentages
        for value, count in top_values.items():
            percentage = (count / total_rows) * 100
            results.append({
                'column_name': column,
                'value': value,
                'count': count,
                'percentage': round(percentage, 2)
            })
    
    # Create a new DataFrame with the results
    result_df = pd.DataFrame(results)
    
    # Sort the result DataFrame
    result_df = result_df.sort_values(by=['column_name', 'count'], ascending=[True, False])
    
    # Print the results for each column
    for col in df.columns:
        display(result_df[result_df['column_name'] == col])
        print("\n")  # Add a newline for better readability



#############################################################################
# Processing
#############################################################################



def binary_encode_columns(df, columns):
    """
    Converts binary categorical columns to boolean (True/False) in place and prints the results.
    
    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame.
    columns : list
        A list of column names to be transformed. Each column should contain only 2 unique values and not be numeric.
    """
    converted_columns = []
    not_converted_columns = []

    for column in columns:
        # Check if the column exists in the DataFrame
        if column in df.columns:
            # Get unique values of the column
            unique_values = df[column].dropna().unique()
            
            # Check that there are exactly 2 unique values and they are not numeric
            if len(unique_values) == 2 and all(isinstance(val, str) for val in unique_values):
                # Find the most frequently occurring value
                most_frequent_value = df[column].mode()[0]
                
                # Create a new column name based on the original column and most frequent value
                new_column_name = f"{column}_{most_frequent_value}"
                
                # Transform values to True and False
                df[new_column_name] = df[column].apply(lambda x: True if x == most_frequent_value else (False if pd.notnull(x) else None))
                
                # Store the name of the converted column
                converted_columns.append(f"{column} -> {new_column_name}")
                
                # Drop the old column
                df.drop(columns=[column], inplace=True)
            else:
                not_converted_columns.append(column)
        else:
            not_converted_columns.append(column)

    # Print the results
    print("Converted columns to boolean:", converted_columns)
    
    if not_converted_columns:
        print("Columns that were not converted:", not_converted_columns)


def categorize_columns(df):
    """Categorizes DataFrame columns based on their data types.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to categorize.    

    Returns
    -------
    specific_types : dict
        A dictionary with specific categories of variables and their corresponding column names.
    general_types : dict
        A dictionary with general categories of variables and their corresponding column names.
    """
    # Initialize the dictionaries
    specific_types = {
        'numeric_discrete': [],
        'numeric_continuous': [],
        'categorical_nominal': [],
        'categorical_ordinal': [],
        'categorical_binary': [],
        'temporal': [],
        'general': []
    }

    general_types = {
        'numeric': [],
        'categorical': [],
        'temporal': [],
        'general': []
    }

    # Iterate over the columns and categorize them based on their data types
    for column in df.columns:
        dtype = df[column].dtype
        
        # Check for numeric types
        if pd.api.types.is_integer_dtype(dtype):
            specific_types['numeric_discrete'].append(column)
            general_types['numeric'].append(column)
        elif pd.api.types.is_float_dtype(dtype):
            specific_types['numeric_continuous'].append(column)
            general_types['numeric'].append(column)
        
        # Check for categorical types
        elif pd.api.types.is_categorical_dtype(dtype):
            specific_types['categorical_ordinal'].append(column)
            general_types['categorical'].append(column)
        elif pd.api.types.is_bool_dtype(dtype):
            specific_types['categorical_binary'].append(column)
            general_types['categorical'].append(column)
        
        # Check for object types (not strings) # this condition should always be before string, since string returns True on string and object types
        elif pd.api.types.is_object_dtype(dtype):
            specific_types['general'].append(column)  # Treating object as general
            general_types['general'].append(column)  

        # Check for string types
        elif pd.api.types.is_string_dtype(dtype):
            specific_types['categorical_nominal'].append(column)
            general_types['categorical'].append(column)
        
        # Check for temporal types
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            specific_types['temporal'].append(column)
            general_types['temporal'].append(column)
        
        # If none of the above, categorize as general
        else:
            specific_types['general'].append(column)
            general_types['general'].append(column)

    return specific_types, general_types
        
        


def convert_ordinal_to_category(df, column_dict):
    """
    Converts specified columns in the DataFrame to categorical type with an ordered category.
    
    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame.
    column_dict : dict
        A dictionary where keys are column names and values are lists of categories in order.
    """
    converted_columns = []
    not_converted_columns = []
    
    for column, categories in column_dict.items():
        # Check if the column exists in the DataFrame
        if column in df.columns:
            # Check if all values in the categories list are unique and non-numeric
            if len(categories) == len(set(categories)):
                # Convert the column to categorical with specified order
                df[column] = pd.Categorical(df[column], categories=categories, ordered=True)
                converted_columns.append(column)
            else:
                not_converted_columns.append(column)
        else:
            not_converted_columns.append(column)

    # Print the results
    print("Converted columns to categorical:", converted_columns)
    if not_converted_columns:
        print("Columns that were not converted:", not_converted_columns)
        
        
def compare_variable_types(target_types, specific_types):
    """
    Compares the variable types between two sets of columns and identifies matches and mismatches.

    Parameters
    ----------
    target_types : dict
        A dictionary where keys are category names and values are lists of column names representing the target types.
    specific_types : dict
        A dictionary where keys are category names and values are lists of column names representing the specific types to compare against.
    """
    
    for category in target_types.keys(): 
        target_columns = set(target_types[category]) 
        specific_columns = set(specific_types[category]) 
 
        print(f"{category}:") 
 
        # Lists to store matching and non-matching columns 
        matches = [] 
        not_match = [] 
 
        # Check for column matches 
        for column in specific_columns: 
            if column in target_columns: 
                matches.append(column) 
            else: 
                # Determine the target category for output 
                target_category = next((cat for cat in target_types if column in target_types[cat]), None) 
                if target_category: 
                    not_match.append(f"{column} (to be: {target_category})") 
 
        # Print matching columns if there are any 
        if matches: 
            print("  match:") 
            for item in matches: 
                print(f"  - {item}") 
 
        # Print non-matching columns if there are any 
        if not_match: 
            print("  not match:") 
            for item in not_match: 
                print(f"  - {item}") 
 
        # Check for columns in specific_types that are not in target_types 
        not_in_target_types = specific_columns - set().union(*target_types.values()) 
        if not_in_target_types: 
            print("  not in target types:") 
            for item in not_in_target_types: 
                print(f"  - {item}") 
         
        print()  # Print a blank line to separate categories
        

