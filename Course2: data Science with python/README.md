# Course2: Data Science with Python

Welcome to Course2 folder.  This collection of Jupyter notebooks is designed to guide you step-by-step through the foundational concepts of Pandas, Numpy, Statistics, Data Visualization, Data Wrangling, and Probability Distribution used in Ai/Ml.  — from NumPy functions, slicing, string functions and indexing. After reviewing the ipynb files, there is a `sales_analysis.py` under the Course2 projects folder to try and follow. The README.md in the Course2 projects folder describes the project requirements, use the dataset (`AusApparalSales4thQrt2020.csv`) to follow along!

Then we will jump to the Pandas folder

---

##  1. NumPy

- `Numpy.ipynb`  
- used for working with arrays
- working with multidimensional array of objects 
- manipulating functions
- conducting mathematical and logical operations on arrays   

- `Numpy_functions.ipynb`  
-  ndim: It is the number of axes (dimensions) of the array. 
-  shape: It provides the size of the array for each dimension. The output data type is a tuple. 
-  size : It is the total number of elements in the array. 
-  dtype: It shows the data type of the elements in the array. 
-  itemsize: It shows the length of one array element in bytes. 
-  data: It is an attribute offering direct access to the raw memory of a NumPy array.
-  reshape: It returns an array containing the same data with a new shape.
-  flatten: It returns a copy of the array flattened into a 1D array.
-  transpose: It swaps the rows and columns of a 2D array.

- `Numpy_Arithmetic_and_StringFunctions.ipynb`  
- Addition
- Subtraction
- Multiplication
- Division
- Power of
- Calculating Median, Mean, Standard Deviation, and Variance in the Array
- Calculating Percentiles
- numpy.char is a module that provides vectorized string operations for NumPy arrays.
- allows for applying string functions element-wise on entire NumPy arrays of strings

- `Numpy_indexing.ipynb`  
- Access 1D NumPy Array Elements
- Access 2D NumPy Array Elements
- Access 3D NumPy Array Elements
- Negative indices count backward from the end of the array.
- In a negative indexing system, the last element will be the first element with an index of- 1, the second last element with an index of- 2, and so on.

- `Numpy_slicing.ipynb`  
- In Python, slicing refers to moving elements from one index to another.
- Instead of using an index, the slice is passed as [start:end].
- Another way to pass the slice is to add a step as [start:end:step].
- In slicing, if the starting is not passed, it is considered as 0. If the step is not passed as 1 and if the end is not passed, it is considered as the length of the array in that dimension.

- `Numpy_practice.ipynb`  
- Create a one-dimensional NumPy array containing at least ten elements
- Create a 2D NumPy array with a minimum of 3 rows and 4 columns
- Create a 3D NumPy array with at least 2 matrices, each containing 2 rows and 3 columns
- Access elements in NumPy arrays and utilize indexing and slicing techniques for efficient data retrieval
- Access and print various elements from 1D, 2D, and 3D arrays using positive indexing


## 2. Pandas

- `pandas_intro.ipynb`
- introduces the fundamentals, and features of Pandas
- we learn Data Structures
- Introduces to Serieas and how to perform operations and Transformations on them

- `pandas_DataFrame.ipynb`
- 3 csv files are provided
- introduces how to create a dataframe
- how to call an existing dataframe using the built in read_csv function using the provided csv files
- introduces stats, means, median, and standard deviation using pandas
- introduces the correlation matrix analysis

- `pandas_date_time.ipynb` 
- Date and TimeDelta in Pandas
- Date Handling in Pandas
- Extracting Components from Dates
- Timedelta in Pandas
- Creating a Timedelta
- Performing Arithmetic Operations
- Resampling Time Series Data
- Categorical Data Handling
- Creating a Categorical Variable
- Counting Occurrences of Each Category
- Creating Dummy Variables
- Label Encoding

`pandas_strings.ipynb`
- Text Data in Pandas
- Iteration
- Iterating over Rows
- Applying a Function to Each Element
- Vectorized Operations
- Iterating over Series
- Sorting
- Sorting DataFrame by Column
- Sorting DataFrame by Multiple Columns
- Sorting DataFrame by Index
- Sorting a Series
- Plotting with Pandas

`pandas_practice.ipynb`
- practice using a provided sample data
1. Create a Pandas Series for sales data
- Use a list of daily sales figures to create a Pandas Series
- Assign days of the week as the index
2. Access and manipulate sales data
- Access sales data for specific days using index labels
- Calculate total sales for the week
   
## 3. Data Visualization

`data_visualization_intro.ipynb`
- Data visualization is the graphical representation of data to reveal patterns, trends, and insights that might not be easily apparent from raw data.
- It involves creating visual elements such as charts, graphs, and maps to communicate complex information in an understandable and interpretable form.
- Data visualization tools and libraries, such as Matplotlib, Seaborn, and Plotly allows analysts, scientists and professionals to create compelling visualizations that make understanding of data and support evidence-based decisions-making. 

`seaborn_plotly_visualization.ipynb`
- Seaborn is a Python library for statistical data visualization that builds on Matplotlib.
- It provides an interface for creating attractive and informative statistical graphics.
- It comes with several built-in themes and color palettes to make creating aesthetically pleasing visualizations easy.
- It is particularly well-suited for exploring complex datasets with multiple variables.

## 4. Math and Statistics

`linear_algebra.ipynb`
- Introduction to Linear Algebra
- Scalars and Vectors
- Vector Operation: Multiplication
- Norm of a Vector
- Matrix and Matrix Operations
- Rank of Matrix
- Determinant of Matrix and Identity Matrix
- Inverse of Matrix, Eigenvalues, and Eigenvectors
- Eigenvalues and Eigenvectors
- Calculus in Linear Algebra

## 5. Statistic Fundamentals
`statistic_fundamentals.ipynb`
- Importance of Statistics for Data Science:
- What Is Statistics?
- Common Terms Used in Statistics
- Statistics Types
- Types of Data
- Measures of Central Tendency
- Measures of Dispersion
- Range
- Interquartile Range
- Standard Deviation
- Variance
- Measures of Shape
- Skewness
- Kurtosis
- Covariance and Correlation

## 6.Probability Distribution
`probability_distribution.ipynb`
- Probability and Its Importance
- Random Variable
- Probability Distribution
- Discrete Probability Distribution
- Continuous Probability Distribution   
- Discrete Probability Distribution
- Bernoulli Distribution
- Binomial Distribution
- Poisson Distribution
- Continuous Probability Distribution
- Normal Distribution
- Uniform Distribution

## 7. Advanced Statistics
`advanced_statistics.ipynb`
- Hypothesis Testing and Mechanism
- Introduction to Hypothesis
- Hypothesis Components
- Null and Alternative Hypothesis
- Hypothesis Testing
- Hypothesis Testing Outcomes: Type I and Type II Errors
- Steps Involved in Hypothesis Testing
- Confidence Interval
- Margin of Error
- Confidence Levels
- Z-Distribution (Standard Normal Distribution)
- T-Distribution
- Plotting T and Z Distribution
- T-Test
- Z-Test
- Choosing between T-Test and Z-Test
- P-Value
- Decision-Making Using P-Value
- Chi-Square Distribution
- Chi-Square Test and Independence Test

## 8. Data Wrangling
`data_wrangling.ipynb`
- Introduction to Data Wrangling
- Data Collection
- Data Inspection
- Accessing Rows Using .iloc and .loc
- Checking for Missing Values
- Handling Missing Data
- Dealing with Duplicates
- Data Cleaning
- Data Transformation
- Data Binning
- Handling Outliers
- Pandas Joining Techniques
- Pandas Concatenate
- Pandas Merge Dataframes
- Pandas Join Dataframes
- Aggregating Data
- Reshaping Data

## Feature Engineering
`feature_engineering.ipynb`
- Introduction to Feature Engineering
- Feature Engineering Methods
- Transforming Variables
- Log Transformation
- Square Root Transformation
- Box-Cox Transformation
- Features Scaling
- Label Encoding
- One Hot Encoding
- Hashing
- Hashlib Module
- Grouping Operations

## Project: Sales Analysis Dashboard

`sales_analysis.py`
#  Sales Analysis Dashboard
This project presents a detailed group-wise sales analysis for customer segments. 

The objective is to provide actionable insights to the Sales & Marketing (S&M) team through intuitive visualizations, enabling better strategic planning, hyper-personalization, and the identification of the Next Best Offers (NBO).

---

## Key Features

### Group-Wise Sales Across States
- Comparative sales performance analysis across **different states** for each demographic group.
- Provides regional insights into target audiences.

### Time-of-Day Sales Patterns
- Identifies **peak and off-peak hours** based on transaction timestamps.
- Helps S&M teams optimize campaign timing and staffing.

###  Time-Based Performance Charts
Includes interactive and static charts at the following granularity levels:
- **Daily**
- **Weekly**
- **Monthly**
- **Quarterly**

These views empower leadership to track performance trends and seasonality.

---

## Visualization Tools Used

### Library: **Seaborn**
**Reason for Recommendation:**
Seaborn is a statistical data visualization library built on top of Matplotlib. It simplifies the creation of complex visualizations and integrates well with Pandas DataFrames, making it an ideal tool for:
- **Group-wise comparisons**
- **Trend identification**
- **Heatmaps and categorical plots**

Seaborn’s expressive syntax and built-in support for aggregation, bootstrapping, and color palettes make it suitable for executive dashboards intended for non-technical stakeholders.

### Optional: `matplotlib` and `plotly`
- `matplotlib` was used as a base for layout customization.
- `plotly` can optionally be used for interactive charts where real-time filtering or drill-downs are needed.

---

## Example Visualizations

- **Barplots:** Group-wise sales across states
- **Lineplots:** Monthly trends per group
- **Boxplots:** Time-of-day distribution by segment
- **Heatmaps:** Weekly performance per state/group

---

## Recommendations

1. **Adopt time-aware marketing strategies:** Focus outreach during identified peak hours for each group.
2. **Prioritize personalization programs:** Use demographic-specific sales data to develop Next Best Offer recommendations.
3. **Utilize state-level insights:** Tailor regional campaigns to high-performing or underperforming states per demographic group.
4. **Automate Reporting:** Implement scheduled runs of the dashboard with updated charts to support daily stand-ups and quarterly reviews.

---




