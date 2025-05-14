# NumPy and Pandas Folder

Welcome to my NumPy folder.  This collection of Jupyter notebooks is designed to guide you step-by-step through the foundational concepts of NumPy used in Ai/Ml.  — from NumPy functions, slicing, string functions and indexing. After reviewing the ipynb files, there is a `Numpy-practice` file to tackle. 
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
   