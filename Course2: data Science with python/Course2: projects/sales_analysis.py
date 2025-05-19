import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


# first load the file
df = pd.read_csv('AusApparalSales4thQrt2020.csv')

# Show basic info and first 5 rows of the dataset
# df_info = df.info()
df_head = df.head()

# return a summary statistics for all columns in the DataFrame
df.describe(include='all')

# Look for missing values in the column dataset
missing_per_column = df.isna().sum()

# Total number of missing values in the dataset
total_missing = df.isna().sum().sum()
print(f'all columns and number of missing values:\n', missing_per_column)

print(f'\ntotal number of missing values in this dataFrame:\n', total_missing)

# This dataset does not have many missing values but if it did:
# use the df.dropna() get rid of the missing data,
# use the .fillna() to fill in for the missing values,
# use the .transform(lambda x: x.fillna(x.mean())) to fill values with mean values


# Data Wrangling: Normalization on numberica columns, using min max scaler
norm_columns = ['Unit', 'Sales']

# set minmaxscaler
scaler = MinMaxScaler()

# fit and transform the columns above
# copy df so you dont minipulate original dataframe
df_normalized = df.copy()
df_normalized[norm_columns] = scaler.fit_transform(df[norm_columns])

# Print first 5 rows of transformed data
print(df_normalized.head())

# print summary of transformed data
print(df_normalized.describe())

# using the groupby function for chunking to categoricalize columns
# total sales by the state
df.groupby('State')['Sales'].sum().sort_values(ascending=False)

# Average units sold by the product
df.groupby('Group')['Unit'].mean()

# Average sales by state and the time of the day
df.groupby(['State', 'Time'])['Sales'].mean()

# Adding column for group avarage sales
group_avg_sales = df.groupby('Group')['Sales'].transform('mean')
df['GroupAvgSales'] = group_avg_sales

# Descriptive stats for 'Sales' and 'Unit' columns
sales_stats = {
    'Sales Mean': df['Sales'].mean(),
    'Sales Median': df['Sales'].median(),
    'Sales Mode': df['Sales'].mode().iloc[0],
    'Sales Std Dev': df['Sales'].std()
}

unit_stats = {
    'Unit Mean': df['Unit'].mean(),
    'Unit Median': df['Unit'].median(),
    'Unit Mode': df['Unit'].mode().iloc[0],
    'Unit Std Dev': df['Unit'].std()
}

print(sales_stats, unit_stats)

# Group by Group and Sales columns
group_sales = df.groupby('Group')['Sales'].sum().sort_values(ascending=False)
# Identify group with highest and lowest sales
highest_sales_group = group_sales.idxmax(), group_sales.max()
lowest_sales_group = group_sales.idxmin(), group_sales.min()

print(group_sales, highest_sales_group, lowest_sales_group)

# Group with the highest sales
df.groupby('Group')['Sales'].sum()

# First, make sure 'Date' is datetime (already done, but reconfirming)
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as index for resampling
df_reports = df.set_index('Date')

# Weekly Sales and Units
weekly_report = df_reports.resample('W').agg({'Sales': 'sum', 'Unit': 'sum'})

# Monthly Sales and Units
monthly_report = df_reports.resample('M').agg({'Sales': 'sum', 'Unit': 'sum'})

# Quarterly Sales and Units
quarterly_report = df_reports.resample(
    'Q').agg({'Sales': 'sum', 'Unit': 'sum'})

print(weekly_report.head())
print(monthly_report.head())
print(quarterly_report.head())

# Creating csv exports for the reports
weekly_report.to_csv('weekly_sales_report.csv')
monthly_report.to_csv('monthly_sales_report.csv')
quarterly_report.to_csv('quarterly_sales_report.csv')

# Data Visualization

# Convert Date to datetime if needed
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Normalize categorical columns
for col in ['Time', 'State', 'Group']:
    df[col] = df[col].str.strip().str.title()

# Filter only positive values (optional)
df = df[(df['Sales'] > 0) & (df['Unit'] > 0)]

# 1. State-wise sales by demographic group
state_group_sales = df.groupby(['State', 'Group'])['Sales'].sum().reset_index()

# 2. Group-wise sales across states
group_state_sales = df.groupby(['Group', 'State'])['Sales'].sum().reset_index()

# 3. Time-of-day sales pattern
time_sales = df.groupby('Time')['Sales'].sum().reset_index()

# Plot 1: State-wise sales by demographic group
fig1 = px.bar(state_group_sales, x='State', y='Sales', color='Group', barmode='group',
              title='State-wise Sales by Demographic Group')
fig1.show()

# Plot 2: Group-wise sales across states
fig2 = px.bar(group_state_sales, x='Group', y='Sales', color='State', barmode='group',
              title='Group-wise Sales by State')
fig2.show()

# Plot 3: Time-of-day sales analysis
fig3 = px.pie(time_sales, names='Time', values='Sales',
              title='Sales Distribution by Time of Day')
fig3.show()

# Re-aggregate explicitly selecting only numeric columns to avoid FutureWarnings
numeric_df = df[['Sales', 'Unit']]

# Aggregate sales by daily, weekly, monthly, and quarterly
daily_sales = numeric_df.resample('D').sum()
weekly_sales = numeric_df.resample('W').sum()
monthly_sales = numeric_df.resample('M').sum()
quarterly_sales = numeric_df.resample('Q').sum()

# Plotting with seaborn and matplotlib (no warnings)
sns.set(style='whitegrid', palette='muted', font_scale=1.1)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Sales & Marketing Dashboard: Temporal Sales Analysis',
             fontsize=16, fontweight='bold')

# Plot Daily Sales
sns.lineplot(data=daily_sales, x=daily_sales.index, y='Sales', ax=axes[0, 0])
axes[0, 0].set_title("Daily Sales Trend")
axes[0, 0].set_xlabel("Date")
axes[0, 0].set_ylabel("Sales ($)")

# Plot Weekly Sales
sns.lineplot(data=weekly_sales, x=weekly_sales.index, y='Sales', ax=axes[0, 1])
axes[0, 1].set_title("Weekly Sales Trend")
axes[0, 1].set_xlabel("Week")
axes[0, 1].set_ylabel("Sales ($)")

# Plot Monthly Sales
sns.lineplot(data=monthly_sales, x=monthly_sales.index,
             y='Sales', ax=axes[1, 0])
axes[1, 0].set_title("Monthly Sales Trend")
axes[1, 0].set_xlabel("Month")
axes[1, 0].set_ylabel("Sales ($)")

# Plot Quarterly Sales
sns.lineplot(data=quarterly_sales, x=quarterly_sales.index,
             y='Sales', ax=axes[1, 1])
axes[1, 1].set_title("Quarterly Sales Trend")
axes[1, 1].set_xlabel("Quarter")
axes[1, 1].set_ylabel("Sales ($)")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
