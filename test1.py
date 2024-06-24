# %% [markdown]
# Step 1. Ensure that you have the dataset file named `transactions.csv` in the current directory.
# 
# The dataset is a subset of https://www.kaggle.com/ealaxi/paysim1/version/2 which was originally generated as part of the following research:
# 
# E. A. Lopez-Rojas , A. Elmir, and S. Axelsson. "PaySim: A financial mobile money simulator for fraud detection". In: The 28th European Modeling and Simulation Symposium-EMSS, Larnaca, Cyprus. 2016

# %% [markdown]
# Step 2. Complete the following exercises.
# 
# 0. Read the dataset (`transactions.csv`) as a Pandas dataframe. Note that the first row of the CSV contains the column names.
# 
# 0. Return the column names as a list from the dataframe.
# 
# 0. Return the first k rows from the dataframe.
# 
# 0. Return a random sample of k rows from the dataframe.
# 
# 0. Return a list of the unique transaction types.
# 
# 0. Return a Pandas series of the top 10 transaction destinations with frequencies.
# 
# 0. Return all the rows from the dataframe for which fraud was detected.
# 
# 0. Bonus. Return a dataframe that contains the number of distinct destinations that each source has interacted with to, sorted in descending order. You will find [groupby](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html) and [agg](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.groupby.DataFrameGroupBy.aggregate.html) useful. The predefined aggregate functions are under `pandas.core.groupby.GroupBy.*`. See the [left hand column](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.groupby.DataFrameGroupBy.nunique.html).

# %% [markdown]
# Use the empty cell to test the exercises. If you modify the original `df`, you can rerun the cell containing `exercise_0`.

# %%
import pandas as pd
import matplotlib.pyplot as plt

def exercise_0(file):
    df=pd.read_csv(file)
    return df

def exercise_1(df):
    return df.columns

def exercise_2(df, k):
    return df.head(k)

def exercise_3(df, k):
    return df.sample(n=k)

def exercise_4(df):
    return df['type'].unique()

def exercise_5(df):
    df1=df['nameDest'].value_counts()
    return df1.head(10)

def exercise_6(df):
    df1=df[df['isFraud']==1]
    return df1


# %%
df = exercise_0('transactions.csv')
print(df)

# %%
# Test exercises here
print(exercise_1(df))


# %%
print(exercise_2(df,5))

# %%
print(exercise_3(df,7))

# %%
print(exercise_4(df))

# %%
print(exercise_5(df))

# %%
print(exercise_6(df))

# %% [markdown]
# Create graphs for the following. 
# 1. Transaction types bar chart, Transaction types split by fraud bar chart
# 1. Origin account balance delta v. Destination account balance delta scatter plot for Cash Out transactions
# 
# Ensure that the graphs have the following:
#  - Title
#  - Labeled Axes
#  
# The function plot the graph and then return a string containing a short description explaining the relevance of the chart.

# %%
import pandas as pd
import matplotlib.pyplot as plt

def visual_1(df):
    def transaction_counts(df):
        # Count occurrences of each transaction type
        return df['type'].value_counts()
    
    def transaction_counts_split_by_fraud(df):
        # Count occurrences of each transaction type split by fraud status
        return df.groupby(['type', 'isFraud']).size().unstack()

    fig, axs = plt.subplots(2, figsize=(6,10))
    
    # Plotting the transaction counts
    transaction_counts(df).plot(ax=axs[0], kind='bar', color='skyblue')
    axs[0].set_title('Transaction Counts by Type')
    axs[0].set_xlabel('Transaction Type')
    axs[0].set_ylabel('Frequency')
    
    # Plotting the transaction counts split by fraud
    transaction_counts_split_by_fraud(df).plot(ax=axs[1], kind='bar', color=['green', 'red'])
    axs[1].set_title('Transaction Counts by Type Split by Fraud Status')
    axs[1].set_xlabel('Transaction Type')
    axs[1].set_ylabel('Frequency')
    
    # Setting the figure's supertitle
    fig.suptitle('Transaction Types Analysis')
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Adding annotations to each bar
    for ax in axs:
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')
    
    return 'The charts show the distribution of transaction types and their fraud status, providing insights into which types are more commonly associated with fraudulent activities.'

# Example usage with a DataFrame `df`
visual_1(df)


# %%
def visual_2(df):
    def query(df):
        df['origin_delta']=df['oldbalanceOrg']-df['newbalanceOrig']
        df['destinations_delta']=df['newbalanceDest']-df['oldbalanceDest']
        return df[df['type']=='CASH_OUT']
    plot = query(df).plot.scatter(x='origin_delta',y='destinations_delta')
    plot.set_title('TODO')
    plot.set_xlim(left=-1e3, right=1e3)
    plot.set_ylim(bottom=-1e3, top=1e3)
    return 'cash_out between old balance at origin vs new balance at destination '

visual_2(df)


# %% [markdown]
# Use your newly-gained Pandas skills to find an insight from the dataset. You have full flexibility to go in whichever direction interests you. Please create a visual as above for this query. `visual_custom` should call `exercise_custom`.

# %%
df.groupby(['isFlaggedFraud','isFraud']).size()

# %%
def exercise_custom(df):
    return df[['isFlaggedFraud', 'isFraud']].value_counts()
    
def visual_custom(df):
    fig, axs = plt.subplots(1, figsize=(6,6))
    exercise_custom(df).plot(ax=axs, kind='bar', color='green')
    axs.set_title('Occurance')
    axs.set_xlabel("comparision")
    axs.set_ylabel("Occurances")
    for p in axs.patches:
        axs.annotate(p.get_height(), (p.get_x(), p.get_height()))

    return 'occurance of frauds which are flagged and frauds which are actually true '

visual_custom(df)

# %% [markdown]
# Submission
# 
# 1. Copy the exercises into `task1.py`.
# 2. Upload `task1.py` to Forage.

# %% [markdown]
# All done!
# 
# Your work will be instrumental for our team's continued success.


