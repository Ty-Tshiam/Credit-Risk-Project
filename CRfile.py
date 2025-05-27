# %% [markdown]
# # Credit Risk Project

# %% [markdown]
# First and foremost the goal is to learn about credit risk and applying machine learning and analysis.
# 
# We need a data set to work with. I am going to look on Kaggle.

# %% [markdown]
# Before starting, I am going to lay out some stuff I know about credit risk and how machine learning applies. First credit risk is quantifying or decidimg how risky it is to loan out money to a person or organization based on the information on their application. Some ML algorithms that would be useful would be supervised learning to help the algorithm make decisions on whether to give a loan to someone based on their labels of past successful or unsuccessful algorithms. ML algorithms that work well with tabular data like XGBoost or Random Forest would be good for this situation. Unsupervised algorithms would be use in order to group applications into certain categories without specifically specifying their labels. They would be group on how similar they are to other. Classification algorithms are the way to go on this one (I think... I haven't seen the data). Specifically Logistical Classification helps with probability of a grant or maybe even supper vector machines that could split the accepted vs unaccepted applications. Knowing all this so far let's do some EDA and figure it out from there. 

# %%
import kagglehub

# Download latest versionc:\Users\ty725\.cache\kagglehub\datasets\wordsforthewise\lending-club\versions\3
path = kagglehub.dataset_download("wordsforthewise/lending-club")

print("Path to dataset files:", path)

# %%
path = "C:\\Users\\ty725\.cache\kagglehub\\datasets\wordsforthewise\\lending-club\\versions\\3"
path1 = path + "\\accepted_2007_to_2018q4.csv\\accepted_2007_to_2018Q4.csv"
path2 = path + "\\rejected_2007_to_2018q4.csv\\rejected_2007_to_2018Q4.csv"

# %% [markdown]
# Okay these datasets are huge! I think this is a good oppurtunity to use PySpark. I heard that it works really well with huge datasets. 

# %% [markdown]
# ## Dataset EDA

# %%
from pyspark.sql import SparkSession

# Start new session
spark = SparkSession.builder \
    .appName("NewSession") \
    .getOrCreate()

# %%
dfa = spark.read.csv(path1, header=True)

# %% [markdown]
# Ok so there's a one thing to notice other than the amount of rows being huge: the columns are very different for the rejected pile vs the accepted pile. The accepted pile includes way more information about the actual loan that give out as well. So before we start cleaning, we have to figure which of the most important features/columns we need to clean. The best way to do that is to figure out which of the columns match because the goal is to get the accepted and the reject in one pile for the ML algo. 

# %% [markdown]
# After some inspiration, I understand how I am supposed to approach this problem. Focus on only accepted loans to predict the probability of default among accepted loans. The key column here is the loan status, mainly between statuses "Charged Off" vs "Fully Paid".

# %% [markdown]
# Ok so the next step is to load it on to the cloud and query it so we can get a dirty dataset. We'll use PySpark to clean it on the occasion that the dataset is still huge.

# %%
dfa.createOrReplaceTempView('loans')

# %%
df = spark.sql('''
    SELECT * FROM loans
    WHERE loan_status = 'Charged Off' OR  loan_status = 'Fully Paid'
''')
df.show()

# %%
#df.describe().show()

# %%
df.printSchema()

# %%
rows = df.count()
rows

# %%
from pyspark.sql.functions import col, isnull, sum

def nullCount(dft):
    nullCounts = dft.agg(*[
        ((sum(isnull(col(c)).cast('integer')) / rows) * 100).alias(c)
        for c in dft.columns
    ])
    nullCounts.show()

nullCount(df)

# %% [markdown]
# I know I keep saying this but this dataset really is huge. My first approach was to go through each of attributes, figure out what they mean, and then figure out if they are useful. The thing is there is 151 attributes and I don't have all night here. We can't have a lot of attributes because this can cause the model to overfit, the key is generalization for a good ML model. So instead of getting 151 attributes down to 20 or so, it's better to hand pick the key features we need for our model and then clean that.

# %%
features = '''
id, loan_status, loan_amnt, int_rate, installment, grade, sub_grade, emp_length,
home_ownership, annual_inc, verification_status, purpose, dti, delinq_2yrs, 
inq_last_6mths, fico_range_low, fico_range_high
'''

def cleanStrings(strings):
    strings = strings.split(',')
    for i, string in enumerate(strings):
        strings[i] = string.strip()
    return strings



# %%
features = cleanStrings(features)
df = df.select(features)

# %%
#df.toPandas()

# %% [markdown]
# ## Data Cleaning 

# %%
df.show(5)

# %%
nullCount(df)

# %% [markdown]
# Ok all missing values seems to be under 6% so it safe to drop all the rows that have a missing values. This saves a massive headache. Now there is something to note. Looking at the data, I assumed that if they're missing employment length, then that must mean that they're unemployed. Using data wrangler, some people have job titles but no employment length, so the unemployment assumption is not always valid. This is kind of significant to note becase employment length has the highest null count of 6%, everything else is less significant. Any ways, we'll drop any rows with missing cells, this dataset is huge.

# %%
df.printSchema()

# %%
colsFloats ='loan_amnt, int_rate, installment, annual_inc, dti'
colsInts = 'delinq_2yrs, fico_range_low, fico_range_high, inq_last_6mths'

colsFloats = cleanStrings(colsFloats)
colsInts = cleanStrings(colsInts)

for column in colsFloats:
    df = df.withColumn(column, col(column).cast('float'))

for column in colsInts:
    df = df.withColumn(column, col(column).cast('int'))

# %%
df = df.na.drop()
rows = df.count()
nullCount(df)

# %%
rows = df.count()
rows

# %%
df.printSchema()

# %% [markdown]
# Ok so we're going to try something. We're going to apply One Hot Encoder becase we have some attributes that cannot be easily converted clean numbers for the algo to understand. One Hot Encoder is good for categorical data, namely in this case: loan_status, grade, sub_grade, emp_length (maybe, there's still hope for this one), home_ownership, verification_status, and purpose.

# %%
df = df.filter(col('home_ownership') != 'ANY')

# %%
otherInfo = df.filter(col('home_ownership') == 'OTHER').groupBy('loan_status').count()
otherInfo

# %%
NoneInfo =  df.filter(col('home_ownership') == 'NONE').groupBy('loan_status').count()
NoneInfo

# %%
from pyspark.sql.functions import when, col
common_owners = ["RENT", "OWN", "MORTGAGE"]

df = df.withColumn(
    "home_ownership",
    when(col("home_ownership").isin(common_owners), col("home_ownership"))
    .otherwise("OTHER")
)

# %%
encodedCols = 'loan_status, emp_length, home_ownership, verification_status, purpose'
encodedCols = cleanStrings(encodedCols)
encodedCols

# %%
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

stages = []
for c in encodedCols:
    indexer = StringIndexer(inputCol = c, outputCol = c + '_')
    encoder = OneHotEncoder(inputCol = c + '_', outputCol = c + '-', dropLast = True)
    stages += [indexer, encoder]

pipeline = Pipeline(stages=stages)
encoded_df = pipeline.fit(df).transform(df)

# %%
encoded_df.show(5)

# %%
empInfo = encoded_df.groupBy("emp_length").count()
empInfo

# %%
empLoanInfo = encoded_df.groupBy("emp_length", "loan_status").count()
empLoanInfo

# %%
import pandas as pd
empLoanInfo = empLoanInfo.toPandas()

# %%
def pivotIt(dft, index):
    pivot = dft.pivot_table(index = index, columns = "loan_status", values = "count", aggfunc = "sum")
    pivot['total'] = pivot.loc[:, 'Charged Off'] + pivot.loc[:, 'Fully Paid']
    pivot['default_percentage'] = (pivot.loc[:, 'Charged Off'] / pivot['total']) * 100
    return pivot.sort_values(ascending = True, by = 'default_percentage')

# %%
pivotIt(empLoanInfo, "emp_length")

# %% [markdown]
# With emp_length, after some data analysis I found that the difference is between 10+ and <1 is 1.75% (with 10+ having a lower charged off percentage), but the range between all the other numbers (2-9) was 0.6% and they were pretty random in order. Based on my findings I think I should use One Hot Encoding because the linear progression at least for the middle numbers is none existent. For the remaining columns (grade and subgrade) needs to be encoded in ordinally which basically means that the order matters when encoding. 

# %%
grade_df = encoded_df.groupBy("grade", "loan_status").count().toPandas()
pivotIt(grade_df, "grade")

# %%
subgrade_df = encoded_df.groupBy("sub_grade", "loan_status").count().toPandas()
pivotIt(subgrade_df, "sub_grade")

# %% [markdown]
# This confirms the decision for Ordinal Encoding.

# %%
grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
levels = [1,2,3,4,5]
submap = {}
k = 0
ran = len(grades) * len(levels)
for i in range(len(grades)):
    for j in range(len(levels)):
        submap[grades[i] + str(levels[j])] = k
        k += 1
        
map = {}
for i in range(len(grades)):
    map[grades[i]] = i

# %%
from pyspark.sql.functions import lit
encoded_df = encoded_df.withColumn('sub_grade-', lit(None))
encoded_df = encoded_df.withColumn('grade-', lit(None))

for value, key in enumerate(map):
    encoded_df = encoded_df.withColumn('grade-', when(col("grade") == lit(key), value).otherwise(col("grade-")))

for value, key in enumerate(submap):
    encoded_df = encoded_df.withColumn('sub_grade-', when(col("sub_grade") == lit(key), value).otherwise(col("sub_grade-")))


# %%
encoded_df.show(5)

# %%
col_names = '''id, loan_status-, loan_amnt, int_rate, installment, grade-, sub_grade-, 
emp_length-, home_ownership-, annual_inc, verification_status-, purpose-, dti, 
delinq_2yrs, inq_last_6mths, fico_range_low, fico_range_high
'''
col_names = cleanStrings(col_names)
df = encoded_df.select(col_names)
df.show(5)

# %%
for i in range(len(col_names)):
    df = df.withColumnRenamed(col_names[i], features[i])   

# %%
df.printSchema()

# %%
df.count()

# %%
df.show(5)

# %%
df.sort(col("delinq_2yrs").desc()).show(5)

# %%
df.sort("loan_amnt").show(5)

# %% [markdown]
# Ok almost everything seems to be in order. I need one extra check to make sure the data is clean. Next I'm going to handle outliers, logical consistencies, and known ranges. 

# %%
df.filter(col('fico_range_low') > col('fico_range_high')).count()

# %%
df.createOrReplaceTempView('cleanDataset')

# %%
query = spark.sql('''
SELECT * FROM cleanDataset
WHERE (fico_range_low < 300 OR fico_range_low > 850)
                  OR (fico_range_high < 300 OR fico_range_high > 850)
''')
query.count()

# %%
query = spark.sql('''SELECT * FROM cleanDataset
                  WHERE loan_amnt < 0
''')
query.count()

# %% [markdown]
# I checked manually for negative values and max loan amount using the sort function for each column (that's numbered anyway). Now time to handle outline.

# %%
def printQuantiles(column):
    quant = df.approxQuantile(column, probabilities = [0.25, 0.75], relativeError = 0)
    IQR = quant[1] - quant[0]
    lower = quant[0] - 1.5 * IQR
    upper = quant[1] + 1.5 * IQR
    print(column)
    print(f'Lower : {lower}')
    print(f'Upper : {upper}')
    print(f'IQR : {IQR}')

# %%
#for feature in colsInts:
    #printQuantiles(feature)


# %%
#for feature in colsFloats:
    #printQuantiles(feature)

# %% [markdown]
# Ok so looking through all the information there is no one size fits all appraoch, you'd have to do deep analysis of each feature to figure out the best appraoch. Now, in general I'm going use a general approach of capping outliers to a certain value. I'm doing this because extreme values are legit but could be too influential during model training. 

# %%
capVals = {"loan_amnt" : 35000, "int_rate" : 30, "installment" : 1100,
           "annual_inc" : 250000, "dti" : 45, "delinq_2yrs" : 4, 
           "inq_last_6mths" : 6}

# %%
for key, values in capVals.items():
    df = df.withColumn(key, when(col(key) > values,  values).otherwise(col(key)))
    df = df.withColumn(key, when(col(key) < 0, 0).otherwise(col(key)))
df.show(10)

# %%
df.describe().show()

# %%
from pyspark.ml.functions import vector_to_array
for feature in encodedCols:
    df = df.withColumn(feature, vector_to_array(col(feature)))

# %%
df.write.mode('overwrite').parquet("cleanLoans.parquet")

# %%
dataset = pd.read_parquet("cleanLoans.parquet")

# %%
dataset

# %%
def expandColumn(dataset, colName):
    columns = []
    for i in range(len(dataset[colName][0])):
        columns.append(f"{colName}_{i}")
    arr = pd.DataFrame(dataset[colName].tolist(), columns = columns)
    dataset = dataset.drop(columns = [colName])
    return pd.concat([dataset, arr], axis = 1)

# %%
df = dataset
for feature in encodedCols:
    df = expandColumn(df, feature)

# %%
df.drop(axis = 1, columns = ['id'], inplace = True)


# %%
df

# %% [markdown]
# # Model Building 

# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# %%
X = df.drop(columns = ['loan_status_0']).to_numpy()
Y = df['loan_status_0'].to_numpy()

# %%
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.2,  
                                                    random_state = 42, stratify = Y)

# %%
scaler = StandardScaler()
xTrainS = scaler.fit_transform(xTrain)
xTestS = scaler.transform(xTest)

# %%
model = LogisticRegression(max_iter = 1000, class_weight = 'balanced')
model.fit(xTrainS, yTrain)

# %%
yPred = model.predict(xTestS)
yProb = model.predict_proba(xTestS)[:, 0]

# %%
print(classification_report(yTest, yPred))

# %%
print(f'ROC AUC Score: {roc_auc_score(yTest, yProb)}')

# %%
probs = pd.DataFrame(yProb, columns = ["Probability of Default"])
probs = probs * 100
probs

# %%
# Fully Paid = 1 and Charged Off = 0
pred = pd.DataFrame(yPred, columns = ['Prediction'])
pred

# %%
# Fully Paid = 1 and Charged Off = 0
test = pd.DataFrame(yTest, columns = ['Real Values'])
test

# %%
results = pd.concat([probs, pred, test], axis = 1)
results

# %%
from sqlalchemy import create_engine 
server = 'dahomey.database.windows.net'
database = 'Stock Data'
username = '****'
password = '****'
driver = '{ODBC Driver 18 for SQL Server}'

conn_url = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+18+for+SQL+Server'
engine = create_engine(conn_url)
try:
    results.to_sql(name="LoanPredictions", con=engine, if_exists="replace", index=False)
except Exception as e:
    results.to_sql(name="LoanPredictions", con=engine, if_exists="replace", index=False)

# %%
resultsPT = results.groupby(['Prediction', 'Real Values']).count()
resultsPT = resultsPT.rename(columns={'Probability of Default': 'Count'})
resultsPT

# %% [markdown]
# # LGD

# %%
from pyspark.sql import SparkSession

# Start new session
spark = SparkSession.builder \
    .appName("NewSession") \
    .getOrCreate()

# %%
df = spark.sql('''
SELECT l.total_pymnt, cd.loan_amnt, cd.int_rate, cd.annual_inc, 
               cd.dti, cd.purpose, cd.fico_range_low, cd.emp_length, cd.grade
FROM loans AS l
JOIN cleanDataset AS cd ON cd.id = l.id
WHERE l.loan_status = 'Charged Off'

''')
df.show(5)

# %%
from pyspark.sql.functions import col
df = df.withColumn('total_pymnt', col('total_pymnt').astype('float'))
df = df.na.drop()
df.show(5)

# %%
df = df.withColumnRenamed('loan_amnt', 'EAD')
df = df.withColumnRenamed('total_pymnt', 'Recovery')

# %%
from pyspark.ml.functions import vector_to_array
cols = ['purpose', 'emp_length']
for feature in cols:
    df = df.withColumn(feature, vector_to_array(col(feature)))

# %%
df = df.toPandas()
df

# %%
df['LGD'] = ((df['EAD'] - df['Recovery']) / df['EAD']) 

# %%
df = expandColumn(df, 'purpose')
df = expandColumn(df, 'emp_length')

# %%
df = df.drop(axis = 1, columns = ['Recovery'])
df

# %%
capVals = {"EAD" : 35000, "int_rate" : 30, "LGD" : 100, 
           "annual_inc" : 250000, "dti" : 45}

# %%
for key, value in capVals.items():
    df[key] = df[key].clip(lower = 0, upper = value)

# %%
X = df.drop(axis = 1, columns = ['LGD', 'EAD'])
Y = df['LGD'] / 100

# %%
X = X.to_numpy()
Y = Y.to_numpy()

# %%
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size  = 0.2, random_state = 42)

# %%
scaler = StandardScaler()
xTrainS = scaler.fit_transform(xTrain)
xTestS = scaler.transform(xTest)

# %%
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xTrainS, yTrain)

# %%
yPred = model.predict(xTestS)

# %%
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(yTest, yPred)
rmse = np.sqrt(mse)
r2 = r2_score(yTest, yPred)
mae = mean_absolute_error(yTest, yPred)

# %%
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 6))
sns.scatterplot(x=yTest, y=yPred, alpha=0.3)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # y = x line
plt.xlabel('Actual LGD')
plt.ylabel('Predicted LGD')
plt.title('Actual vs Predicted LGD')
plt.show()



