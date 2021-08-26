import pandas as pd
from matplotlib import pyplot as plt
import pymongo

# Connection to MongoDB
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["churn-prediction"]
mycol = mydb["train-data-churn-prediction-balance"]

# Import Dataset From MongoDB
df = pd.DataFrame(list(mycol.find()))
print(df.head(10))
print("\n")

# Drop Columns
df.drop("customerID", axis='columns', inplace=True)
df.drop("_id", axis='columns', inplace=True)
print(df.dtypes)

# Total Charges Values
print(df.TotalCharges.values)

# Monthly Charges Values
print(df.MonthlyCharges.values)

# Checking TotalCharges Data
print(df[pd.to_numeric(df.TotalCharges,errors="coerce").isnull()])
print(df.shape)

# Iloc Data
print(df.iloc[488]['TotalCharges'])

df1=df[df.TotalCharges!=' ']
print(df1.shape)
print(df1.dtypes)

# Convert string to numeric
df1.SeniorCitizen = pd.to_numeric(df1.SeniorCitizen)
df1.tenure = pd.to_numeric(df1.tenure)
df1.TotalCharges = pd.to_numeric(df1.TotalCharges)
df1.MonthlyCharges = pd.to_numeric(df1.MonthlyCharges)
print(df1.TotalCharges.dtypes)

# Visualization Churn = No and Churn = Yes
tenure_churn_no = df1[df1.Churn=='No'].tenure
tenure_churn_yes = df1[df1.Churn=='Yes'].tenure
plt.hist([tenure_churn_yes, tenure_churn_no],color=['green','red'],label=['Churn = Yes','Churn = No'])
plt.legend()
# plt.show()

# Visualization Churn = No and Churn = Yes with Monthly Charges
mc_churn_no = df1[df1.Churn=='No'].MonthlyCharges
mc_churn_yes = df1[df1.Churn=='Yes'].MonthlyCharges

plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")
plt.title("Customer Churn Prediction Visualization")

blood_sugar_men = [113,85,90,150,88,93,115,135,80,77,82,129]
blood_sugar_women = [67,98,89,120,133,150,84,69,89,79,120,112,100]

plt.hist([mc_churn_yes, mc_churn_no], rwidth=0.95, color=["green","red"],label=["Churn = Yes", "Churn = No"])
plt.legend()
# plt.show()

# Function Print Unique Value
def print_unique_col_values(df):
    for column in df:
        if df[column].dtypes == 'object':
            print(f'{column}: {df[column].unique()}')

# Call Function to Prin Unique Value
print_unique_col_values(df1)

# Replace Value No internet service and No phone service
df1.replace('No internet service','No',inplace=True)
df1.replace('No phone service','No',inplace=True)

# Call Function to Prin Unique Value
print_unique_col_values(df1)

# Creating Array with Feature Column in Data
yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']

# Looping in Array to Replace Value Yes and No
for col in yes_no_columns:
    df1[col].replace({'Yes': 1,'No': 0},inplace=True)

# Looping Formating to Display Column and Unique Value in Column
for col in df1:
    print(f'{col}: {df1[col].unique()}')

# Looping in Array to Replace Value Yes and No
df1['gender'].replace({'Female':1,'Male':0},inplace=True)
print(df1.gender.unique())

# Get Data in Feature Column
df2 = pd.get_dummies(data=df1, columns=['InternetService','Contract','PaymentMethod'])
print(df2.columns)

# Print 5 Data and Print Data Type
print(df2.sample(5))
print(df2.dtypes)

# Create Scalling Data in Feature Tenure, Monthly Charges, Total Charges
cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])

# Looping Formating to Display Column and Unique Value in Column
for col in df2:
    print(f'{col}: {df2[col].unique()}')

# Spliting Feature Column Between Churn and Other Churn
X = df2.drop('Churn',axis='columns')
y = df2['Churn']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=5)

# Print X Train and X Test
print(X_train.shape)
print(X_test.shape)
print(X_train[:10])

# Check Length Column in X Train
len(X_train.columns)

# Implementation ANN Modelling
from tensorflow import keras
model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5000)
















