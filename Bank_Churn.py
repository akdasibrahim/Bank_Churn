import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import sys


if not sys.warnoptions:
    warnings.simplefilter("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)


df = pd.read_csv('datasets/BankChurners.csv')

df.head()
df.shape
df.info()
df.describe().T
df.isnull().sum()

df = df[df != 'Unknown'].dropna()


df.drop(['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
         'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'],
        axis=1,inplace=True)

df.set_index('CLIENTNUM',inplace=True)
df.reset_index(inplace=True)
df.drop('CLIENTNUM',axis=1,inplace=True)


df.info()
df['Education_Level'].value_counts()
df['Income_Category'].value_counts()
df['Marital_Status'].value_counts()
df['Card_Category'].value_counts()


df_num = df.select_dtypes(['int','Float64'])
df_num.describe().T



def grab_col_names(dataframe, cat_th= 10, car_th= 20):
    """
    Veri setindeki kategorik, nümerik ve kategorik ama kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe : DataFrame
        değişken isimleri alınmak istenen dataframedir.
    cat_th : int, float
        numerik fakat kategorik değişkenler için sınıf eşik değeridir.
    car_th : int, float
        kategorik fakat kardinal değişkenlerin sınıf eşik değeridir.

    Returns
    -------
    cat_cols : list
        kategorik değişkenlerin listesi.
    num_cols : list
        numerik değişkenlerin listesi.
     cat_but_car : list
        kategorik görünümlü kardinal değişkenlerin listesi.
    Notes
    -------
    cat_cols + num_cols + cat_but_car = Toplam değişken sayısı.
    num_but_cat cat_cols içerisinde.
    Return olan 3 liste toplamı toplam değişken sayısına eşittir.
    """

    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtype) in ['object', 'category', 'bool']]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].dtype in ['float', 'int'] and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in cat_cols if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtype) in ['object', 'category']]

    cat_cols = num_but_cat + cat_cols
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtype in ['float', 'int']]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observation: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car, num_but_cat

grab_col_names(df, cat_th=10, car_th=20)

cat_cols, num_cols, cat_but_car, num_but_cat= grab_col_names(df, cat_th=10, car_th=20)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
           'Ratio': 100*dataframe[col_name].value_counts()/len(dataframe)}))
    print('###########################################')
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.title(col_name.upper() + ' Grafiği')
        plt.show(block=True)

def num_summary(dataframe, numarical_cols, plot=False):
    quantiles= [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(dataframe[numarical_cols].quantile(quantiles))

    if plot:
        dataframe[numarical_cols].hist()
        plt.title(numarical_cols.upper() + ' Histogram')
        plt.xlabel(numarical_cols.upper())
        plt.show()

for col in cat_cols:
    cat_summary(df, col, plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)

plt.figure(figsize=(12,8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Korelasyon Haritası")
plt.show()

df['Attrition_Flag'].value_counts()
sns.countplot(x='Attrition_Flag', data=df)
plt.title("Müşteri Kaybı Dağılımı (Attrition_Flag)")
plt.show()


for col in cat_cols:
    print(f"{col} vs Attrition_Flag")
    print(pd.crosstab(df[col], df['Attrition_Flag'], normalize='index') * 100)
    print("##########################################")

for col in num_cols:
    print(f"{col} ortalamaları:")
    print(df.groupby("Attrition_Flag")[col].mean())
    print("##########################################")

for col in num_cols:
    plt.figure(figsize=(8,4))
    sns.boxplot(x="Attrition_Flag", y=col, data=df)
    plt.title(f"{col} Dağılımı (Attrition_Flag'e göre)")
    plt.show()


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * iqr
    up_limit = quartile3 + 1.5 * iqr
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, ":", check_outlier(df, col))

cat_cols = [col for col in cat_cols if col != 'Attrition_Flag']

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)



scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df['Attrition_Flag'] = np.where(df['Attrition_Flag'] == 'Attrited Customer', 1, 0)

X = df.drop('Attrition_Flag', axis=1)
y = df['Attrition_Flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.show()