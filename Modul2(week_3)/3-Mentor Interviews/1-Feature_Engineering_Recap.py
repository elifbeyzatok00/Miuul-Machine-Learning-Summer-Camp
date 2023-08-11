#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def load_application_train():
    data = pd.read_csv("C:/Users/Tugce.Dogan/Downloads/application_train.csv")
    return data

dff = load_application_train()
dff.head()


def load():
    data = pd.read_csv("C:/Users/Tugce.Dogan/Downloads/titanic.csv")
    return data

df = load()
df.head()

#############################################
# OUTLIERS
#############################################

#############################################
# AYKIRI GOZLEMLERI YAKALAMAK
#############################################

####################################################
# Grafik teknikle aykırılar nasıl gözlemlenir?
###################################################
sns.boxplot(x=df["Age"])
plt.show()

df["Age"].describe().T

####################################################
# Aykırı değerler nasıl yakalanır?
####################################################

q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

# alt sınırdan küçük ya da üst sınırdan büyük olanlar
df[(df["Age"] < low) | (df["Age"] > up)]
df[(df["Age"] < low) | (df["Age"] > up)]["Age"]
df[(df["Age"] < low) | (df["Age"] > up)].index

####################################################
# Aykırı değer var mı yok mu?
####################################################

df[(df["Age"] > up) | (df["Age"] < low)].any(axis=None)

# 1. eşik deger belirledik
# 2. aykırılara eriştik
# 3. aykırı var mı diye sorduk

###################
# İşlemleri Fonksiyonlaştırmak
###################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, "Age")
outlier_thresholds(df, "Fare")

low, up = outlier_thresholds(df, "Age")

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


check_outlier(df, "Age")

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df, col))

dff = load_application_train()
cat_cols, num_cols, cat_but_car = grab_col_names(dff)

num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]

for col in num_cols:
    print(col, check_outlier(dff, col))

####################################################
# Aykırı Gözlemleri Yakalamak
####################################################

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "Age")
grab_outliers(df, "Age", True)
age_index = grab_outliers(df, "Age", True)

# RECAP

outlier_thresholds(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", True)


#############################################
# AYKIRI GOZLEM PROBLEMINI COZME
#############################################

#############################################
# SİLME
#############################################

low, up = outlier_thresholds(df, "Fare")
df.shape

df[~((df["Fare"] < low) | (df["Fare"] > up))].shape

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

df.shape
remove_outlier(df, "Fare").shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    new_df = remove_outlier(df, col)

#2.yol
for col in ["Age", "Fare"]:
    new_df = remove_outlier(df, col)

df.shape[0] - new_df.shape[0]

#############################################
# BASKILAMA YONTEMI (re-assignment with thresholds)
#############################################

low, up = outlier_thresholds(df, "Fare")

df[((df["Fare"]>up) | (df["Fare"]< low))]["Fare"]

df.loc[(df["Fare"] > up), "Fare"] = up

# titanic veri setinde low değeri -26.724 çıktı. Fare eksi değer olamaz. bu sebeple low ataması yapmamak lazım
df.loc[(df["Fare"] < low), "Fare"] = low

# genelleştirilmiş fonk hali:

def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if low_limit > 0:
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    else:
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit


df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
df.shape

for col in ["Age", "Fare"]:
    print(col, check_outlier(df, col))

for col in ["Age", "Fare"]:
    replace_with_thresholds(df, col)

for col in ["Age", "Fare"]:
    print(col, check_outlier(df, col))


#########################
# RECAP
########################

df = load()

outlier_thresholds(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", index=True)

remove_outlier(df, "Age").shape
replace_with_thresholds(df, "Age")
check_outlier(df, "Age")

#############################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor ( LOF )
#############################

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()

for col in df.columns:
    print(col, check_outlier(df, col))

low, up = outlier_thresholds(df, "carat")

df[((df["carat"] < low) | (df["carat"] > up))].shape

low, up = outlier_thresholds(df, "depth")

df[((df["depth"] < low) | (df["depth"] > up))].shape

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
df_scores[0:5]

np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style='.-')
plt.show()

th = np.sort(df_scores)[3]

df[df_scores < th]

df[df_scores < th].shape

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

# label'ına göre bu lof ile yakaladığımız aykırı değ.lerden 1 tanesini silmek istersek;
df[df_scores < th].drop(axis=0, labels=41918)

# hepsini silmek istersek;
df=df.drop(axis=0, labels=df[df_scores < th].index)

# ÖDEV (Keyfi)
# LOF ile yakalanan index'lerdeki aykırı değerleri değişkenleri tek tek düşünerek baskılayınız.

def outlier_thresholds(dataframe, col_name):
    quartile1 = dataframe[col_name].quantile(0.05)
    quartile3 = dataframe[col_name].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if low_limit > 0:
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    else:
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit

df_lof=df[df_scores < th]

df_lof.index

outlier_thresholds(df,df.columns)

for i in df_lof.index:
    for col in ["carat", "depth", "table", "price", "x", "y", "z"]:
        replace_with_thresholds(df, col)
        print(i, col, df.loc[i, col])

df_lof = df[df_scores < th]
outlier_thresholds(df, df.columns)


df[df.index== 41918]
df[df.index== 48410]
df[df.index== 49189]


# YORUM:
#########

#        carat  depth  table  price     x      y      z
# 41918  1.030 78.200 54.000   1262 5.720  5.590  4.420
# 48410  0.510 61.800 54.700   1970 5.120  5.150 31.800
# 49189  0.510 61.800 55.000   2075 5.150 31.800  5.120

#        carat  depth  table    price     x     y     z
# 41918  1.030 70.550 54.000 1262.000 5.720 5.590 4.420
# 48410  0.510 61.800 54.700 1970.000 5.120 5.150 7.850
# 49189  0.510 61.800 55.000 2075.000 5.150 12.675 5.120



#############################################
# MISSING VALUES
#############################################

#############################################
# EKSIK DEGERLERIN YAKALANMASI
#############################################

df = load()
df.head()

df.isnull().values.any()

df.isnull().sum()

df.notnull().sum()

df[df.isnull().any(axis=1)]

df[df.notnull().all(axis=1)]

df.isnull().sum().sort_values(ascending=False)

# Oransal olarak görmek icin
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)
missing_values_table(df, True)


dff = load_application_train()
missing_values_table(dff)

#############################################
# EKSIK DEGER PROBLEMINI COZME
#############################################

####
# COZUM 1: Hızlıca silmek
####

df.dropna()

#############################################
# COZUM 2: Basit Atama Yöntemleri ile Doldurmak (Lambda ve apply ile doldurmak)
#############################################
df = load()
df["Age"].fillna(0)
df["Age"].fillna(df["Age"].mean())
df["Age"].fillna(df["Age"].median())

df.apply(lambda x: x.fillna(x.mean()), axis=0)

df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()

df = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

df.isnull().sum().sort_values(ascending=False)

df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()

# eksik değerleri bazen takip edip çeşitli isimlendirmeler yazarak da takip edebilirz.
df["Embarked"].fillna("missing")

###################
# Kategorik Değişken Kırılımında Değer Atama
###################

df.groupby("Sex")["Age"].mean()

df["Age"].mean()

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"]=="male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()

# TÜM KATEGORİK DEĞİŞKENLER için mod almayı fonksiyonlarştırırsak;
# kategorik değişken sınıfı 10'dan küçük olanları seçiyoruz çünkü kat deg sınıfı 10 dan büyük olanlar kat değişken değildir KARDİNAL ya da Numeric olarak sınıflandırmak daha doğru olur

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

#############################################
# Scikit-learn ile eksik deger atama
#############################################

# pip install scikit-learn

V1 = np.array([1, 3, 6, np.NaN, 7, 1, np.NaN, 9, 15])
V2 = np.array([7, np.NaN, 5, 8, 12, np.NaN, np.NaN, 2, 3])
V3 = np.array([np.NaN, 12, 5, 6, 14, 7, np.NaN, 2, 31])

df_v= pd.DataFrame(
    {"V1": V1,
     "V2": V2,
     "V3": V3}
)

df_v

from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

imp_mean.fit(df_v)
imp_mean.transform(df_v)

#############################################
# Kategorik Değişken Kırılımında Değer Atama
#############################################

V1 = np.array([1, 3, 6, np.NaN, 7, 1, np.NaN, 9, 15])
V2 = np.array([7, np.NaN, 5, 8, 12, np.NaN, np.NaN, 2, 3])
V3 = np.array([np.NaN, 12, 5, 6, 14, 7, np.NaN, 2, 31])
V4 = np.array(["IT", "IT", "IK", "IK", "IK", "IK", "IT", "IT", "IT"])

df_v = pd.DataFrame(
    {"salary": V1,
     "V2": V2,
     "V3": V3,
     "departman": V4}
)

df_v

df_v.groupby("departman")["salary"].mean()

df_v["salary"].fillna(df_v.groupby("departman")["salary"].transform("mean"))

# transform kullanmadan manuel yapmak istersek;
df_v.loc[(df_v["salary"].isnull()) & (df_v["departman"] == "IK"), "salary"] = df_v.groupby("departman")["salary"].mean()["IK"]
df_v.loc[(df_v["salary"].isnull()) & (df_v["departman"] == "IT"), "salary"] = df_v.groupby("departman")["salary"].mean()["IT"]

# titanic veri seti için: CİNSİYET ORT. larına göre Yaş değişkenindeki eksiklikleri doldurdu
df = load()
df.Age.isnull().sum()
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

#############################################
# Çözüm 3: Tahmine Dayalı Atama ile Doldurma
#############################################

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

dff.head()

# değişkenlerin standartlatırılması
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()


# knn'in uygulanması.
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

df["age_imputed_knn"] = dff[["Age"]]

df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]
df.loc[df["Age"].isnull()]


#############################################
# RECAP
#############################################

df = load()
# missing table
missing_values_table(df)
# sayısal değişkenleri median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

#############################################
# GELISMIS ANALIZLER
#############################################

#############################################
# EKSIK VERI YAPISININ INCELENMESI
#############################################
# eksiklikler ancak tamamen rasgele ortaya çıktıysa giderilebilir, temizlenebilri!!
# şayet eksik gözlemler arasında ilişki varsa bunu dikkate almak gerekir(çok değişkenli istatistik literatüründe)

msno.bar(df)
plt.show()

# matrix grafigi: değişkenler arasında yapısal bir eksiklik olup-olmadığını görmek açısından faydalıdır.
"""
Solution: get into missingno.py, search for grid(b=False) and update it to grid(visible=False). There should be 3 occurrences.
Once you done that, return to your own code, re-import the missingno packAge and it should work.
"""

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()


#############################################
# EKSIK DEGERLERIN BAGIMLI DEGISKEN ILE ILISKISININ INCELENMESI
#############################################

missing_values_table(df, True)
na_cols = missing_values_table(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Survived", na_cols)

#############################################
# RECAP
#############################################

df = load()
na_cols = missing_values_table(df, True)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile Doldurma
missing_vs_target(df, "Survived", na_cols)

#############################################
# 3. LABEL ENCODING(Binary Encoding)
#############################################

df = load()
df.head()
df["Sex"].head()
#from sklearn import preprocessing
#le = preprocessing.LabelEncoder()
le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]
le.inverse_transform([0, 1])

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

df = load()

binary_cols = [col for col in df.columns if df[col].dtypes == "O"
               and len(df[col].unique()) == 2]

#2.yol
#binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()

df = load_application_train()
df.shape

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

df[binary_cols].head()


for col in binary_cols:
    label_encoder(df, col)


df = load()
df["Embarked"].value_counts()
df["Embarked"].nunique()
len(df["Embarked"].unique())

#############################################
# 4. ONE-HOT ENCODING
#############################################

df = load()
df["Embarked"].value_counts()

pd.get_dummies(df, columns=["Embarked"]).head()
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()
pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()
pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = load()
# cat_cols, num_cols, cat_but_car = grab_col_names(df)
ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]

one_hot_encoder(df, ohe_cols).head()

one_hot_encoder(df, ohe_cols, drop_first=True).head()

df=one_hot_encoder(df, ohe_cols, drop_first=True)

#############################################
# 5. RARE ENCODING
#############################################

#############################################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
#############################################

df = load_application_train()
df.head()

df["NAME_EDUCATION_TYPE"].value_counts()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
#cat_cols = [col for col in df.columns if df[col].dtypes == 'O']

def cat_summary(dataframe,col_name,plot=False):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                        "Ratio":100* dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

#############################################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
#############################################

df["NAME_INCOME_TYPE"].value_counts()

df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

# 1. Sınıf Frekansı
# 2. Sınıf Oranı
# 3. Sınıfların target açısından değerlendirilmesi

[col for col in df.columns if df[col].dtypes == 'O' and (df[col].value_counts() / len(df) < 0.10).any(axis=None)]

def rare_analyser(dataframe, target, rare_perc):

    rare_columns = [col for col in dataframe.columns if dataframe[col].dtypes == 'O'
                    and (dataframe[col].value_counts() / len(dataframe) < rare_perc).any(axis=None)]

    for col in rare_columns:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "TARGET", 0.01)

"""
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)
"""

#############################################
# 3. Rare encoder'ın yazılması.
#############################################

temp_df = df.copy()
temp_df["ORGANIZATION_TYPE"].unique()
len(temp_df["ORGANIZATION_TYPE"].unique())
temp_df["ORGANIZATION_TYPE"].value_counts() / len(temp_df)
tmp = temp_df["ORGANIZATION_TYPE"].value_counts() / len(temp_df)
rare_labels = tmp[tmp < 0.01].index
len(rare_labels)

temp_df["ORGANIZATION_TYPE"] = np.where(temp_df["ORGANIZATION_TYPE"].isin(rare_labels), 'Rare',
                                        temp_df["ORGANIZATION_TYPE"])

len(temp_df["ORGANIZATION_TYPE"].unique())
temp_df["ORGANIZATION_TYPE"].value_counts() / len(temp_df)


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "TARGET", 0.01)
rare_analyser(df, "TARGET", 0.01)

df["OCCUPATION_TYPE"].value_counts()

#############################################
# 6. FEATURE SCALING (Özellik Ölçeklendirme)
#############################################

# StandardScaler: Klasik standartlaştırma z = (x - mu) / s
df = load()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(df[["Age"]])
df["Age"] = scaler.transform(df[["Age"]])
df["Age"].describe().T

#2. yol
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])

scaler = StandardScaler().fit(df[["Fare"]])
df["Fare"] = scaler.transform(df[["Fare"]])
df["Fare"].describe().T

# RobustScaler: z = (x - median) / IQR
df = load()
from sklearn.preprocessing import RobustScaler

transformer = RobustScaler().fit(df[["Age"]])
df["Age"] = transformer.transform(df[["Age"]])
df["Age"].describe().T

#2.yol
rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])

# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

df = load()
from sklearn.preprocessing import MinMaxScaler

transformer = MinMaxScaler().fit(df[["Age"]])
df["Age"] = transformer.transform(df[["Age"]])

df["Age"].describe().T

#2.yol
mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])

# Log: Logaritmik dönüşüm.

df = load()
df["Age"] = np.log(df["Age"])
df["Age"].describe().T

age_cols = [col for col in df.columns if "Age" in col]

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in age_cols:
    num_summary(df, col, plot=True)

###################
# Numeric to Categorical: Sayısal Değişkenleri Kateorik Değişkenlere Çevirme
# Binning
###################

df["Age_qcut"] = pd.qcut(df['Age'], 5)

df = load()

df.loc[(df['Age'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['Age'] >= 18) & (df['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.head()

#############################################
# 7. FEATURE EXTRACTION
#############################################

#############################################
# Binary Features: Flag, Bool, True-False
#############################################

df = load()
df.head()
df["NEW_CABIN_BOOL"] = df["Cabin"].isnull().astype('int') # null olarak yakaladıklarını 1 olarak işaretler
df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')  # null olmayan yakaladıklarını 1 olarak işaretler

df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})

from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#############################################
# Text'ler Üzerinden Özellik Türetmek
#############################################

#############################################
# LETTER COUNT
#############################################

df["NEW_NAME_COUNT"] = df["Name"].str.len()
df.head()

df.groupby("NEW_CABIN_BOOL").agg({"NEW_NAME_COUNT":"mean"})

#############################################
# WORD COUNT
#############################################

df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))


#############################################
# OZEL YAPILARI YAKALAMAK
#############################################

# Bu gemide Dr var mı? ve Dr olmak ile hayatta kalmak arasında ilişki varmı?

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr.")]))
df.head()

df["NEW_NAME_DR"].value_counts()

df.groupby("NEW_NAME_DR").agg({"Survived": ["mean", "count"]})

#############################################
# TITLE'LARI CEKMEK ( w/ Regex )
#############################################

df = load()
df.isnull().sum()
df["Age"].mean()

# name değişkeni içerisinden herbir title yakalamk için regex( boşluk büyük ya da küçük a'dan z'ye harflar nokta boşluk)
df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df.head()

df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})

#############################################
# Date Değişkenleri Üretmek
#############################################

dff = pd.read_csv("C:/Users/Tugce.Dogan/Downloads/course_reviews.csv")
dff.head()
dff.info()

dff['Timestamp'] = pd.to_datetime(dff["Timestamp"])

# year
dff['year'] = dff['Timestamp'].dt.year

# month
dff['month'] = dff['Timestamp'].dt.month

# year diff
dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year

# month diff (iki tarih arasındaki ay farkı): yıl farkı + ay farkı
dff['month_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month


# day name
dff['day_name'] = dff['Timestamp'].dt.day_name()

dff.head()

# date

#############################################
# 8. INTERACTIONS
#############################################

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1
df.head()

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df[["NEW_SEX_CAT", "Survived"]].groupby(["NEW_SEX_CAT"]).agg({"Survived": ["count", "mean"]})

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

#############################################
# 9. END TO END APPLICATION
#############################################

#############################################
# TITANIC
#############################################

df = load()
df.head()
df.shape
df.info()

#############################################
# 1. FEATURE ENGINEERING
#############################################

df["NEW_CABIN_BOOL"] = df["Cabin"].isnull().astype('int')
df["NEW_NAME_COUNT"] = df["Name"].str.len()
df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr.")]))
df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1
df["NEW_Age_PCLASS"] = df["Age"] * df["Pclass"]

df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

df.loc[(df['Age'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['Age'] >= 18) & (df['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'

#df[["NEW_AGE_CAT", "Survived"]].groupby(["NEW_AGE_CAT"]).agg({"Survived": ["count", "mean"]})

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.shape

df.columns = [col.upper() for col in df.columns]

#############################################
# 2. AYKIRI GOZLEM
#############################################

num_cols = [col for col in df.columns if len(df[col].unique()) > 20
            and df[col].dtypes != 'O'
            and col not in "PASSENGERID"]


df.head()

def outlier_thresholds(dataframe, col_name):
    quartile1 = dataframe[col_name].quantile(0.15)
    quartile3 = dataframe[col_name].quantile(0.85)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))


def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if low_limit > 0:
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    else:
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)


for col in num_cols:
    print(col, check_outlier(df, col))

#############################################
# 3. EKSIK GOZLEM
#############################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)

df.drop("CABIN", inplace=True, axis=1)
missing_values_table(df)

remove_vars = ["TICKET", "NAME"]
df.drop(remove_vars, inplace=True, axis=1)
df.head()
missing_values_table(df)

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

missing_values_table(df)

df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), ''] = 'youngmale'
df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['NEW_SEX_CAT']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df["SEX"] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

#############################################
# 4. LABEL ENCODING
#############################################
df.head()
df.shape
binary_cols = [col for col in df.columns if len(df[col].unique()) == 2 and df[col].dtypes == 'O']
#binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    df = label_encoder(df, col)

#############################################
# 5. RARE ENCODING
#############################################

def rare_analyser(dataframe, target, rare_perc):
    rare_columns = [col for col in dataframe.columns if dataframe[col].dtypes == 'object'
                    and (dataframe[col].value_counts() / len(dataframe) < rare_perc).any(axis=None)]
    for col in rare_columns:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'object'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

rare_analyser(df, "SURVIVED", 0.05)

df = rare_encoder(df, 0.01)
rare_analyser(df, "SURVIVED", 0.01)

#############################################
# 5. ONE-HOT ENCODING
#############################################

ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]


useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

# df.drop(useless_cols, axis=1, inplace=True)

#############################################
# 6. STANDART SCALER
#############################################

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

#############################################
# 7. MODEL
#############################################

y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED",''], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

#############################################
# Hiç bir işlem yapılmadan elde edilecek skor?
#############################################

dff = load()
dff.dropna(inplace=True)
dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)
y = dff["Survived"]
X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# Yeni ürettiğimiz değişkenler ne alemde?

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)