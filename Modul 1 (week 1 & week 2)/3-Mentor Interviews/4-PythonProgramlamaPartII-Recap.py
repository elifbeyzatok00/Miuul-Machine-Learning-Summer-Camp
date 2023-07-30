

#############################################
# Pythpn Programlama Part II
#############################################

#NumPy
#Pandas
#Veri Görselleştirme (Matplotlib & Seaborn)
#Keşifsel Veri Analizi

#############################################
# NumPy
#############################################

################################
# Neden Numpy? (List vs NumPy)
################################


################################
# Numpy Array Ozellikleri
################################

import numpy as np

# ndim: boyut sayısı
# shape: boyut bilgisi
# size: toplam eleman sayısı
# dtype: array veri tipi

x =np.random.randint(10,size=5)
x

x.ndim
x.shape
x.size
x.dtype

y =np.random.randint(10,size=(2,5))
y
y.ndim
y.shape
y.size
y.dtype



################################
# Yeniden Boyutlandırma (reshape)
################################

a = np.random.randint(20,size=12)
a
a.reshape(3,4)


################################
# Index Secimi
################################

x = np.random.randint(20,size=15)
x
x[0]
x[0:12]
x[0]=999
x

y = np.random.randint(30,size=(3,5))
y

y[2,4]
y[0,0]


y[0,0] = 24
y


y[:,1]

y[2,:]
y
y[0:2, 1:3]


################################
# Fancy Index
################################

t = np.arange(0,30,3)
t

t[1]
t[2]
t[3]

index = [1,2,3]
t[index]

################################
# NumPy'da Kosullu İşlemler
################################

k=np.array([1,2,3,4,5,6,7])

# Klasik Döngü İle
a=[]
for i in k:
    if i < 5:
        a.append(i)

a

#NumPy ile:

k<5

k[k<5]

k[k>3]

################################
# Matematiksel İşlemler
################################

t = np.array([1,2,3,4,5])

np.subtract(t,1) # cıkarma işlemi
np.add(t,2) # toplama işlemi
np.mean(t)
np.min(t)
np.max(t)
np.var(t)

#####################################################################################################################

################################
# PANDAS
################################

# Veri Bilimi için Olmazsa Olmaz! :)

# Nedir Pandas'ın Özellikleri ?

################################
# Veri Okuma
################################
import pandas as pd

df = pd.read_csv("dosya_yolu.... /.csv")
df = pd.read_excel("dosya_yolu.... /.xlsx")


################################
# Veriye Hızlı Bir Bakıs
################################

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")

# İlk 5 gözlem:
df.head()

# Son 5 gözlem:
df.tail()

# Boyut bilgisi:
df.shape

# Veri Setinin MetaDataları:
df.info()

# Verinin Betimsel Özellikleri

df.describe().T #sayısal değişkenler


df.isnull().values.any()

df.isnull().sum()

df["age"].head()

df["class"].value_counts()  # kategorik değişkenler


################################
# Pandas'da Secim İşlemleri
################################

import pandas as pd
import seaborn as sns
df=sns.load_dataset("titanic")
df.head()

df.index
df[0:13]
df[0:13:2]

df.drop(0,axis=0).head()
# axis=0 ?
df.head()
df.drop(0,axis=0,inplace=True)

df.drop("survived",axis=1).head()

df=df.drop(1,axis=0)

#kalıcı olarak silmek istersek?

#----------------------------------
import pandas as pd
import seaborn as sns
df=sns.load_dataset("titanic")
df.head()

delete_index=[0,2,4,6]
df.drop(delete_index,axis=0).head()


################################
# Değişkeni Indexe Cevirmek
################################

df["age"].head()
df.age.head()

df.index = df["age"]
df.head()


################################
# Indexi Değişkene Cevirmek
################################

df["age"]= df.index
df.head()

df.drop("age",axis=1,inplace=True)
df.head()

df.reset_index().head()

df.reset_index(inplace=True)
df.head()
# df = df.reset_index()


#################################
# Değişkenler Üzerinden İşlemler
#################################

import pandas as pd
import seaborn as sns

df=sns.load_dataset("titanic")
df.head()

"age" in df

df.age.head()
df["age"].head()

type(df["age"].head())

type(df[["age"]].head())

df[["age", "class"]]

columns = ["age", "class"]
df[columns]



# Yeni Değişken Üretmek:
df["new_age"] = df.age **2
df.head()


##################################
# loc & iloc
##################################
# iloc: integer based selection
# loc: label based selection


import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()

# loc
df.iloc[0:3]
df.loc[0:3]

df.loc[0:3,"age"]
df.iloc[0:3,3]
df.iloc[0:3,2:4]

columns = ["age", "class" , "embarked"]
df.loc[0:3,columns]


#########################
# Koşullu Seçim
#########################

df = sns.load_dataset("titanic")
df.head()

df[df["age"]> 50].head()
df[df["age"]> 50]["age"].count()

df.loc[(df["age"]>50) & (df["sex"] == "male"), ["age","class"]].head()

#yenş değişken olusturulabilir:
df.loc[(df["age"] < 25) & (df["sex"] == "male"), "NEW_COLUMN"] = "GENC_ERKEK"
df.head()


#########################
# Toplulaştırma ve Gruplama
#########################

# count()
# min()
# max()
# mean()
# median()
# std()
# var()
# first()
# last()

import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")

df["age"].mean()
df.groupby("sex")["age"].mean()

# agg : matematiksel işlemler
df.groupby("sex").agg({"age": ["mean","min","max"]})
df.groupby(["sex","class"]).agg({"age": ["mean","min","max"],
                       "survived": ["count"]})



#########################
# Pivot Table
#########################

import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")

df.pivot_table ("survived", "sex","embarked")
# df.pivot_table (value, kırılım(satır),kırılım(sutun))
# value degerleri default mean olarak gelir

df.pivot_table("survived", "sex",["embarked","class"])


# Yeni bir amacım olsun;
# Yaş değişkenimi kategorikleştirmek istiyorum:

# cut ile sınırlar-aralık belirliyorum:
df["new_age"]=pd.cut(df["age"],[0,10,18,25,40,90])
df.head()
df["new_age"]

df.pivot_table("survived", "sex",["new_age","class"])
df.pivot_table("survived",["new_age","class"],"sex")

#######################
# Apply ve Lambda
#######################
df = sns.load_dataset("titanic")

df["age2"]=df.age**2
df.head()

for col in df.columns:
    if "age" in col:
        print((df[col]/10).head())

for col in df.columns:
    if "age" in col:
        df[col]=df[col]/10

df.head()
df.head()

# Lambda ve Apply ile yapmak istersek:

df[["age","age2"]].apply(lambda x:x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x:x/10).head()

# apply: istediğim fonksiyonu,methodu uygulayabilirim.

def standart_scaler(col_name):
    return (col_name-col_name.mean()/col_name.std())

df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()
# df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)
df.head()

############################
# Birleştirme İşlemleri
############################

m = np.random.randint(1,30,size=(5,3))
df1=pd.DataFrame(m, columns=["col1","col2","col3"])
df2=df1+99
df1.head()
df2.head()

pd.concat([df1,df2])

pd.concat([df1,df2],ignore_index=True)

pd.concat([df1,df2],axis= 1, ignore_index=True)

####################################
# Merge ile Birleştirme İşlemleri
####################################

df1= pd.DataFrame({'employees': ['john', 'dennis','mark','maria'],
                   'group': ['accounting','engineering', 'engineering','hr']})

df2= pd.DataFrame({'employees': ['john', 'dennis','mark','maria'],
                   'start_date': [2010,2012,2013,2011]})

df1.head()
df2.head()
pd.merge(df1,df2)
pd.merge(df1,df2,on='employees')


#################################
# VERİ GÖRSELLEŞTİRME
#################################

################################
# MATPLOTLİB
################################

# İki farklı değişkenimiz var:
# Kategorik: sutun grafik,countplot
# Sayısal: hist,boxplot


################################################
# Kategorik Değişkenlerin Görselleştirilmesi
################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind='bar')
plt.show()

################################################
# Sayısal Değişkenlerin Görselleştirilmesi
################################################

# Matplotlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=sns.load_dataset("titanic")
df.head()

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])


#############################
# SEABORN
#############################

################################################
# Kategorik Değişkenlerin Görselleştirilmesi
################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=sns.load_dataset("titanic")
df.head()

sns.countplot(x=df["sex"],data=df)


################################################
# Sayısal Değişkenlerin Görselleştirilmesi
################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=sns.load_dataset("titanic")
df.head()

sns.boxplot(x=df["age"])

df["age"].hist()


###############################################
# GELİŞMİŞ FONKSİYONEL KESIFCI VERİ ANALİZİ
###############################################

# Genel Resim
# Kategorik Değişken Analizi
# Sayısal Değişken Analizi
# Hedef Değişken Analizi
# Korelasyon Analizi

#########################
# 1. Genel Resim
#########################


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=sns.load_dataset("titanic")
df.head()


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### Tail #####################")
    print(dataframe.tail())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df["age"])


######################################
# 2. Kategorik Değişken Analizi
######################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=sns.load_dataset("titanic")
df.head()

df.dtypes

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category","object","bool"]]
cat_cols

num_but_cat = [col for col in df.columns if df[col].nunique()< 10 and df[col].dtypes in ["int","float"]]
num_but_cat

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category","object"]]
cat_but_car

cat_cols = cat_cols + num_but_cat
cat_cols

cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique()

[col for col in df.columns if col not in cat_cols]


# Amacım değişkene dair degerlerin oranına göz atmak.
def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),  #değişkende hangi degerden kacar adet var?
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)})) # deger adetlerini toplam deger sayısına bölümü oran verir.
    print("##########################################")

cat_summary(df,"sex")

for col in cat_cols:
    cat_summary(df,col)


######################################
# 3. Sayısal Değişken Analizi
######################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=sns.load_dataset("titanic")
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category","object","bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique()< 10 and df[col].dtypes in ["int","float"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category","object"]]
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[["age","fare"]].describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T

num_cols = [col for col in df.columns if df[col].dtypes in ["int","float"]]
num_cols = [col for col in num_cols if col not in num_but_cat] #numerik_gorunumlu kategorikler hariç

num_cols

def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99] # hangi ceyreklikleri istiyorum?
    print(dataframe[numerical_col].describe(quantiles).T) # istedigim ceyreklikler bazında describe göz atıyorum.


num_summary(df,"age")

for col in num_cols: # num_cols: grab_col_names fonksiyonundan elde ettigim numerik değişkenlerim.
    num_summary(df, col)



##########################################################################

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
        car_th: int, optional
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

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
    cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in num_but_cat]  # numerik_gorunumlu kategorikler hariç

    print(f"Observations: {dataframe.shape[0]}") # satır
    print(f"Variables: {dataframe.shape[1]}") # sutun
    print(f'cat_cols: {len(cat_cols)}') # categorik degişken sayısı
    print(f'num_cols: {len(num_cols)}') # numerik değişkenler
    print(f'cat_but_car: {len(cat_but_car)}') # categorik fakat kardinal
    print(f'num_but_cat: {len(num_but_cat)}') # numerik görünümlü kategorik

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

##########################################################################


######################################
# 4. Hedef Değişken Analizi
######################################

# Kategorik degişkenlerin target değişkene göre ortalamalarını inceleyelim:
def target_summary_with_cat(dataframe, target, categorical_col):
    print(dataframe.groupby(categorical_col)[target].mean(), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "survived", col)

#############################################################################

# Numerik degişkenlerin target değişkene göre ortalamalarını inceleyelim:
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "survived", col)


############################
# 5. KORELASYON
############################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("/diabetes.csv")
df.head()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category","object","bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique()< 10 and df[col].dtypes in ["int","float"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category","object"]]
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]
num_cols = [col for col in df.columns if df[col].dtypes in ["int","float"]]
num_cols = [col for col in num_cols if col not in num_but_cat] #numerik_gorunumlu kategorikler hariç

num_cols

df[num_cols].corr()