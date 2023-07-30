import seaborn as sns
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.

df = sns.load_dataset('titanic')
df.head()
# sns.get_dataset_names()

# Görev 2: Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.

df['sex'].value_counts()

male = df['sex'].value_counts()['male']
male

# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.

df.nunique()

# Görev 4: pclass değişkeninin unique değerlerinin sayısını bulunuz.

df['pclass'] # series
df.pclass # series
df[['pclass']] # dataframe

df['pclass'].nunique()
df.pclass.unique()

# Görev 5: pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.

df[['pclass', "parch"]].nunique()

# Görev 6: embarked değişkeninin tipini kontrol ediniz.
# Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.

# * Az sayıda unique değerden oluşan string tipinde bir değişken var ise
# veri tipini category olarak değiştirmek bellek avantajı ve işlem hızı sağlar.


df['embarked'].dtype

df.info()
df.head()

df['embarked'] = df['embarked'].astype('category')

# Görev 7: embarked değeri C olanların tüm bilgilerini gösteriniz.

df[df['embarked'] == 'C'].head()
df.head()

# Görev 8: embarked değeri S olmayanların tüm bilgilerini gösteriniz.

df[df['embarked'] != 'S']
df[~(df['embarked'] == 'S')]

# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.

df[(df['age'] < 30) & (df['sex'] == 'female')].head()


# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.

df[(df['fare'] > 500) | (df['age'] > 70)].head()

# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.

df.isnull().sum()
df.isna().sum()

df.isnull()

int(True + True)

df.shape[0]

# Görev 12: who değişkenini dataframe’den çıkarınız.

df.head()
df.drop('who', axis=1, inplace=True)


# Görev 13: deck değişkenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.

df['deck'].fillna(df['deck'].mode()[0], inplace=True)

df['deck'].value_counts().index[0]

df.isnull().sum()

# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz.

df['age'] = df['age'].fillna(df['age'].median())

# Görev 15: survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.

df.groupby(['pclass', "sex"]).agg({'survived': ['sum', 'count', 'mean']})
# df.groupby(['pclass', "sex"])['survived'].mean()

# Görev 16: 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazın.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz.
# (apply ve lambda yapılarını kullanınız)

df['age_flag'] = df['age'].apply(lambda x: 1 if x < 30 else 0)
df.head()

df[df.embark_town.isnull()]

# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.

df = sns.load_dataset('tips')
df.head()

# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.

df.groupby('time').agg({'total_bill': ['sum', 'min', 'max', 'mean', 'count']})

# df.groupby("time")["total_bill"].agg(["sum","min","max","mean"])

# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.


df.groupby(['day','time']).agg({'total_bill': ['sum', 'min', 'max', 'mean', 'count']})


# Görev 20: Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.

df_female_lunch = df[(df['time'] == 'Lunch') & (df['sex'] == 'Female')]
df_female_lunch.groupby('day').agg({'total_bill': ['sum', 'min', 'max', 'mean', 'count'],
                                    'tip': ['sum', 'min', 'max', 'mean', 'count']})


# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)

df.loc[(df['size'] < 3) & (df['total_bill'] > 10), 'total_bill'].mean()


# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturunuz.
# Her bir müşterinin ödediği totalbill ve tip in toplamını versin.

df['total_bill_tip_sum'] = df['tip'] + df['total_bill']
df.head()

# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız
# ve ilk 30 kişiyi yeni bir dataframe'e atayınız

df_top30 = df.sort_values(by='total_bill_tip_sum', ascending=False).head(30)
df_top30

# df.sort_values(by='total_bill_tip_sum', ascending=False).iloc[:30]