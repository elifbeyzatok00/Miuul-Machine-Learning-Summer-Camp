######################################################
# Sales Prediction with Linear Regression
######################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################

df = pd.read_csv("Modul3(week4&week5&week6)/1-Subjects/datasets/advertising.csv")
df.shape

X = df[["TV"]]
y = df[["sales"]]


##########################
# Model
##########################

reg_model = LinearRegression().fit(X, y)

# y_hat = b + w*x
# y_hat = b + w*TV

# sabit (b - bias)
reg_model.intercept_[0]  # 7.032593549127694
# reg_model.intercept_ -> array([7.03259355]) şeklinde dizi olarak döndürüyor
#bu yüzden reg_model.intercept_[0]  olarak çağırdık

# tv'nin katsayısı (w1)
reg_model.coef_[0][0]
# coef -> coefficient(katsayı)
# reg_model.coef_  array([[0.04753664]]) şeklinde dizi olarak döndürüyor
#bu yüzden reg_model.coef_[0][0]  olarak çağırdık

##########################
# Tahmin
##########################

# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?

reg_model.intercept_[0] + reg_model.coef_[0][0]*150  # 14.163089614080658

# 500 birimlik tv harcaması olsa ne kadar satış olur?

reg_model.intercept_[0] + reg_model.coef_[0][0]*500 # 30.80091376563757

df.describe().T

plt.interactive(False)
# Modelin Görselleştirilmesi
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()


##########################
# Tahmin Başarısı
##########################

# MSE
y_pred = reg_model.predict(X)  # elimizde yeterince tahmin değeri yoktu reg_model.predict(X) ile regresyon modeline sorduk. Bizim için tahmin edilen değişken üretir
mean_squared_error(y, y_pred)  # mean_squared_error a y=gerçek y_pred=tahmin edilen değerleri gönderdik ve MSE hesaplattık
# 10.51
y.mean()
y.std()

# RMSE
np.sqrt(mean_squared_error(y, y_pred))
# 3.24

# MAE
mean_absolute_error(y, y_pred)
# 2.54

# R-KARE
reg_model.score(X, y) # regresyon modeline bağımsız değ ve bağımlı dğ verdik bir score hesapla diyoruz
# r-kare, veri setindeki bağımsız değişkenlerin bağımlı değişkeni açılama yüzdesi
# yani bu modelde TV değişkeninin satış değişkenindeki değişikliği açıklama yüzdesi
'''
 Notes: değişken sayısı arttıkça R kare şişmeye meyillidir. burada düzeltilmiş R kare sayısının da göz önünde bulundurlması gerekir
 konuya istatistiksel, ekonometrik, iktisadi modeller açısından bakmıyoruz.Bundan dolayı bu modellerin katsayılarının anlamlılığı,modellerin anlamlılığı, f-istatistiği, t-istatistikleri
 gibi normallik varsayımı ve diğer bazı varsayımlar gibi varsayımlarla,yani özetle istatistiki çıktılarla ilgilenmiyoruz
 "Burada model anlamlılıkları, katsayı testleri vesaire yapmıyoruz.Konuya makine öğrenmesi açısından yaklaşıyoruz.
 doğrusal bir formda tahmin etme görevi var ve bu tahmini en yüksek başarıyla elde etmeye çalışacağız.
 Dolayısıyla aslında basit doğrusal reyesyon da, çoklu doğrusal regresyon yüksek tahmin başarılı modeller değildir.Ama konunun temellerinde olduğu için bu konularla ilgili bir bilgi ediniyoruz.
 Yani en genelinde bizim gelişmiş regresyon problemleri çözmek için kullanacağımız modeller regresyon modelleri olmayacak.
 Ağaca dayalı regresyon modelleri olacak, doğrusal regresyon modelleri olmayacak
'''
######################################################
# Multiple Linear Regression
######################################################

df = pd.read_csv("Modul3(week4&week5&week6)/1-Subjects/datasets/advertising.csv")
X = df.drop('sales', axis=1)
y = df[["sales"]]


##########################
# Model
##########################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

X_train.shape
X_test.shape
y_train.shape
y_test.shape

reg_model = LinearRegression().fit(X_train, y_train)
'''
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
bu şekilde de kullanılabilir
'''

# sabit (b - bias)
reg_model.intercept_

# coefficients (w - weights)
reg_model.coef_


##########################
# Tahmin
##########################

# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio: 10
# newspaper: 40

# 2.90 - bias değeri
# 0.0468431 , 0.17854434, 0.00258619  - coefficient değerleri

#model denklemini yaz -> mülakat aşaması önemli
# Sales = 2.90  + TV * 0.04 + radio * 0.17 + newspaper * 0.002

2.90794702 + 30 * 0.0468431 + 10 * 0.17854434 + 40 * 0.00258619 # = 6.20213102

# bu işlemi fonksiyonel şekilde yapalım, bir yerlere excel formunda vercez diyelim
yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T  # array([[6.20213102]])

reg_model.predict(yeni_veri)

##########################
# Tahmin Başarısını Değerlendirme
##########################

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 1.73
'''
daha önce RMSE hatamız 3.24'tü şimdi 1.73 çok düştü
yeni değişken eklediğimizde başarı artar. Hata düşer
'''

# TRAIN RKARE
reg_model.score(X_train, y_train)
# 0.89

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 1.41

# Test RKARE
reg_model.score(X_test, y_test)
# 0.89


# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))

# 1.69


# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 1.71




######################################################
# Simple Linear Regression with Gradient Descent from Scratch
######################################################

# Cost function MSE
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse


# update_weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w


# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):  # learning_rate, num_iters hiperparametre, yani kullanıcı olarak biz vericez. diğerleri normal parametre veri setinden model buluyor

    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)


        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))


    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w


df = pd.read_csv("datasets/advertising.csv")

X = df["radio"]
Y = df["sales"]

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 100000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)










