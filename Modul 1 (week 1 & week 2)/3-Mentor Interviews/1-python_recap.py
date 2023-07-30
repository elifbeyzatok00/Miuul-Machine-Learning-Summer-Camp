
################################## Machine Learning Summer Camp #########################################

# Python Alıştırmalar Recap #


# Working Settings
# Pycharm
# Virtual Environment
# Package Management

# Data Structures


# Fonksiyonlar
# Docstring
# Fonksiyon Okuryazarlığı
# Return#

#
# Koşullar-Döngüler
# If
# Else & If
# For Döngüsü
# Break & While & Continue
# Enumurate
# Zip
# Lambda & Filter & Reduce
#
#
# Comprehensions
# List Comprehension
# Alıştırmalar



##################################################################################
# WORKING ENVIRONMENT SETTINGS - ANACONDA- PYCHARM
##################################################################################


# Anaconda nedir?
"""
Anaconda, Python tabanlı bir veri bilimi platformudur ve birçok popüler Python paketini ve aracını içerir.
Paket yöneticisi olan conda ve kullanıcı dostu bir grafiksel kullanıcı arayüzü olan Anaconda Navigator ile birlikte,
veri bilimcileri ve geliştiricileri için güçlü bir geliştirme ve analiz ortamı sağlar.
"""

# Pycharm nedir?
"""
IDE(Integrated Development Environment)
Python programlama dili için özel olarak tasarlanmış bir entegre geliştirme ortamıdır (IDE). PyCharm, Python projelerini oluşturmanız, düzenlemeniz, hata ayıklamanız, test etmeniz ve dağıtmanız için bir dizi araç ve özellik sunar.
"""

# Virtual Environment neden oluşturuyoruz?
"""
İzole bir çalışma ortamı oluşturmak için "sanal ortam" oluşturmaktayız.
"""

"""
Sanal ortamlar birbirinden farklı kütüphane ve versiyonlar içerisinde projeler birbirini etkilemeden çalışma imkanı
sağlamaktadır.
"""

# Virtual ortam(Sanal ortam) araçları nelerdir?
"""
venv, virtualenv, pipenv, conda
"""


# Package Management(paket yönetim) toollarına neden ihtiyacımız bulunmaktadır?
"""
Tools that manage the dependency management work of libraries/packages.
"""


# Package management tools? Conda ve pip arasındaki ilişki nedir?
"""
pip, pipenv, conda
"""

"""
Conda, hem paket yönetimi hem de sanal ortam yönetimi yapmaktadır.
pip, paket yönetimi yapmaktadır.
"""


"""
venv ve virtualenv paket yönetim aracı olarak pip kullanır.

conda ve pipenv hem paket yönetimi hem virtual env yönetimi yapabilmektedir.

"""


# Virtual environment ve package management
"""
Create New Environment
conda create -n mynenviroment

List environment

"""



##################################################################################
# DATA STRUCTURES ( VERI YAPILARI)
##################################################################################

# Veri yapıları nelerdir?

"""
# Sayılar
# Strings
# List
# Dictionary
# Tuple
# Set
"""


# SAYILAR
# int
x = 10
type(x)

# float
y = 20.3
type(y)

# complex
z = 3j + 5
type(z)

t = 1j + 10
type(t)


# STRING
x = "DATA SCIENCE"
type(x)

y = "Data\t Science"

# BOOLEAN
True
False
type(False)

10 % 5 == 3 % 3

type(10 % 5 == 3 % 3)

10 / 5 == 6 // 3

# LIST
"""
Değiştirilebilir

Sıralıdır.

Index işlemi yapılabilir.

Kapsayıcıdır.

"""

l= ["Data",1,"python",1.2,"machine learning",2.4]
type(l)

# check methods?
dir(l)

# eleman ekleme
l.append("chatgpt")
print(l)

# l.append(4,"comprehension")

# eleman çıkarma
l.pop(2)
print(l)

# indexleme
l[2] = "chatgpt"
print(l)


## Yorumlama ##
b = ["String",1,2,"Python",(0,1,2)]

b[-2]
b[-6]
b.pop(-1)
print(b)

# DICT
"""
Değiştirilebilir

Sırasızdır

Kapsayıcıdır.

"""

d = {"Captain America: The First Avenger": 2011,
     "Avengers": 2012,
     "Avengers:Ultron Age": 2015,
     "Avengers:Infinity War":2018,
     "Avengers:End Game":2019}

print(d)

d.keys()
d.values()

# Key Sorgulama
"Avengers" in d

# Value Değiştirmek
d["Avengers"] = 2014

# Value Erişmek
d.values()

# Key-Value Değerini Güncellemek
d.update({"Captain America: The First Avenger": 2010})
print(d)

# Son eleman silmek
d.popitem()

dir(d)



# TUPLE
"""
Değiştirilemez

Sıralıdır

Kapsayıcıdır.

"""

t = ("Machine Learning", "Data Science","Data Analyst","Data Engineer")
type(t)


# Indexleme yapılabilir mi?
t[0]= "Machine"


coral = ('blue coral', 'staghorn coral','pillar coral', 'elkhorn coral','black coral')
print(coral)
coral[-4:-2]

# Yukarıdaki tuple içerisinde "black coral" nesnesini siliniz ve tekrardan tuple olarak gösteriniz.

"""
c = list(coral)
c.pop(4)
c = tuple(c)
print(c)

"""



# SET
"""
Değiştirilebilir

Sırasız + Eşsizdir.

Kapsayıcıdır.

"""

s = {"Python", "Machine Learning", "Data Science","Python","Machine Learning"}
type(s)
print(s)

# Indexleme yapılabilir mi?
s[1]



# Yorumlarınızı nedir?

# tuple ---
x = {42, 'foo', (1, 2, 3), 3.14159}

# list
y = {42, 'foo', [1, 2, 3], 3.14159}

# dictionary
z = {1,2, {'a': 1, 'b': 2},5}


#######################
# difference(): İki kümenin farkı
#######################

set1 = set([1, 3, 5])
set2 = set([1, 2, 3])

# set1'de olup set2'de olmayanlar.
set1.difference(set2)
set1 - set2

# set2'de olup set1'de olmayanlar.
set2.difference(set1)
set2 - set1


#######################
# isdisjoint(): İki kümenin kesişimi boş mu?
#######################

set1 = set([7, 8, 9])
set2 = set([5, 6, 7, 8, 9, 10])

set1.isdisjoint(set2)
set2.isdisjoint(set1)

#######################
# intersection(): İki kümenin kesişimi
#######################

set1 = set([1, 3, 5])
set2 = set([1, 2, 3])

set1.intersection(set2)
set2.intersection(set1)

set1 & set2


#######################
# union(): İki kümenin birleşimi
#######################

set1.union(set2)
set2.union(set1)


####################################################################################
#                       DATA STRUCTURE ALIŞTIRMALAR
# ##################################################################################


# 1) Write a line of code that creates a list containing the first 10 Fibonacci numbers.
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

fibonacci = [0,1]

while len(fibonacci)<10:
    fibonacci.append(fibonacci[-1]+fibonacci[-2])
print(fibonacci)


# 2) Write a line of code that counts the number of unique characters in a string.
string = "hello world"
# 8
unique_char = len(set(string))
print(unique_char)

# 3) Write a line of code that finds the second smallest element in a list.
my_list = [5, 3, 1, 4, 2,12,0,-4]
# 0

sorted(set(my_list))[1]

list(set(my_list))[1]



# 4) Write a line of code that creates a tuple containing the squares of numbers from 1 to 5.
#(1, 4, 9, 16, 25)

tuple([x**2 for x in range(1,6)])


# 5) Write a line of code that removes duplicates from a list and converts it into a tuple.
my_list = [1, 2, 3, 2, 4, 5, 3, 1]
# (1, 2, 3, 4, 5)

tuple(set(my_list))

sorted(set(my_list))

##################################################################################
#                              FUNCTIONS
##################################################################################

# Fonksiyon nedir?

"""
Bir fonksiyon, belirli bir işi gerçekleştirmek için tasarlanmış bir kod bloğudur. 
İçine verilen girdileri alır, bu girdileri işler ve belirli bir çıktıyı üretir.

Programlamada fonksiyonlar da belirli bir görevi yerine getirmek için tasarlanmıştır ve bize istediğimiz sonuçları sunar.


"""


# Parametre nedir? Argüman nedir?

"""
Parametre : Tanımlama esnasındaki temsildir.
Argüman : Parametreyi kullandığımızda argüman olur.

"""

"""
# def function_name(parameters/arguments):
#     statements (function body)
"""

def function_name(parameters,arguments):
# statements(function_body)


# Print/Return ?

#######################
# Return: Fonksiyon Çıktılarını Girdi Olarak Kullanmak
######################

"""
Return yaptığımızda igili objeyi dışarıya çıkarabiliyoruz. Diğer durumda fonksiyondan hiçbir bilgiyi dışarı çıkaramayız.
"""

def calculate_area(length, width):
    return length * width

rectangle_area = calculate_area(10, 5)
print(rectangle_area)


def calculate_area(length, width):
    print(length * width)

rectangle_area = calculate_area(10, 5)
print(rectangle_area)




# Rakamlar arasından ortalamayı hesaplamak
def calculate_average(numbers):
    return sum(numbers) / len(numbers)

sayilar = [85, 90, 92, 88, 95]
average_grade = calculate_average(sayilar)
print(average_grade)

#

def yashesapla(dogumyili):
    return 2023-dogumyili


def emeklilikhesap(dogumyili,isim):
    yas = yashesapla(dogumyili)
    emeklilik = 65- yas
    if emeklilik > 0 :
        print(f'emekliliğe {emeklilik} yıl kaldı')
    else:
        print("Emekli oldunuz")


emeklilikhesap(1989,'Gürkan')


# Yaş, sigorta ve çalışma yılına uygun emeklilik hesap fonksiyonu

def emeklilik_yas_hesabi():
    def emeklilik_yasi(calisma_yili):
        if calisma_yili >= 25:
            return 60
        else:
            return 65

    def uygunluk_durumu(age, insurance, calisma_yili):
        if age >= emeklilik_yasi(calisma_yili) and insurance == True:
            return "Emekli olabilirsiniz."
        else:
            return "Emekli olma şartlarınızı sağlamıyorsunuz."

    return uygunluk_durumu


uygunluk_durumu = emeklilik_yas_hesabi()
age = 34
insurance = True
calisma_yili = 11
result = uygunluk_durumu(age, insurance, calisma_yili)
print(result)


# Bir hesap makinesi fonksiyonu kullanarak, toplama ve çıkarma işlemlerini gerçekleştiren bir hesap makinesi uygulaması

def hesap_makinesi():
    def add(a, b):
        return a + b

    def subtract(a, b):
        return a - b

    return add, subtract


toplama, cikarma = hesap_makinesi()
num1 = 10
num2 = 5
toplama_sonucu = toplama(num1, num2)
cikarma_sonucu = cikarma(num1, num2)
print("Toplama Sonucu:", toplama_sonucu)
print("Çıkarma Sonucu:", cikarma_sonucu)


##################################################################################
#                              CONDITIONS
##################################################################################

# IF -ELSE

def sayi_kontrol(sayi):
    if sayi % 2 == 0:
        print(f"{sayi} çift bir sayıdır.")
    else:
        print(f"{sayi} tek bir sayıdır.")

sayi_kontrol(10)

kullanici_sayi = int(input("Bir sayi giriniz:"))
sayi_kontrol(kullanici_sayi)


"""kullanici_sayisi = int(input("Bir sayı girin: "))
sayi_kontrol(kullanici_sayisi)
"""

# For Döngüsü

"""
Python'da for döngüsü, bir iterable (tekrarlanabilir) nesne üzerinde çalışır ve her döngü adımında bir elemana erişir.

for eleman in iterable:

"""


liste = [2, 5, 8, 3, 1, 6]

for sayi in liste:
    print(sayi**2)


# Liste içindeki çift sayıları bulmak
liste = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for sayi in liste:
    if sayi % 2 == 0:
        print(sayi)


# Fonksiyon - For Döngüsü ve If/Else Koşulu
def kontrol_fonksiyonu(sayilar):
    for eleman in sayilar:
        if eleman > 5:
            print(f"{eleman} 5'ten büyüktür.")
        elif eleman < 5:
            print(f"{eleman} 5'ten küçüktür.")
        else:
            print(f"{eleman} 5'e eşittir.")

liste = [2, 5, 8, 3, 1, 6]
kontrol_fonksiyonu(liste)

# Örnekler

for i in range(1,11):
    for j in range(1,11):
        carpim=i*j
        print(f'{i} x {j} = {carpim}')
    print("----")

# Çarpım Tablosu
# 1 x 1 = 1
# 1 x 2 = 2
# 1 x 3 = 3
# 1 x 4 = 4
# 1 x 5 = 5
# 1 x 6 = 6
# 1 x 7 = 7
# 1 x 8 = 8
# 1 x 9 = 9
# 1 x 10 = 10
# --------------------

"""
for i in range(1, 11):
    for j in range(1, 11):
        carpim = i * j
        print(f"{i} x {j} = {carpim}")
    print("--------------------")
"""




# Üçgen Sayılar
# [1,3,6,10,15,21]

for i in range(1,7):
    ucgen_sayi = (i*(i+1))//2
    print(ucgen_sayi)

n = int(input("Kaç terimli üçgen sayısı bulmak istersin?"))


"""
n = int(input("Kaç terimli üçgen sayılarını bulmak istiyorsunuz? "))

for i in range(1, n + 1):
    ucgen = (i * (i + 1)) // 2
    print(ucgen)
"""

# Palindromik Sayılar
# 22
# 33
# 44
# 55
# 66
# 77
# 88
# 99

"""
baslangic = int(input("Başlangıç sayısını girin: "))
bitis = int(input("Bitiş sayısını girin: "))

for sayi in range(baslangic, bitis + 1):
    sayi_str = str(sayi)
    ters_str = sayi_str[::-1]
    if sayi_str == ters_str:
        print(sayi)
"""

sayi_str = str(212)
sayi_str[::-1]

212

basla = int(input("Başlangıç sayısını seçiniz:"))
bitis = int(input("Bitiş sayısını seçiniz"))

for sayi in range(10,250):
    sayi_str = str(sayi)
    ters_str = sayi_str[::-1]
    if sayi_str == ters_str:
        print(sayi)

for sayi in range(basla, bitis):
    sayi_str = str(sayi)
    ters_str = sayi_str[::-1]
    if sayi_str == ters_str:
        print(sayi)


# Break - Continue - While

# break?


# continue
"""
Döngüyü atla
"""

# while

"""
break ifadesi bir döngüyü tamamen sonlandırmak için kullanılırken, 
continue ifadesi bir döngü adımını atlayarak bir sonraki adıma geçmek için kullanılır. 
while döngüsü ise belirli bir koşul sağlandığı sürece döngüyü tekrarlar.

"""

# Sample 1
i = 1
toplam = 0
while i <= 10:
    if i == 6:
        i += 1
        continue
    toplam += i
    i += 1
print("Toplam:", toplam)


# Sample 2
toplam = 0
while True:
    sayi = int(input("Bir sayı girin: "))
    if sayi < 0:
        break
    toplam += sayi
print("Toplam:", toplam)

""""
while döngüsü sonsuz bir döngü olarak başlıyor (while True). Kullanıcıdan sürekli olarak bir sayı girmesini istiyoruz
"""


# Enumerate : Otomatik Index ile for loop

"""
enumerate() fonksiyonu, bir iterable (tekrarlanabilir) nesne üzerinde indeksleriyle birlikte 
döngü yapmayı sağlayan bir Python fonksiyonudur. 

enumerate(iterable, start=0)

"""


players = ["modric","kross","iniesta","xavi"]

for i in players:
    print(i)


for index, person in enumerate(players):
    print(index+1 ,person)



# Sample
kelimeler = input("Kelime listesini girin (virgülle ayırın): ").split(",")

for indeks, kelime in enumerate(kelimeler,1):
    print(indeks, kelime.strip())


#######################
#  Lambda
#######################

# lambda arguments : expression

kare = lambda x: x ** 2
print(kare(4))


carpimlar = []
for i in range(1, 6):
    carpimlar.append((lambda x: x * i)(2))
print(carpimlar)

"""
lambda ifadesi, basit veya tek seferlik kullanımlar için hızlı bir şekilde bir fonksiyon tanımlamak için kullanılır. 
Genellikle, kısa ve basit fonksiyonları belirli bir ifade veya işlem için kullanırken tercih edilir. 
"""


def func(n):
  return lambda a : a ** n

square = func(2)
print(square(7))


"""
map() fonksiyonu ile bu fonksiyonları bir iterable üzerinde uygulayarak yeni bir iterable oluşturuyoruz. 
Benzer şekilde, filter() fonksiyonu ile de belirli bir koşulu sağlayan elemanları bir iterable'dan filtreleyebiliyoruz.

map() fonksiyonu, bir iterasyon yapısı üzerinde belirli bir işlemi uygulayarak yeni bir iterasyon yapısı oluştururken, 
filter() fonksiyonu belirli bir koşulu sağlayan elemanları seçerek yeni bir iterasyon yapısı oluşturur.

"""
# Sample 1
liste1 = [1, 2, 3, 4, 5]
liste2 = [10, 20, 30, 40, 50]

toplam = list(map(lambda x, y: x + y, liste1, liste2))
print(toplam)


# Sample 2
liste = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
tekler = list(filter(lambda x: x % 2 != 0,liste))
print(tekler)



##################################################################################
#                              COMPREHENSION
##################################################################################


# If - Else :

# [ [ifresult] if [condition] else [elseresult] for i in [.....] ]


# If :

# [ [ifresult] for i in [....] if [condition] ]


"""
List comprehension, Python'da kompakt bir şekilde liste oluşturmak için kullanılan bir yapıdır.

"""


# Bir liste içindeki sayıları negatif ise sıfıra eşitleme, pozitif ise kendilerini koruma
liste = [1, -2, 3, -4, 5, -6]
# [1, 0, 3, 0, 5, 0]

[ x if x>0 else 0 for x in liste]

"""
sonuclar = [x if x >= 0 else 0 for x in liste]
print(sonuclar)
"""


# Bir liste içindeki çift sayıları ikiye bölenlerin karesini, tek sayıları ise üçe bölenlerin karesini içeren bir liste oluşturma:
liste = [3, 8, 27, 10, 9, 12]
# [1.0, 64, 81.0, 100, 9.0, 144]

sonuc = [x ** 2 if x % 2 == 0 else (x//3)**2 for x in liste]
print(sonuc)


"""
sonuclar = [x**2 if x % 2 == 0 else (x/3)**2 for x in liste]
print(sonuclar)
"""


# Bir dize içindeki kelimeleri ters çevirmek
dize = "Machine Learning Summer Camp"
# ['enihcaM', 'gninraeL', 'remmuS', 'pmaC']

"""
ters_kelimeler = [kelime[::-1] for kelime in dize.split()]
print(ters_kelimeler)
"""

ters_kelime = [ kelime[::-1] for kelime in dize.split()]
print(ters_kelime)

# Bir cümle içindeki harflerin büyük harf veya küçük harf olmasına göre ilgili şekilde yeni bir liste oluşturma:

sentence = "dAta sCieNce!"
# ['D', 'a', 'T', 'A', ' ', 'S', 'c', 'I', 'E', 'n', 'C', 'E', '!']

sonuc = [harf.upper() if harf.islower() else harf.lower() for harf in sentence]
print(sonuc)

"""
sonuclar = [harf.upper() if harf.islower() else harf.lower() for harf in sentence]
print(sonuclar)
"""

# Sample
positions = ["goalkeeper","defence","midfielder","forward","winger"]
liste=[]
for x in positions:
    if "e" in x:
        liste.append(x)

liste
# Out[110]: ['goalkeeper', 'defence', 'midfielder', 'winger']



# Sample 2
# 100 'e kadar sayılarda 2 ve 3'e tam kalanlı bölünen sayıları listeme

liste = []
for y in range(100):
    if (y%2 == 0) &  (y%3  == 0):
        liste.append(y)

liste

[ y for y in range(100) if (y%2 == 0) &  (y%3  == 0)]

# List Comprehenson
[ y for y in range(100) if (y%2 == 0) &  (y%3  == 0)]


# Sample 3
strings = ["hello", "world", "python", "programming"]
# ['HELLO', 'WORLD', 'PYTHON', 'PROGRAMMING']

uppercase_strings = [word.upper() for word in strings]
print(uppercase_strings)



##################################################################################
#                              LIST COMPREHENSION ALIŞTIRMALAR
##################################################################################


# 1 Write a list comprehension that generates a list of all possible substrings of a given string.
string = "myth"
# ['m', 'my', 'myt', 'myth', 'y', 'yt', 'yth', 't', 'th', 'h']

len(string)

sub_string=[]
for i in range(len(string)):
    for j in range(i+1,len(string)+1):
        sub_string.append(string[i:j])

print(sub_string)

sub = [string[i:j] for i in range(len(string)) for j in range(i+1,len(string)+1) ]
print(sub)


[string[i:j] for i in range(4) for j in range(i+1,5)]


# 2 Write a list comprehension that flattens a nested list into a single list.
nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]

single=[]

for flat in nested_list:
    for x in flat:
        single.append(x)

print(single)

# List Comprehension

[x for flat in nested_list for x in flat]

# 3 Write a list comprehension that generates a list of all possible combinations of two strings from two given lists.
list1 = ['a', 'b']
list2 = ['x', 'y']
# ['ax', 'ay', 'bx', 'by']

[ x + y for x in list1 for y in list2]



# 4# Write a list comprehension that generates a list of prime numbers up to a given number n.
# [2, 3, 5, 7, 11, 13, 17, 19]

n = 20

def is_prime(number):
    if number < 2:
        return False
    for i in range(2,int(number**0.5)+1):
        if number % i == 0:
            return False
    return True

prime_sayılar = []
for number in range(2,n+1):
    if is_prime(number):
        prime_sayılar.append(number)

print(prime_sayılar)


[number for number in range(2,n+1) if is_prime(number)]

#Alternative Solution for List Comprehension
n=20
[number for number in range(2,n+1) if all(number % i != 0 for i in range(2,int(number**0.5)+1))]


# 5#Write a list comprehension that finds all numbers in a given list that are divisible by the sum of their digits.
numbers = [12, 23, 34, 45, 56, 67, 78, 89, 90]
# [12, 45, 90]


result = [num for num in numbers if num % sum(int(digit) for digit in str(num)) == 0]
print(result)


# Extracting even-length words from a list of strings:
strings = ["apple", "banana", "orange", "kiwi", "grape"]
# ['banana', 'orange', 'kiwi']

[word for word in strings if len(word) % 2 == 0]



"""
even_length_words = [word for word in strings if len(word) % 2 == 0]
print(even_length_words)
"""

# 7) A listesindeki elemanları for döngüsü kullanarak B listesine taşımak
A = [20, 35, 48, 50, 23]
B = []
# [20, 35, 48, 50, 23]

for eleman in A:
    B.append(eleman)

A.clear()

print("A listesi:", A)
print("B listesi:", B)


def transform(liste1,liste2):
    for eleman in liste1:
        liste2.append(eleman)

    liste1.clear()
    return liste1,liste2

liste1,liste2 = transform(A,B)


# 8) unique elemanları döndüren fonksiyonu yazınız.

no_unique_list = [1,1,1,2,2,2,3,5,5,5,7,7,7,9,9,9]
# [1, 2, 3, 5, 7, 9]

"""
def no_unique(i):
    liste = list(set(i))

    return liste

no_unique(no_unique_list)

"""



# 9) Bir sayı listesi alıp bu listenin içindeki tüm elemanları toplayan fonksiyonu yazınız.

sampleList = [15,25,40,55,60]
# Liste Elemanlarının Toplamı: 195

def sum(sayi_listesi):
    toplam = 0
    for eleman in sayi_listesi:
        toplam += eleman
    return toplam

sampleList = [15, 25, 40, 55, 60]
toplam = sum(sampleList)
print("Liste Elemanlarının Toplamı:", toplam)



# 10.0 For döngüsü kullanarak faktöriyel hesabını yazınız.


def faktoriyel(n):
    faktoriyeller = [1]
    for i in range(1, n+1):
        faktoriyel = faktoriyeller[-1] * i
        faktoriyeller.append(faktoriyel)
    return faktoriyeller

n = 5
faktoriyeller = faktoriyel(n)
print(faktoriyeller)



# 10.1 For döngüsü kullanarak girilen sayının faktöriyel hesabını yazan fonksiyonu yazınız

def faktoriyel_hesapla(n):
    faktoriyel = 1
    if n < 0:
        return "Negatif sayıların faktöriyeli tanımsızdır."
    elif n == 0:
        return 1
    else:
        for i in range(1, n+1):
            faktoriyel *= i
        return faktoriyel

sayi = int(input("Faktöriyelini hesaplamak istediğiniz sayıyı girin: "))
faktoriyel = faktoriyel_hesapla(sayi)
print("Faktöriyel:", faktoriyel)


#11 players listesinde kelime uzunluğu 6'den küçük olanları getiren listeyi tanımla

players = ["messi","ronaldo","benzema","mbappe","haaland"]

liste= []
for i in players:
    if len(i)< 6:
        liste.append(i)

liste

goat = [i for i in players if len(i)< 6]
print(goat)