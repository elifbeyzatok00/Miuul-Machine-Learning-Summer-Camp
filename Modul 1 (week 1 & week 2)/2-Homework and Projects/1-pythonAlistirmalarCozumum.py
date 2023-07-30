###############################################
# Python Alıştırmalar
###############################################

###############################################
# GÖREV 1: Veri yapılarının tipleriniz inceleyiniz.
# Type() metodunu kullanınız.
###############################################

x = 8
type(x)

y = 3.2
type(y)

z = 8j + 18
type(z)

a = "Hello World"
type(a)

b = True
type(b)

c = 23 < 22
type(c)

l = [1, 2, 3, 4]
type(l)  # list
'''list
# Sıralıdır
# Kapsayıcı
# Değiştirilebilir
'''

d = {"Name": "Jake",
     "Age": 27,
     "Adress": "Downtown"}
type(d)
'''dict
# Sırasız
# Kapsayıcı
# Değiştirilebilir
# Key değerleri farklı olacak
'''

t = ("Machine Learning", "Data Science")
type(t)
'''tuple
# Sıralı
# Kapsayıcı
# Değiştirilemez
'''

s = {"Python", "Machine Learning", "Data Science"}
type(s)
'''set
# Sırasız + Eşsiz
# Kapsayıcı
# Değiştirilebilir
'''

###############################################
# GÖREV 2: Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. Virgül ve nokta yerine space koyunuz, kelime kelime ayırınız.
# String metodlarını kullanınız.
###############################################

text = "The goal is to turn data into information, and information into insight."
text.upper().replace(",", " ").replace(".", " ").split()

###############################################
# GÖREV 3: Verilen liste için aşağıdaki görevleri yapınız.
###############################################

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

# Adım 1: Verilen listenin eleman sayısına bakın.
len(lst)

# Adım 2: Sıfırıncı ve onuncu index'teki elemanları çağırın.
lst[0]
lst[10]

# Adım 3: Verilen liste üzerinden ["D","A","T","A"] listesi oluşturun.
new_lst = lst[0:4]
new_lst

# Adım 4: Sekizinci index'teki elemanı silin.
lst.pop(8)

# Adım 5: Yeni bir eleman ekleyin.
lst.append(101)
lst

# Adım 6: Sekizinci index'e  "N" elemanını tekrar ekleyin.
lst.insert(8, "N")
lst

###############################################
# GÖREV 4: Verilen sözlük yapısına aşağıdaki adımları uygulayınız.
###############################################

dict = {'Christian': ["America", 18],
        'Daisy': ["England", 12],
        'Antonio': ["Spain", 22],
        'Dante': ["Italy", 25]}

# Adım 1: Key değerlerine erişiniz.
dict.keys()

# Adım 2: Value'lara erişiniz.
dict.values()

# Adım 3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
dict.update({"Daisy": ["England", 13]})
dict
dict.get("Daisy")

# Adım 4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
dict.update({"Ahmet": ["Turkey", 24]})
dict

# Adım 5: Antonio'yu dictionary'den siliniz.
dict.pop("Antonio")
dict

###############################################
# GÖREV 5: Arguman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atıyan ve bu listeleri return eden fonskiyon yazınız.
# Liste elemanlarına tek tek erişmeniz gerekmektedir.
# Her bir elemanın çift veya tek olma durumunu kontrol etmekiçin % yapısını kullanabilirsiniz.
###############################################

l = [2, 13, 18, 93, 22]


def func(list):
    odd = []
    even = []

    for i in list:
        if i % 2 == 0:
            odd.append(i)
        else:
            even.append(i)
    return odd, even


var = func(l)
var

###############################################
# GÖREV 6: Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye giren öğrencilerin isimleri bulunmaktadır.
# Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken son üç öğrenci de tıp fakültesi öğrenci sırasına aittir.
# Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.
###############################################

ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]

for index, ogrenci in enumerate(ogrenciler):
    print(index, ogrenci)

for index, ogrenci in enumerate(ogrenciler[0:3], 1):
    print(f"Mühendislik fakültesi {index} . öğrenci: {ogrenci}")

for index, ogrenci in enumerate(ogrenciler[3:], 1):
    print(f"Tıp fakültesi {index} . öğrenci: {ogrenci}")

###############################################
# GÖREV 7: Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir dersin kodu, kredisi ve kontenjan bilgileri yer almaktadır. Zip kullanarak ders bilgilerini bastırınız.
###############################################

ders_kodu = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3, 4, 2, 4]
kontenjan = [30, 75, 150, 25]

zipped_list = list(zip(ders_kodu, kredi, kontenjan))
zipped_list
# [('CMP1005', 3, 30), ('PSY1001', 4, 75), ('HUK1005', 2, 150), ('SEN2204', 4, 25)]

for element in zipped_list:
    print(f"Kredisi {element[1]} olan {element[0]} kodlu dersin kontenjanı {element[2]} kişidir")

###############################################
# GÖREV 8: Aşağıda 2 adet set verilmiştir.
# Sizden istenilen eğer 1. küme 2. kümeyi kapsiyor ise ortak elemanlarını eğer kapsamıyor ise 2. kümenin 1. kümeden farkını yazdıracak fonksiyonu tanımlamanız beklenmektedir.
# Kapsayıp kapsamadığını kontrol etmek için issuperset() metodunu,farklı ve ortak elemanlar için ise intersection ve difference metodlarını kullanınız.
###############################################

kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

def kume(set1,set2):
    if set1.issuperset(set2):
        print(set1.intersection(set2))
    else:
        print(set2.difference(set1))

kume(kume1,kume2)