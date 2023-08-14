################################################
# End-to-End Diabetes Machine Learning Pipeline III
################################################
'''
Model development sürecinden sonra bir modeli canlı sistemlere entegre etmek demek
o model nesnesini çağırmak demek. Bunu kapsamlı şekilde ele aldığımız diğer programlar var.
Fakat o noktalarda da, kendi kişisel çalışmalarınızda da çok sinsi bir problem var.
Sizi çok düşündürüp "Bu alan bana göre değil mi acaba?" dedirten

Diyelim ki çalışma bitti,bunu birisiyle paylaştık. Bakın çalışma ortamını baştan başlattım şu anda.
Neticesinde bu model nesnesini okuyup kullanabiliyoruz. "Okumaya çalışalım bakalım"
diye düşünüyorum.
'''

import joblib
import pandas as pd

df = pd.read_csv("datasets/diabetes.csv")  #veri setini çağır
# sonra veri setine yeni veriler eklenmiş örn model hastane sisteminde çalıştırılacak

random_user = df.sample(1, random_state=45)  # random veri seç -> bunu yeni veri, hasta gibi ele alalım

new_model = joblib.load("voting_clf.pkl")  #kayıtlı modeli çağırdık

new_model.predict(random_user)   # modeli çalıştırdık

''' ama veri setine yeni veri eklendiği için hata alıcaz. çünkü eski model yeni veriyi tanımıyor
bir boyut hatası verdi.Boyutların uyuşmadığına dair bir bilgi verdi.
yeni bir kullanıcı geldi, yeni bir hasta geldi.İyi de o hastanın bilgileri artık sizin eski yapınızla uyumlu değil ki,
bunlar farklı değişkenler,sizin modeli kurduğunuz verideki değişkenler farklı değişkenler.

bu veri setinin eski veri setiyle aynı olması dönüştürülmesi lazım.
Model deployment süreçlerindeki önemli bariyerlerden, problemlerden birisi budur.
Bu veri setini benzer şekilde diabetes_data_prep() modülümüzden geçirmemiz lazım.
Fonksiyonumuzdan geçirmemiz lazım.

from diabetes_pipeline import diabetes_data_prep() dersem benim bu Script imin içerisinden sadece bu fonksiyonu, modülü, Script i getiriyor olacak.
sonraki adımlarda da yeni gelen veriyi daha önce yaptığımız aynı adımlardan geçiriyor olacağız
daha sonra yeni oluşan modele yeni gelen gözlem birimini terkrar soracağım
'''

from diabetes_pipeline import diabetes_data_prep  # diabetes_data_prep i çalışmama import ettim

X, y = diabetes_data_prep(df)

random_user = X.sample(1, random_state=50)

new_model = joblib.load("voting_clf.pkl")

new_model.predict(random_user)
