# Kalp Krizi Risk Tahmini Projesi

Bu projede, California Üniversitesi'nden alınan sağlık verisi kullanılarak, kullanıcıların kalp krizi riskini tahmin eden bir model geliştirilmiştir. Model, PyTorch Geometric ve Neo4j teknolojileri kullanılarak oluşturulmuş ve test edilmiştir.

## İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Kullanılan Teknolojiler](#kullanılan-teknolojiler)
- [Kurulum](#kurulum)
- [Veri Seti](#veri-seti)
- [Model Eğitimi](#model-eğitimi)
- [Aykırı Değerlerin Düzeltilmesi](#aykırı-değerlerin-düzeltilmesi)
- [F1 Skoru Hesaplama](#f1-skoru-hesaplama)
- [Sonuçların Değerlendirilmesi](#sonuçların-değerlendirilmesi)
- [Katkıda Bulunanlar](#katkıda-bulunanlar)
- [Lisans](#lisans)

## Proje Hakkında

Bu proje, kullanıcıların sağlık verilerini analiz ederek kalp krizi risklerini tahmin eden bir sistem geliştirmeyi amaçlamaktadır. Kalp hastalıkları, dünya genelinde en yaygın ölüm nedenlerinden biridir ve erken teşhis, tedavi süreçlerini büyük ölçüde iyileştirebilir. Bu projede, tıbbi verileri analiz ederek bireylerin kalp krizi riskini tahmin etmek için gelişmiş makine öğrenimi ve grafik veritabanı teknolojileri kullanılmıştır.

Grafik Sinir Ağı (GNN) kullanılarak geliştirilen model, sağlık verilerini analiz ederek ve düğümler arasındaki ilişkileri öğrenerek tahminler yapmaktadır. GNN teknolojisi, düğümler ve kenarlar arasındaki ilişkileri öğrenme yeteneği sayesinde, karmaşık veri yapılarını etkili bir şekilde modelleyebilir. Bu projede, GNN modeli, kullanıcıların sağlık verilerini analiz ederek kalp krizi riskini yüksek doğrulukla tahmin etmektedir.

Veri yönetimi ve ilişkisel analizler için Neo4j graf veritabanı kullanılmıştır. Neo4j, büyük veri kümelerindeki ilişkileri kolayca yönetme ve analiz etme imkanı sağlar. Bu projede, her hasta bir düğüm olarak temsil edilmiş ve benzer sağlık özelliklerine sahip hastalar arasında ilişkiler oluşturulmuştur. Bu sayede, model benzer hastalar arasındaki ilişkileri öğrenerek daha doğru tahminler yapabilmiştir.

Proje, veri toplama ve temizleme, model eğitimi ve optimizasyon, aykırı değerlerin düzeltilmesi, performans değerlendirmesi ve sonuçların görselleştirilmesi gibi çeşitli aşamalardan oluşmaktadır. Bu süreçlerin her biri, modelin doğruluğunu ve güvenilirliğini artırmak için titizlikle uygulanmıştır.

## Kullanılan Teknolojiler

- Python
- PyTorch Geometric
- Neo4j
- Pandas
- Scikit-learn

## Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1. Depoyu klonlayın:
   ```bash
   git clone https://github.com/kayabahri/heart-attack-risk-analysis
   cd heart-attack-risk-analysis


pip install -r requirements.txt


Neo4j veritabanını başlatın ve bağlantı bilgilerini güncelleyin:
uri = "neo4j+s://<YOUR_NEO4J_URI>"
username = "neo4j"
password = "<YOUR_NEO4J_PASSWORD>"



Tabii, "Proje Hakkında" kısmını daha detaylı hale getirebiliriz:

markdown
Kodu kopyala
# Kalp Krizi Risk Tahmini Projesi

Bu projede, California Üniversitesi'nden alınan sağlık verisi kullanılarak, kullanıcıların kalp krizi riskini tahmin eden bir model geliştirilmiştir. Model, PyTorch Geometric ve Neo4j teknolojileri kullanılarak oluşturulmuş ve test edilmiştir.

## İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Kullanılan Teknolojiler](#kullanılan-teknolojiler)
- [Kurulum](#kurulum)
- [Veri Seti](#veri-seti)
- [Model Eğitimi](#model-eğitimi)
- [Aykırı Değerlerin Düzeltilmesi](#aykırı-değerlerin-düzeltilmesi)
- [Modelin Değerlendirilmesi](#Graf-sonuçlarını-ile-model-optimizasyonu)
- [Sonuçların Değerlendirilmesi](#sonuçların-değerlendirilmesi)

## Proje Hakkında

Bu proje, kullanıcıların sağlık verilerini analiz ederek kalp krizi risklerini tahmin eden bir sistem geliştirmeyi amaçlamaktadır. Kalp hastalıkları, dünya genelinde en yaygın ölüm nedenlerinden biridir ve erken teşhis, tedavi süreçlerini büyük ölçüde iyileştirebilir. Bu projede, tıbbi verileri analiz ederek bireylerin kalp krizi riskini tahmin etmek için gelişmiş makine öğrenimi ve grafik veritabanı teknolojileri kullanılmıştır.

Grafik Sinir Ağı (GNN) kullanılarak geliştirilen model, sağlık verilerini analiz ederek ve düğümler arasındaki ilişkileri öğrenerek tahminler yapmaktadır. GNN teknolojisi, düğümler ve kenarlar arasındaki ilişkileri öğrenme yeteneği sayesinde, karmaşık veri yapılarını etkili bir şekilde modelleyebilir. Bu projede, GNN modeli, kullanıcıların sağlık verilerini analiz ederek kalp krizi riskini yüksek doğrulukla tahmin etmektedir.

Veri yönetimi ve ilişkisel analizler için Neo4j graf veritabanı kullanılmıştır. Neo4j, büyük veri kümelerindeki ilişkileri kolayca yönetme ve analiz etme imkanı sağlar. Bu projede, her hasta bir düğüm olarak temsil edilmiş ve benzer sağlık özelliklerine sahip hastalar arasında ilişkiler oluşturulmuştur. Bu sayede, model benzer hastalar arasındaki ilişkileri öğrenerek daha doğru tahminler yapabilmiştir.

Proje, veri toplama ve temizleme, model eğitimi ve optimizasyon, aykırı değerlerin düzeltilmesi, performans değerlendirmesi ve sonuçların görselleştirilmesi gibi çeşitli aşamalardan oluşmaktadır. Bu süreçlerin her biri, modelin doğruluğunu ve güvenilirliğini artırmak için titizlikle uygulanmıştır.

## Kullanılan Teknolojiler

- Python
- PyTorch Geometric
- Neo4j
- Pandas
- Scikit-learn

## Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1. Depoyu klonlayın:
   ```bash
   git clone https://github.com/kullaniciadi/proje-adi.git
   cd proje-adi
Gerekli paketleri yükleyin:

bash
Kodu kopyala
pip install -r requirements.txt
Neo4j veritabanını başlatın ve bağlantı bilgilerini güncelleyin:

python
Kodu kopyala
uri = "neo4j+s://<YOUR_NEO4J_URI>"
username = "neo4j"
password = "<YOUR_NEO4J_PASSWORD>"
Veri Seti
Veri seti, California Üniversitesi'nden alınan sağlık verilerini içermektedir. Bu veriler, yaş, cinsiyet, kan basıncı, kolesterol seviyeleri gibi çeşitli sağlık ölçümlerini içerir. Veri seti corrected_heart_data.csv dosyasında bulunabilir.

Model Eğitimi
Model eğitimi için aşağıdaki adımlar izlenmiştir:

Verilerin hazırlanması ve temizlenmesi.
Aykırı değerlerin tespit edilmesi ve düzeltilmesi.
Neo4j veritabanına verilerin yüklenmesi.
GNN modelinin oluşturulması ve eğitilmesi.
Model performansının doğruluk, kayıp, ROC eğrisi ve AUC skoru ile değerlendirilmesi.
Aykırı Değerlerin Düzeltilmesi
Aykırı değerler, Z-score yöntemi ile tespit edilmiştir. Tespit edilen aykırı değerler medyan ile değiştirilmiştir.

F1 Skoru Hesaplama
Modelin performansını değerlendirmek için F1 skoru hesaplanmıştır. F1 skoru, doğruluk ve duyarlılık arasında bir denge kurarak modelin başarısını ölçer.

Sonuçların Değerlendirilmesi
Modelin eğitim sürecinde doğruluk %90'a ulaşmış, test sürecinde ise %85 doğruluk elde edilmiştir. Modelin ROC eğrisi ve AUC skoru hesaplanmış, AUC skoru 0.92 olarak bulunmuştur. Kullanıcıların kalp krizi riski, eğitim ve test süreçlerinde elde edilen sonuçlarla değerlendirilmiştir.