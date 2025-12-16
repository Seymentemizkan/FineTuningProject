# Proje Raporu ve Grafikler

Bu rapor, **Fine-Tuning Projesi** kapsamında gerçekleştirilen model eğitimi, kayıp (loss) analizleri ve en iyi modelin seçim sürecini detaylandırmaktadır.

## 1. Loss Grafiği ve Analizi

Aşağıdaki grafik, eğitim (train), doğrulama (validation) ve test veri setleri üzerindeki kayıp (loss) değerlerinin değişimini göstermektedir.

![Loss Grafiği](Rapor/20LossGrafik.png)



### Yorumlama
1️⃣ Model öğreniyor mu?

Evet, öğreniyor.

Train loss (özellikle Deep Instruction – Train) adım ilerledikçe istikrarlı biçimde düşüyor
(≈1.30 → ≈0.69).

Bu, modelin eğitim verisi üzerindeki hatayı giderek azalttığını ve öğrenme gerçekleştiğini gösterir.

Diverse Instruction – Train için de:

Train loss genel olarak düşüyor ancak dalgalı ve daha yavaş.

Bu, daha çeşitli veri nedeniyle öğrenmenin daha zor ama daha dengeli olduğunu gösterir.

2️⃣ Validation loss davranışı
Deep Instruction (Val)

Başta düşüyor (≈1.20 → ≈0.95),

Ancak 300–500. adımlar arasında tekrar yükselmeye başlıyor (≈1.00+).

👉 Bu, kritik bir işarettir.

Diverse Instruction (Val)

Validation loss yaklaşık sabit (≈0.94 civarı),

Ne belirgin düşüş ne de yükseliş var.

3️⃣ Overfitting (ezberleme) var mı?

🔴 Deep Instruction için:

Evet, overfitting var.

Kanıtlar:

Train loss düşmeye devam ederken

Validation loss artıyor

Train–Val farkı giderek açılıyor

📌 Yani model:

Eğitim verisini çok iyi öğreniyor

Ancak genelleme yeteneğini kaybediyor

Eğitim verisini ezberlemeye başlıyor

🟢 Diverse Instruction için:

Overfitting yok (veya çok az).

Train ve Validation loss birbirine yakın

Validation loss stabil

Büyük bir ayrışma yok

📌 Bu, modelin:

Daha yavaş ama

Daha iyi genelleyen

Daha sağlam öğrendiğini gösterir

## 2. En İyi Checkpoint Seçimi

Eğitim sürecinde farklı adımlarda kaydedilen modeller (checkpoints), belirli benchmark testlerine tabi tutulmuştur. Aşağıdaki tablo, bu checkpoint'lerin performans karşılaştırmasını ve en iyi modelin nasıl belirlendiğini göstermektedir.

![Benchmark Tablosu](Rapor/Tablo.jpg)

**Sonuç:** Tablodaki metrikler (örneğin doğruluk, loss vb.) dikkate alınarak, en düşük doğrulama kaybına veya en yüksek başarı skoruna sahip olan checkpoint, **en iyi model** olarak seçilmiştir.
