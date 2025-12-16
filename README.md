# Unsloth ile Qwen2.5-Coder Fine-Tuning

Bu proje, **Unsloth** ve **LoRA** (Low-Rank Adaptation) kullanarak **Qwen2.5-Coder-1.5B-Instruct** modelinin fine-tuning (ince ayar) işlemini göstermektedir. Eğitim süreci, modelin kodlama yeteneklerini geliştirmek amacıyla **Google Colab** üzerinde gerçekleştirilmiştir.

## 🚀 Proje Özeti

Bu projenin amacı, verimli eğitim teknikleri kullanarak hafif ama güçlü bir kodlama modelini fine-tune etmektir. Daha hızlı ve bellek açısından verimli bir eğitim için Unsloth kütüphanesi kullanılmıştır.

*   **Temel Model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct`
*   **Teknik:** Unsloth ile LoRA (Low-Rank Adaptation)
*   **Platform:** Google Colab (T4/L4/A100 GPU önerilir(A100 kullanıldı!))

## 📂 Veri Setleri

Aşağıdaki veri setleri kullanılarak iki farklı eğitim stratejisi izlenmiştir:

1.  **Deep Instruction :** `Naholav/CodeGen-Deep-5K` (`DeepTrain.py` dosyasında kullanıldı)
2.  **Diverse Instruction :** `Naholav/CodeGen-Diverse-5K` (`DiverseTrain.py` dosyasında kullanıldı)

## 🛠️ Google Colab Kurulumu

Bu proje Google Colab için optimize edilmiştir. Kodlar, checkpoint'leri ve modelleri kaydetmek için Google Drive'ı bağlayacak şekilde ayarlanmıştır.

1.  Scriptleri (`DeepTrain.py` veya `DiverseTrain.py`) ya da notebook dosyasını (`eval.ipynb`) Google Colab'da açın.
2.  GPU çalışma zamanının (runtime) seçili olduğundan emin olun.
3.  Kodlar, çıktıları `/content/drive/MyDrive/NLPlora/` dizinine kaydetmek için Google Drive'ınızı otomatik olarak bağlayacaktır.

## 📦 Kurulum

Projeyi yerel ortamınızda çalıştırmak veya ortamı yeniden oluşturmak isterseniz, gerekli bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

*Not: Unsloth, CUDA sürümünüze bağlı olarak özel kurulum adımları gerektirebilir. Detaylar için [Unsloth dokümantasyonuna](https://github.com/unslothai/unsloth) bakabilirsiniz.*

## 💻 Kullanım

### Eğitim (Training)
"Deep" veri seti ile modeli eğitmek için:
```python
python DeepTrain.py
```

"Diverse" veri seti ile modeli eğitmek için:
```python
python DiverseTrain.py
```

### Değerlendirme (Evaluation)
Değerlendirme işlemi `eval.ipynb` dosyası ile yapılır. Bu notebook:
1.  Değerlendirme ortamını (LiveCodeBench) kurar.
2.  Fine-tune edilmiş modelleri Google Drive'dan yükler.
3.  AtCoder gibi platformlar üzerinde benchmark testlerini çalıştırır.

## 📊 Sonuçlar

Detaylı analizler, loss grafikleri ve benchmark karşılaştırmaları **[Proje Raporu (REPORT.md)](REPORT.md)** dosyasında bulunabilir.

## 📁 Dosya Yapısı

*   `DeepTrain.py`: Deep veri seti için eğitim scripti.
*   `DiverseTrain.py`: Diverse veri seti için eğitim scripti.
*   `eval.ipynb`: Modelleri değerlendirmek için Jupyter notebook.
*   `REPORT.md`: Eğitim sonuçlarının ve analizlerin yer aldığı detaylı rapor.
*   `requirements.txt`: Python bağımlılık listesi.
*   `Rapor/`: Raporda kullanılan görselleri ve grafikleri içeren klasör.

---
*Bu proje, bir Fine-Tuning ödevi kapsamında hazırlanmıştır.*
