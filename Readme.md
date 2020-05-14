# Clustering
Clustering adalah sebuah algoritma yang digunakan untuk melakukan pengelompokkan data. Jadi komputer akan secara otomatis melakukan pengelompokkan dengan algoritma-algoritma tertentu, salah satunya adalah **K-Means**.
Contoh simple dari clustering adalah sebagai berikut:
Kita mempunyai data-data di bawah:

| Nomor | Nama | Umur | Jenis Kelamin | Tipe Olahraga Favorit | Keaktifan Posting di Instagram | Keaktifan di Kelas |
| ----- | ---- | ---- | ------------- | --------------------- | ------------------------------ | ------------------ |
| 1 | Ari | 20 | Laki-Laki | Individual | Aktif | Aktif |
| 2 | Davis | 19 | Laki-Laki | Tim | Aktif | Aktif |
| 3 | Yuviena | 22 | Perempuan | Tim | Aktif | Pasif |
| 4 | Fugo | 19 | Laki-Laki | Individual | Pasif | Pasif |

Coba kita kelompokkan data-data berikut menjadi 2 kelompok untuk menentukan apakah orang tersebut Introvert atau Ekstrovert. Secara naluri dan pengetahuan yang kita punya, kita tidak bisa menentukan seseorang itu Introvert atau Ekstrovert dari **Nomor**, **Nama**, **Umur**, **Jenis Kelamin** (dari data di atas). Tapi, kita bisa menentukannya dari **Tipe Olahraga**, **Keaktifan di Instagram** dan **Keaktifan di Kelas**.

Ketika kita memutuskan untuk mengelompokkan data berdasarkan 3 atribut tersebut, kita akan melakukan kalkulasinya dengan menggunakan metode **K-Means** (silahkan cek [video ini](https://www.youtube.com/watch?v=5o_JQ6AHA0Y) untuk tutorial K-Means, saya lebih mengerti kalkulasinya dari video tersebut). Kita tentukan jumlah clusternya menjadi **2** karena yang kita inginkan hanya **Ektrovert** dan **Introvert**

Dan pada akhirnya, kita akan mengelompokkan data tersebut menjadi:

| Nomor | Nama | ... | Cluster (Kelompok) |
| ----- | ---- | ... | ------------------ |
| 1 | Ari | ... | 1 (meskipun yang ini suka olahraga individual, tapi attribut lainnya lebih mendukung kalau data ini berada pada kelompok 1)|
| 2 | Davis | ... | 1 |
| 3 | Yuviena | ... | 1 |
| 4 | Fugo | ... | 2 |

Ketika kelompoknya sudah ada (1 dan 2 berdasarkan data di atas), maka kita akan melakukan mapping (rename) attribut kelompok dari angka menjadi bahasa yang dimengerti banyak orang. Di sini, kita ganti **kelompok 1 menjadi Ektrovert** dan **kelompok 2 menjadi Introvert**. 

### Accuracy
dengan adanya data training (data cluster di atas), kita mendapatkan 2 centroid (titik pusat dari masing-masing cluster), maka kita bisa melakukan testing pada data-data baru. Seandainya ada data baru seperti ini:

```
    RESULT DI SINI ADALAH RESULT DARI DATA YANG SUDAH ADA (YANG BENAR) DAN DIGUNAKAN UNTUK MENGECEK AKURASI
```

| Nama | Umur | Jenis Kelamin | Tipe Olahraga Favorit | Keaktifan Posting di Instagram | Keaktifan di Kelas | Result |
| ---- | ---- | ------------- | --------------------- | ------------------------------ | ------------------ | ------ |
| Defa | 13 | Perempuan | Individual | Pasif | Aktif | Introvert |
| Venn | 23 | Perempuan | Tim | Pasif | Aktif | Introvert |

Komputer akan langsung melakukan kalkulasi dari data tersebut, dan membandingkannya dengan centroid terdekat, maka kira-kira dapatlah **Defa** berada di **kelompok 2** dan **Venn** di **kelompok 1**. Tetapi data asli mengatakan kalau Venn adalah seorang **Introvert**. Maka terdapat kesalahan pada data yang sudah kita train sebelumnya. Dan dapatlah kalau akurasinya adalah 50% (1 benar 1 salah).

Nah, seandainya kita memilih atribut yang lain, kita katakan nama.. Mungkin saja akurasinya 0%.

Jadi di sini kita melakukan tahap:
- Import
- Preprocess (kalau ada data yang tidak valid, maka harus dibuang, dsb.)
- Training Data (dengan K-Means pada umumnya)
- Testing Data (untuk mendapatkan akurasi)

```
    NOTES: SETIAP DATA SEPERTI AKTIF, PASIF, DSB. HARUS DI CONVERT MENJADI ANGKA, KARENA K-MEANS MELAKUKAN KALKULASI SECARA MATEMATIKA
```

# Clustering dengan PySpark

## Import dan Preprocess Data Training

```
    # Import Data Training
    df_train = spark.read.option("inferSchema", "true").csv("Training.csv", header=True)

    # Memilih Atribut yang dipakai
    df_train = df_train.select("Algae Concentration", "Oil Concentration", "Trash Pollution")

    # Preprocess: membuang data training yang kosong (NA)
    df_train = df_train.na.drop()

    # Mengconvert Data Trash Pollution menjadi Angka (Mapping)
    df_train = df_train.withColumn("Trash Pollution", when(df_train["Trash Pollution"] == "Low", 0).
                                                    when(df_train["Trash Pollution"] == "Medium", 1).
                                                    when(df_train["Trash Pollution"] == "High", 2))
```

## Normalize Data Training
Normalization sangat penting dalam clustering untuk mengeliminasi data redundan dan meningkatkan efisiensi dalam algoritma Clustering.
```
    # Vector Assembler berguna untuk menggabungkan beberapa attribut menjadi 1 object yang.
    cols = df_train.columns
    df_train = VectorAssembler(inputCols = cols, outputCol = "Vector").transform(df_train)

    # Standard Scaler berguna untuk menormalize data yang sudah di assemble oleh Vector Assembler
    scaler = StandardScaler(inputCol = "Vector", outputCol = "features")
    df_train = scaler.fit(df_train).transform(df_train)
```

## Mengulangi Kedua Tahap di Atas, namun Ganti dengan Data Testing
```
    df_test = spark.read.option("inferSchema", "true").csv("Testing.csv", header=True)
    df_test = df_test.select("Algae Concentration", "Oil Concentration", "Trash Pollution", "Polluted")
    df_test = df_test.na.drop()
    df_test = df_test.withColumn("Trash Pollution", when(df_test["Trash Pollution"] == "Low", 0).
                                                    when(df_test["Trash Pollution"] == "Medium", 1).
                                                    when(df_test["Trash Pollution"] == "High", 2))

    # Kita juga harus melakukan mapping pada attribut yang berguna sebagai label (hasil data asli)
    df_test = df_test.withColumn("Polluted", when(df_test["Polluted"] == "No", 0).
                                            when(df_test["Polluted"] == "Yes", 1))
    cols = df_test.columns

    # Kita harus mengexclude atribut Polluted agar atribut tersebut tidak kena normalize 
    cols.remove("Polluted")
    df_test = VectorAssembler(inputCols = cols, outputCol = "Vector").transform(df_test)

    scaler = StandardScaler(inputCol = "Vector", outputCol = "features")
    df_test = scaler.fit(df_test).transform(df_test)
```

## Melakukan Testing

Setelah selesai mengimport dan preprocess data training dan testing, selanjutnya akan melakukan training data untuk mendapatkan model yang berguna untuk 

```
    # Karena kita hanya mau prediksi apakah air ini berpolusi atau tidak, 
    # maka itu sama saja kita akan mengelompokkan datanya menjadi 2 kelompok
    # dan kita set K = 2
    kmeans = KMeans().setK(2)
    model = kmeans.fit(df_train)
    
    #Testing
    predictions = model.transform(df_test)
```


## Visualisasi

Selanjutnya, kita akan menggunakan mengvisualisasikan hasil cluster kita, di sini kita menggunakan graphik berupa **scatter**.

```
    # Convert hasil prediction menjadi sebuah tabel (Panda)
    predictions = predictions.toPandas()

    # Inisialisasi Graph
    fig = plt.figure()

    # add_subplot(111) berarti 1x1 di plot 1, silahkan googling untuk detailnya
    ax = fig.add_subplot(111)

    # menentukan X, Y dan warna dari setiap kelompok
    plt.scatter(predictions["Algae Concentration"], predictions["Oil Concentration"], c=predictions["prediction"])

    # Membuat judul dan keterangan X dan Y
    ax.set_title('Relationship Between Algae Concentration and Oil Concentration in Cluster Prediction')
    ax.set_xlabel('Algae Concentration')
    ax.set_ylabel('Oil Concentration')

    # Menampilkan Data
    plt.show()
```

## Perhitungan Akurasi

Akurasi di dapat dari rumus
```
    Akurasi = (total benar) / (total data) * 100
```

Kodingannya:

```
    predictions = predictions.toPandas()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(predictions["Algae Concentration"], predictions["Oil Concentration"], c=predictions["prediction"])
    ax.set_title('Relationship Between Algae Concentration and Oil Concentration in Cluster Prediction')
    ax.set_xlabel('Algae Concentration')
    ax.set_ylabel('Oil Concentration')
    plt.show()
```