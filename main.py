# Import library
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Membaca dataset
df = pd.read_excel("upah.xlsx")

# 2. Mengecek missing value
print("\nJumlah missing value setiap kolom:")
print(df.isnull().sum())


num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(exclude=np.number).columns

num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

df[num_cols] = num_imputer.fit_transform(df[num_cols])
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# 4. Deteksi dan penanganan outlier dengan metode IQR
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    batas_bawah = Q1 - 1.5 * IQR
    batas_atas = Q3 + 1.5 * IQR
    outliers = df[(df[col] < batas_bawah) | (df[col] > batas_atas)]
    

    df[col] = np.where(df[col] < batas_bawah, batas_bawah, df[col])
    df[col] = np.where(df[col] > batas_atas, batas_atas, df[col])

# 5. Encoding kolom kategorikal
encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])


scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 7. Hasil transformasi
print("\nDataset setelah cleaning dan transformasi:")
print(df.head())

# 8. Visualisasi hasil transformasi
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Korelasi antar variabel setelah transformasi")
plt.show()

# 9. Simpan hasil transformasi
df.to_csv("data_cuaca_harian_cleaned.csv", index=False)
print("\nDataset hasil transformasi disimpan sebagai 'data_cuaca_harian_cleaned.csv'")