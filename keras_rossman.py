# pip install keras
# pip install tensorflow
# pip install --upgrade numpy scipy

import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 1000)

data_zip = zipfile.ZipFile("rosssman (4).zip")

train = pd.read_csv(data_zip.open("train.csv"), low_memory=False)
store = pd.read_csv(data_zip.open("store.csv"), low_memory=False)

data = train.merge(store, on=["Store"], how="inner")
df = data.copy()


# Date

df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)

df["day"] = df["Date"].dt.day
df["week"] = df["Date"].dt.isocalendar().week
df["month"] = df["Date"].dt.month
df["year"] = df["Date"].dt.year
df["season"] = np.where(df["month"].isin([3, 4, 5]), "Spring",
           np.where(df["month"].isin([6, 7, 8]), "Summer",
              np.where(df["month"].isin([9, 10, 11]), "Autumn",
                np.where(df["month"].isin([12, 1, 2]), "Winter", "None"))))


# grafik
plt.figure(figsize=(15,8))
sns.barplot(x="month", y= "Sales", data= df)
plt.show()


# Feature Engineering
df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].mode()[0])

y = df["Sales"]


num_cols = ['Customers', 'Open', 'Promo', 'SchoolHoliday', 'CompetitionDistance', 'Promo2']
cat_cols = ['DayOfWeek', 'StateHoliday', 'StoreType', 'Assortment', 'week', 'month', 'year', 'season']


def category(df, col):
    le = LabelEncoder()
    le1 = le.fit_transform(df[col]).reshape(-1, 1)
    oh = OneHotEncoder(sparse=False)
    col_name = [col + " " + str(i) for i in le.classes_]
    result = pd.DataFrame(oh.fit_transform(le1), columns=col_name)
    return result

df_new = df[num_cols]

for i in cat_cols:
    k_df = category(df, i)
    df_new = pd.concat([df_new, k_df], axis=1)



# Model

X_train, X_test, y_train, y_test = train_test_split(df_new, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="relu"))

model.compile(optimizer="adam", loss="mae", metrics=["mean_absolute_error"])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
