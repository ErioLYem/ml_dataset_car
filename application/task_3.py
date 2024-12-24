import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('./used_car_dataset.csv')
df.head()

df.info()

df.isnull().sum()

df['AdditionInfo'].tail()

df['PostedDate'].unique()

df = df.drop('PostedDate', axis=1)
df = df.drop('AdditionInfo', axis=1)

df.head()

df.shape

df.info()

df.isnull().sum()

df['kmDriven'].head()

# Clean and convert 'kmDriven'
df['kmDriven'] = df['kmDriven'].str.replace(r'[^\d.]', '', regex=True).astype('float64')

print(f"Минимальное значение километров:{df['kmDriven'].min()}\nМаксимальное значение киллометров:{df['kmDriven'].max()}")

df['kmDriven'].fillna(df['kmDriven'].median(), inplace=True)
df.isnull().sum()

"""Бренды"""

df['Brand'].unique()

df['Brand'].nunique()

df['Brand'].value_counts()

top_brands = df['Brand'].sort_values(ascending=True).value_counts().nlargest(20).index


plt.figure(figsize=(10, 6))
sns.countplot(data=df[df['Brand'].isin(top_brands)], x='Brand', palette='viridis')
plt.title('Топ 20 брендов', fontsize=18)
plt.xlabel('Бренд', fontsize=15)
plt.ylabel('Количество', fontsize=15)
plt.xticks(rotation=90, ha='right', fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.show()

"""Модели машин"""

df['model'].nunique()

df['model'].value_counts()

top_models = df['model'].value_counts().nlargest(15).index

plt.figure(figsize=(10, 6))
sns.countplot(data=df[df['model'].isin(top_models)], x='model', palette='viridis')
plt.title('Топ 15 моделей', fontsize=18)
plt.xlabel('Модель', fontsize=15)
plt.ylabel('Количество', fontsize=15)
plt.xticks(rotation=60, ha='right', fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.show()

"""Трансмиссия"""

df['Transmission'].unique()

df['Transmission'].value_counts()

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Transmission', palette='viridis')
plt.title('Зависимость количества от типа трансмиссии', fontsize=18)
plt.xlabel('Тип трансмиссии', fontsize=15)
plt.ylabel('Количество', fontsize=15)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.show()

"""Владельцы"""

df['Owner'].unique()

df['Owner'].value_counts()

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Owner', palette='viridis')
plt.title('Зависимость количества от первичных и вторичных владельцев', fontsize=18)
plt.xlabel('Владелец', fontsize=15)
plt.ylabel('Количество', fontsize=15)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.show()

"""Тип топлива"""

df['FuelType'].unique()

df['FuelType'].value_counts()

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='FuelType', palette='viridis')
plt.title('Зависимость количества от вида топлива', fontsize=18)
plt.xlabel('Вид топлива', fontsize=15)
plt.ylabel('Количество', fontsize=15)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.show()

df['AskPrice'].head()

# Clean and convert 'AskPrice'
df['AskPrice'] = df['AskPrice'].str.replace(r'[^\d]', '', regex=True).astype('float64')

numeric_summary = df.describe()

# Frequency counts for categorical columns
categorical_columns = ['Brand', 'model', 'Transmission', 'Owner', 'FuelType']
categorical_summary = {col: df[col].value_counts() for col in categorical_columns}

print("Numeric Summary:\n", numeric_summary)
print("\nCategorical Summary:")
for col, summary in categorical_summary.items():
    print(f"\n{col}:\n{summary}")

sns.scatterplot(x=df['kmDriven'], y=df['AskPrice'], hue=df['FuelType'])
plt.title('Соотношение цены и пробега в зависимости от типа топлива')
plt.xlabel('Километры')
plt.ylabel('Цена')
plt.legend(title='Тип топлива')
plt.show()

categorical_columns = ['Brand', 'model', 'Transmission', 'Owner', 'FuelType']
leb_enc = LabelEncoder()
for col in categorical_columns:
    df[col] = leb_enc.fit_transform(df[col])

print(df[categorical_columns].head())

df.head()

"""Машиное обучение:"""

df.head()

df.info()

scaler = StandardScaler()
numerical_columns = ['Age', 'kmDriven','AskPrice']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

print(df[numerical_columns].head())

# Установка KNN: разделение данных
X_knn = df.drop('AskPrice', axis=1)
y_knn = df['AskPrice']

# Разделение тренировочных и тестируемых данных для KNN модели
X_knn_train, X_knn_test, y_knn_train, y_knn_test = train_test_split(X_knn, y_knn, test_size=0.3, random_state=42)

# Тренировочные KNN регрессии
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_knn_train, y_knn_train)

# Предсказания
y_knn_pred = knn_model.predict(X_knn_test)

# Оценка KNN модели
knn_mse = mean_squared_error(y_knn_test, y_knn_pred)
knn_r2 = r2_score(y_knn_test, y_knn_pred)

print(f"KNN Среднеквадратичная ошибка: {knn_mse:.2f}")
print(f"KNN Точность: {knn_r2:.2f}")

# Установка Байеса: разделение данных по категориям
X_nb = df.drop('AskPrice', axis=1)
price_bins = pd.cut(df['AskPrice'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
y_nb = price_bins

# Разделение тренировочных и тестируемых данных для модели Байеса 70% тренировочных и 30% тестируемых
X_nb_train, X_nb_test, y_nb_train, y_nb_test = train_test_split(X_nb, y_nb, test_size=0.3, random_state=42)

# Классификация тренировачных данных Байеса
nb_model = GaussianNB()
nb_model.fit(X_nb_train, y_nb_train)

# Предсказания
y_nb_pred = nb_model.predict(X_nb_test)

# Оценка модели Байеса
nb_accuracy = accuracy_score(y_nb_test, y_nb_pred)
nb_classification_report = classification_report(y_nb_test, y_nb_pred)

print(f"Точность модели Байеса: {nb_accuracy:.2f}")
print("Отчет:\n", nb_classification_report)

plt.scatter(y_knn_test, y_knn_pred, alpha=0.5)
plt.title('KNN: Прогнозируемые и фактические цены')
plt.xlabel('Фактическая цена')
plt.ylabel('Прогнозируемая цена')
plt.plot([min(y_knn_test), max(y_knn_test)], [min(y_knn_test), max(y_knn_test)], color='k')
plt.show()

ConfusionMatrixDisplay.from_predictions(y_nb_test, y_nb_pred, cmap='Oranges')
plt.title('Матрица путаницы модели Байеса')
plt.show()