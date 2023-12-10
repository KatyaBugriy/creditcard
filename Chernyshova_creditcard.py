#!/usr/bin/env python
# coding: utf-8

# Набор данных содержит транзакции, совершенные европейскими держателями карт по кредитным картам в сентябре 2013 года.
# В этом наборе данных представлены транзакции, произошедшие за два дня, где у нас есть 492 мошенничества из 284 807 транзакций. Набор данных сильно несбалансирован, на положительный класс (мошенничества) приходится 0,172% всех транзакций.
# 
# Он содержит только числовые входные переменные, которые являются результатом преобразования PCA. К сожалению, из-за проблем конфиденциальности мы не можем предоставить исходные характеристики и дополнительную справочную информацию о данных. Характеристики V1, V2,… V28 являются основными компонентами, полученными с помощью PCA, единственные функции, которые не были преобразованы с помощью PCA, - это «Время» и «Количество». Функция «Время» содержит секунды, прошедшие между каждой транзакцией и первой транзакцией в наборе данных. Функция «Сумма» представляет собой сумму транзакции. Эту функцию можно использовать для обучения с учетом затрат в зависимости от примера. Функция «Класс» — это переменная ответа, которая принимает значение 1 в случае мошенничества и 0 в противном случае.

# ### Провести разведочный анализ данных 

# In[27]:


# Импорт необходимых библиотек
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.api as sms
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
import plotly.express as px


# In[2]:


# чтение данных из csv файла
df = pd.read_csv('C:/Users/katya/OneDrive/Рабочий стол/ML2023/Task2/dataset/creditcard.csv')

# выведем первые 10 строк
df.head()


# ## какие зависимости в данных?

# In[3]:


# Построим HeatMap для выявления зависимостей в данных
#Исключен столбец с классификацией
corr = df.drop(columns=["Class"]).corr().abs()      
np.fill_diagonal(corr.values, 0)

sns.heatmap(corr, cmap="YlOrBr", annot=True, annot_kws={'size': 2})
plt.tight_layout()


# Наибольшие зависимости от времени и суммарной стоимости покупки (Amount).
# 
# Между признаками (V1-V28) высокой корреляции не выявлено.

# ## сбалансированы ли классы?

# In[4]:


df['Class'].value_counts()


# In[5]:


labels=["Genuine","Fraud"]

fraud_or_not = df["Class"].value_counts().tolist()
values = [fraud_or_not[0], fraud_or_not[1]]

fig = px.pie(values=df['Class'].value_counts(), names=labels , width=700, height=400, color_discrete_sequence=["green","black"]
             ,title="Fraud vs Genuine transactions")
fig.show()


# In[6]:


plt.figure(figsize=(3,4))
ax = sns.countplot(x='Class',data=df)
for i in ax.containers:
    ax.bar_label(i,)
    
print('Genuine:', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds:', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')


# Нет, классы не сбалансированны. Данных о мошеннических операциях значительно меньше.

# ## основные статистики признаков

# In[7]:


df.describe()


# ## Разделить данные на train/test: разделить на обучающую и тестовую подвыборки

# In[8]:


X = df.drop('Class', axis=1)
y = df['Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.3, random_state = 42)


# In[9]:


#сравнить статистики подвыборок и генеральной выборки
X_train.describe()


# In[10]:


X_test.describe()


# In[11]:


y_train.value_counts()


# In[12]:


y_test.value_counts()


# ## Обучение моделей классификации (без доп преобразований)
- какие метрики точности
- какая модель лучше всего справилась
# In[14]:


#Logistic Regression (Логистическая регрессия)

# программная реализация алгоритма логистическая регрессия
from sklearn.linear_model import LogisticRegression
# программная реализация расчета метрики точности
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# Создадим объект класса LogosticRegression
classifier_LR = LogisticRegression(random_state=33, max_iter=1000, multi_class='multinomial')

# Обучение модели
classifier_LR.fit(X_train, y_train)

# Прогноз
y_pred = classifier_LR.predict(X_test)

# Метрики
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print('Accuracy:',accuracy)
print('Precision:',precision)
print('Recall:',recall)


# In[16]:


# Алгоритм kNN
from sklearn.neighbors import KNeighborsClassifier

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Создадим объект класса KNN с параметром n_neighbors=3
classifier_kNN = KNeighborsClassifier(n_neighbors=3)

# Обучение модели
classifier_kNN.fit(X_train, y_train)
y_pred = classifier_kNN.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print('Accuracy:',accuracy)
print('Precision:',precision)
print('Recall:',recall)


# In[17]:


# Алгоритм Decision Tree
from sklearn.tree import DecisionTreeClassifier

DTC = DecisionTreeClassifier(random_state=33)

DTC.fit(X_train, y_train)

y_pred = DTC.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print('Accuracy:',accuracy)
print('Precision:',precision)
print('Recall:',recall)


# ## Обучение моделей классификации (SMOTE Oversampling)

# In[33]:


from imblearn.over_sampling import RandomOverSampler
# define oversampling strategy
ros = RandomOverSampler(random_state=42)
# fit and apply the transform
X_over, y_over = ros.fit_resample(X_train, y_train)


# In[34]:


#Logistic Regression (Логистическая регрессия)

# программная реализация алгоритма логистическая регрессия
from sklearn.linear_model import LogisticRegression
# программная реализация расчета метрики точности
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# Создадим объект класса LogosticRegression
classifier_LR = LogisticRegression(random_state=33, max_iter=1000, multi_class='multinomial')

# Обучение модели
classifier_LR.fit(X_over, y_over)

# Прогноз
y_pred = classifier_LR.predict(X_test)

# Метрики
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print('Accuracy:',accuracy)
print('Precision:',precision)
print('Recall:',recall)


# In[35]:


# Алгоритм kNN
from sklearn.neighbors import KNeighborsClassifier

# Создадим объект класса KNN с параметром n_neighbors=3
classifier_kNN = KNeighborsClassifier(n_neighbors=3)

# Обучение модели
classifier_kNN.fit(X_over, y_over)
y_pred = classifier_kNN.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print('Accuracy:',accuracy)
print('Precision:',precision)
print('Recall:',recall)


# In[37]:


# Алгоритм Decision Tree
from sklearn.tree import DecisionTreeClassifier

DTC = DecisionTreeClassifier(random_state=33)

DTC.fit(X_over, y_over)

y_pred = DTC.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print('Accuracy:',accuracy)
print('Precision:',precision)
print('Recall:',recall)


# ## Обучение моделей классификации (SMOTE Undersampling)

# In[38]:


from imblearn.under_sampling import RandomUnderSampler
# define oversampling strategy
rus = RandomUnderSampler(random_state=42)
# fit and apply the transform
X_under, y_under = rus.fit_resample(X_train, y_train)


# In[39]:


#Logistic Regression (Логистическая регрессия)

# программная реализация алгоритма логистическая регрессия
from sklearn.linear_model import LogisticRegression
# программная реализация расчета метрики точности
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# Создадим объект класса LogosticRegression
classifier_LR = LogisticRegression(random_state=33, max_iter=1000, multi_class='multinomial')

# Обучение модели
classifier_LR.fit(X_under, y_under)

# Прогноз
y_pred = classifier_LR.predict(X_test)

# Метрики
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print('Accuracy:',accuracy)
print('Precision:',precision)
print('Recall:',recall)


# In[40]:


# Алгоритм kNN
from sklearn.neighbors import KNeighborsClassifier

# Создадим объект класса KNN с параметром n_neighbors=3
classifier_kNN = KNeighborsClassifier(n_neighbors=3)

# Обучение модели
classifier_kNN.fit(X_under, y_under)
y_pred = classifier_kNN.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print('Accuracy:',accuracy)
print('Precision:',precision)
print('Recall:',recall)


# In[41]:


# Алгоритм Decision Tree
from sklearn.tree import DecisionTreeClassifier

DTC = DecisionTreeClassifier(random_state=33)

DTC.fit(X_under, y_under)

y_pred = DTC.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print('Accuracy:',accuracy)
print('Precision:',precision)
print('Recall:',recall)

- какие метрики точности?
Были выбраны: Метрика точности accuracy, Метрика точности Precision,Метрика полноты recall
- какая модель лучше всего справилась?
Поскольку данные несбалансированы, то корректнее брать модели с преобразованием (SMOTE)
Лучше всего с задачей справились модели SMOTE Oversampling: Decision Tree