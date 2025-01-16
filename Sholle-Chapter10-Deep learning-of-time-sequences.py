# Загрузим и распакуем архив с данными
!wget https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip
!unzip jena_climate_2009_2016.csv.zip


# Листинг 10.1. Обзор набора метеорологических данных Jena

# Импортируем модуль os, который предоставляет функции для работы с операционной системой
import os

# Определяем имя файла с данными, используя функцию os.path.join для объединения пути и имени файла
fname = os.path.join("jena_climate_2009_2016.csv")

# Открываем файл с именем fname для чтения
with open(fname) as f:
    # Считываем содержимое файла в переменную data
    data = f.read()

# Разделяем содержимое файла на строки, используя символ новой строки \n в качестве разделителя
lines = data.split("\n")

# Извлекаем первую строку файла (заголовок) и разбиваем её на поля, используя символ запятой , в качестве разделителя
header = lines[0].split(",")

# Удаляем первую строку (заголовок) из списка lines
lines = lines[1:]

# Выводим заголовок на экран
print(header)

# Выводим количество строк в файле (за исключением заголовка)
print(len(lines))



# Листинг 10.2. Преобразование данных
# Импортируем библиотеку NumPy
import numpy as np

# Создаём массив temperature, состоящий из такого же количества элементов, сколько строк в файле lines
temperature = np.zeros((len(lines),))

# Создаём двумерный массив raw_data, состоящий из количества строк в lines и на одну строку меньше, чем количество элементов в header
raw_data = np.zeros((len(lines), len(header) - 1))

# Перебираем строки файла lines с помощью цикла for
for i, line in enumerate(lines):
    # Создаём список values, состоящий из элементов строки line, преобразованных в числа с плавающей точкой
    values = [float(x) for x in line.split(",")[1:]]
    
    # Присваиваем второму элементу массива temperature значение, которое находится на втором месте в списке values
    temperature[i] = values[1]
    
    # Присваиваем элементам двумерного массива raw_data значения из списка values
    raw_data[i, :] = values[:]




# Листинг 10.3. Создание графика изменения температуры
from matplotlib import pyplot as plt
plt.plot(range(len(temperature)), temperature)

#Листинг 10.4. Создание графика изменения температуры по данным
#за первые десять дней

plt.plot(range(1440), temperature[:1440])


# Листинг 10.5. Вычисление количества образцов в каждой выборке
# Определяем количество обучающих примеров как половину от общего количества данных
num_train_samples = int(0.5 * len(raw_data))

# Определяем количество валидационных примеров как четверть от общего количества данных
num_val_samples = int(0.25 * len(raw_data))

# Определяем количество тестовых примеров как разницу между общим количеством данных и количеством обучающих и валидационных примеров
num_test_samples = len(raw_data) - num_train_samples - num_val_samples
print("num_train_samples:", num_train_samples)
print("num_val_samples:", num_val_samples)
print("num_test_samples:", num_test_samples)
#num_train_samples: 210225
#num_val_samples: 105112
#num_test_samples: 105114



## ПОДГОТОВКА ДАННЫХ

#Вот точная формулировка вопроса, на который нам нужно ответить в рамках
#задачи: можно ли предсказать температуру на следующие 24 часа по данным замеров, выполнявшихся один раз в час и охватывающих предыдущие пять дней?

## Нормализация

#Мы должны нормализовать
#временные последовательности независимо друг от друга, чтобы все они состояли из небольших по величине значений примерно одинакового масштаба.

mean = raw_data[:num_train_samples].mean(axis=0) # Находим среднее значение для всех признаков в обучающей выборке
raw_data -= mean # Вычитаем среднее значение из каждого признака
std = raw_data[:num_train_samples].std(axis=0) # Находим стандартное отклонение для всех признаков в обучающей выборке
raw_data /= std # Делим каждый признак на его стандартное отклонение

#пример:
#import numpy as np
#from tensorflow import keras
#int_sequence = np.arange(10)
#dummy_dataset = keras.utils.timeseries_dataset_from_array(
#    data=int_sequence[:-3],
 #   targets=int_sequence[3:],
 #   sequence_length=3,
 #   batch_size=2,
 #   )
#for inputs, targets in dummy_dataset:
#    for i in range(inputs.shape[0]):
#        print([int(x) for x in inputs[i]], int(targets[i]))


#Листинг 10.7. Создание наборов данных: обучающего, проверочного и контрольного

# Определяем частоту дискретизации данных
sampling_rate = 6
# Устанавливаем длину последовательности данных
sequence_length = 120
# Рассчитываем задержку в выборке данных
delay = sampling_rate * (sequence_length + 24 - 1)
# Задаем размер пакета данных для обработки
batch_size = 256

train_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay], # raw_data[:-delay] — берём реальные данные, исключая задержку
    targets=temperature[delay:], # temperature[delay:] — берём целевые значения температуры, начиная с момента задержки
    sampling_rate=sampling_rate, # sampling_rate=sampling_rate — устанавливаем частоту дискретизации данных
    sequence_length=sequence_length, # sequence_length=sequence_length — задаём длину последовательности данных
    shuffle=True, # shuffle=True — включаем перемешивание данных
    batch_size=batch_size, # batch_size=batch_size — задаём размер пакета данных
    start_index=0, # устанавливаем начальный индекс
    end_index=num_train_samples) # устанавливаем конечный индекс, равный размеру обучающей выборки

val_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples)


test_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples)

for samples, targets in train_dataset:
    print("samples shape:", samples.shape)
    print("targets shape:", targets.shape)
    break



#10.2.2. Базовое решение без привлечения машинного обучения

# Листинг 10.9. Оценка базового решения MAE
# Функция для оценки модели с использованием наивного метода
def evaluate_naive_method(dataset):
    # Инициализируем переменные для хранения общей абсолютной ошибки и количества просмотренных образцов
    total_abs_err = 0.
    samples_seen = 0
    # Проходимся по набору данных, состоящему из пар «образцы — цели»
    for samples, targets in dataset:
        # Прогнозируем значения, используя последний столбец образцов, умноженный на стандартное отклонение и добавленный к среднему значению
        preds = samples[:, -1, 1] * std[1] + mean[1]
        # Добавляем абсолютную разницу между прогнозами и целями к общей абсолютной ошибке
        total_abs_err += np.sum(np.abs(preds - targets))
        # Увеличиваем количество просмотренных образцов на количество образцов в текущей итерации
        samples_seen += samples.shape[0]
    # Возвращаем общую абсолютную ошибку, делённую на количество просмотренных образцов
    return total_abs_err / samples_seen
# Выводим значение средней абсолютной ошибки (MAE) на валидационном наборе данных
print(f"Validation MAE: {evaluate_naive_method(val_dataset):.2f}")
# Выводим значение средней абсолютной ошибки (MAE) на тестовом наборе данных
print(f"Test MAE: {evaluate_naive_method(test_dataset):.2f}")


# 10.2.3. Базовое решение c привлечением машинного обучения

# Листинг 10.10. Обучение и оценка полносвязной модели
# Импортируем необходимые библиотеки
from tensorflow import keras
from tensorflow.keras import layers
# Создаём входной слой с формой, соответствующей длине последовательности и количеству признаков в данных
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
# Сглаживаем входные данные
x = layers.Flatten()(inputs)
# Добавляем полносвязный слой с 16 нейронами и функцией активации ReLU
x = layers.Dense(16, activation="relu")(x)
# Добавляем выходной слой с одним нейроном
outputs = layers.Dense(1)(x)
# Собираем модель из входного и выходного слоёв
model = keras.Model(inputs, outputs)
#Использовать обратный вызов, чтобы сохранить лучшую модель
callbacks = [
    keras.callbacks.ModelCheckpoint("jena_dense.keras",
    save_best_only=True)
]



model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=callbacks)



#Загрузить лучшую модель и оценить ее на контрольных данных
model = keras.models.load_model("jena_dense.keras")
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")


# Листинг 10.11. Вывод результатов
import matplotlib.pyplot as plt
loss = history.history["mae"]
val_loss = history.history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="MAE на обучающих данных")
plt.plot(epochs, val_loss, "b", label="MAE на проверочных данных")
plt.title("MAE на обучающих и проверочных данных")
plt.legend()
plt.show()



# 10.2.4. Попытка использовать одномерную сверточную модель

# Создаём входной слой с указанием формы, которая соответствует длине последовательности и количеству признаков в данных
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
# Добавляем свёрточный слой с 8 фильтрами размером 24, используя функцию активации ReLU
x = layers.Conv1D(8, 24, activation="relu")(inputs)
# Применяем операцию максимального пула с размером пула 2
x = layers.MaxPooling1D(2)(x)
# Добавляем ещё один свёрточный слой с 8 фильтрами размером 12, используя функцию активации ReLU
x = layers.Conv1D(8, 12, activation="relu")(x)
# Снова применяем операцию максимального пула с размером пула 2
x = layers.MaxPooling1D(2)(x)
# Добавляем третий свёрточный слой с 8 фильтрами размером 6, используя функцию активации ReLU
x = layers.Conv1D(8, 6, activation="relu")(x)
# Используем глобальное усреднение для объединения карт признаков
x = layers.GlobalAveragePooling1D()(x)
# Добавляем выходной слой с одним нейроном
outputs = layers.Dense(1)(x)
# Собираем модель из входного и выходного слоёв
model = keras.Model(inputs, outputs)
# Определяем список обратных вызовов для сохранения лучшей модели
callbacks = [
    keras.callbacks.ModelCheckpoint("jena_conv.keras",
    save_best_only=True)
]
# Настраиваем оптимизатор, функцию потерь и метрики для компиляции модели
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
# Обучаем модель на обучающем наборе данных, используя 10 эпох, и валидируем на валидационном наборе данных
# Также используем определённые ранее обратные вызовы
history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=val_dataset,
                    callbacks=callbacks)
# Загружаем сохранённую модель
model = keras.models.load_model("jena_conv.keras")
# Вычисляем тестовую MAE и выводим результат
print(f"Тестовая MAE: {model.evaluate(test_dataset)[1]:.2f}")




## 10.2.5. Первое базовое рекуррентное решение

#Простая модель на основе слоя LSTM

# Создаём входной слой с указанием формы, которая соответствует длине последовательности и количеству признаков в данных
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))

# Добавляем слой LSTM с 16 нейронами
x = layers.LSTM(16)(inputs)

# Добавляем выходной слой с одним нейроном
outputs = layers.Dense(1)(x)

# Собираем модель из входного и выходного слоёв
model = keras.Model(inputs, outputs)

# Определяем список обратных вызовов для сохранения лучшей модели
callbacks = [
    keras.callbacks.ModelCheckpoint("jena_lstm.keras",
    save_best_only=True)
]

# Настраиваем оптимизатор, функцию потерь и метрики для компиляции модели
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

# Обучаем модель на обучающем наборе данных, используя 10 эпох, и валидируем на валидационном наборе данных
# Также используем определённые ранее обратные вызовы
history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=val_dataset,
                    callbacks=callbacks)

# Загружаем сохранённую модель
model = keras.models.load_model("jena_lstm.keras")

# Вычисляем тестовую MAE и выводим результат
print(f"Тестовая MAE: {model.evaluate(test_dataset)[1]:.2f}")



### 10.3. РЕКУРРЕНТНЫЕ НЕЙРОННЫЕ СЕТИ

##Листинг 10.14. Более подробная реализация RNN в псевдокоде
state_t = 0 # Состояние в момент t
for input_t in input_sequence: #Цикл по последовательности элементов
output_t = activation(dot(W, input_t) + dot(U, state_t) + b)
state_t = output_t # Предыдущее выходное значение становится текущим состоянием для следующей итерации


# Листинг 10.15. Реализация сети RNN на основе NumPy
import numpy as np
timesteps = 100 # Число временных интервалов во входной
input_features = 32 # Размерность пространства входных признаков последовательности
output_features = 64 # Размерность пространства выходных признаков
inputs = np.random.random((timesteps, input_features)) # Входные данные: случайный шум для простоты примера
state_t = np.zeros((output_features,)) # Начальное состояние: вектор с нулевыми значениями элементов
# Создание матриц со случайными весами
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))
successive_outputs = []
for input_t in inputs: # input_t — вектор с формой (входные_признаки,)
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b) # Объединение входных данных с текущим состоянием (выходными данными на предыдущем шаге). Функция tanh используется для придания нелинейности (здесь можно взять любую другую функцию активации)
    successive_outputs.append(output_t) # Сохранение выходных данных в список
    state_t = output_t # Обновление текущего состояния сети как подготовка к обработке следующего временного интервала
final_output_sequence = np.stack(successive_outputs, axis=0) # Окончательный результат — двумерный тензор с формой (временные_интервалы, выходные_признаки)


## 10.3.1. Рекуррентный слой в Keras

#Листинг 10.16. Слой сети RNN, способный обрабатывать последовательности
# любой длины
num_features = 14
inputs = keras.Input(shape=(None, num_features))
outputs = layers.SimpleRNN(16)(inputs) # Обратите внимание, что return_sequences=False — это значение по умолчанию
# Это может пригодиться в моделях, предназначенных для обработки последовательностей переменной длины  .Однако если все последовательности имеют
#одинаковую длину, лучше явно указать полную форму входных данных —
#это позволит методу model.summary() отображать информацию о длине выхода,
#что всегда полезно и в отдельных случаях дает возможность применению не-
#которых оптимизаций
print(outputs.shape) # Слой рекуррентной сети, возвращающий полную последовательность результатов
#(120, 16)

# Листинг 10.19. Стек из нескольких слоев RNN
inputs = keras.Input(shape=(steps, num_features))
x = layers.SimpleRNN(16, return_sequences=True)(inputs)
x = layers.SimpleRNN(16, return_sequences=True)(x)
outputs = layers.SimpleRNN(16)(x)

# Листинг 10.20. Реализация архитектуры LSTM в псевдокоде (1/2)
output_t = activation(dot(state_t, Uo) + dot(input_t, Wo) + dot(c_t, Vo) + bo) # вычисляется выходное значение скрытого слоя на текущем шаге
i_t = activation(dot(state_t, Ui) + dot(input_t, Wi) + bi) # строка вычисляет новое значение переменной i_t (input gate), которая определяет, какие входные данные будут использоваться на следующем шаге
f_t = activation(dot(state_t, Uf) + dot(input_t, Wf) + bf) # аналогично предыдущему, здесь вычисляется значение переменной f_t (forget gate), которая определяет, какая информация из предыдущего состояния будет забыта.
k_t = activation(dot(state_t, Uk) + dot(input_t, Wk) + bk) # здесь вычисляется значение переменной k_t (cell gate), которая определяет новый вектор состояния.
#Получим новое перенесенное состояние (c_t), объединив i_t, f_t и k_t.
# Листинг 10.21. Реализация архитектуры LSTM в псевдокоде (2/2)
c_t+1 = i_t * k_t + c_t * f_t



### 10.4. УЛУЧШЕННЫЕ МЕТОДЫ ИСПОЛЬЗОВАНИЯ РЕКУРРЕНТНЫХ НЕЙРОННЫХ СЕТЕЙ
## 10.4.1. Использование рекуррентного прореживания для борьбы с переобучением

#Листинг 10.22. Обучение и оценка модели на основе LSTM с регуляризацией прореживанием
# Создаём входной слой с указанием формы, которая соответствует длине последовательности и количеству признаков в данных
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))

# Добавляем слой LSTM с 32 нейронами и добавляем параметр recurrent_dropout, который устанавливает дропаут для внутренних рекуррентных соединений
x = layers.LSTM(32, recurrent_dropout=0.25)(inputs)

# Для регуляризации слоя Dense добавляем также слой Dropout после LSTM
x = layers.Dropout(0.5)(x)

# Добавляем выходной слой с одним нейроном
outputs = layers.Dense(1)(x)

# Собираем модель из входного и выходного слоёв
model = keras.Model(inputs, outputs)

# Определяем список обратных вызовов для сохранения лучшей модели
callbacks = [
    keras.callbacks.ModelCheckpoint("jena_lstm_dropout.keras",
    save_best_only=True)
]

# Настраиваем оптимизатор, функцию потерь и метрики для компиляции модели
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

# Обучаем модель на обучающем наборе данных, используя 50 эпох, и валидируем на валидационном наборе данных
# Также используем определённые ранее обратные вызовы
history = model.fit(train_dataset,
                    epochs=50,
                    validation_data=val_dataset,
                    callbacks=callbacks)




##10.4.2. Наложение нескольких рекуррентных слоев друг на друга

#Листинг 10.23. Обучение и оценка модели с несколькими слоями GRU
#и с регуляризацией прореживанием

# Создаём входной слой с указанием формы, которая соответствует длине последовательности и количеству признаков в данных
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))

# Добавляем слой GRU с 32 нейронами, параметром recurrent_dropout, который устанавливает дропаут для внутренних рекуррентных соединений, и параметром return_sequences, который возвращает последовательности
x = layers.GRU(32, recurrent_dropout=0.5, return_sequences=True)(inputs)

# Добавляем ещё один слой GRU с 32 нейронами и параметром recurrent_dropout
x = layers.GRU(32, recurrent_dropout=0.5)(x)

# Для регуляризации слоя Dense добавляем также слой Dropout после GRU
x = layers.Dropout(0.5)(x)

# Добавляем выходной слой с одним нейроном
outputs = layers.Dense(1)(x)

# Собираем модель из входного и выходного слоёв
model = keras.Model(inputs, outputs)

# Определяем список обратных вызовов для сохранения лучшей модели
callbacks = [
    keras.callbacks.ModelCheckpoint("jena_stacked_gru_dropout.keras",
    save_best_only=True)
]

# Настраиваем оптимизатор, функцию потерь и метрики для компиляции модели
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

# Обучаем модель на обучающем наборе данных, используя 50 эпох, и валидируем на валидационном наборе данных
# Также используем определённые ранее обратные вызовы
history = model.fit(train_dataset,
                    epochs=50,
                    validation_data=val_dataset,
                    callbacks=callbacks)

# Загружаем сохранённую модель
model = keras.models.load_model("jena_stacked_gru_dropout.keras")

# Вычисляем тестовую MAE (среднюю абсолютную ошибку) и выводим результат
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")


##10.4.3. Использование двунаправленных рекуррентных нейронных сетей

#Листинг 10.24. Обучение и оценка двунаправленной модели LSTM
# Создаём входной слой с указанием формы, которая соответствует длине последовательности и количеству признаков в данных
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))

# Добавляем слой двунаправленного LSTM с 16 нейронами
x = layers.Bidirectional(layers.LSTM(16))(inputs)

# Добавляем выходной слой с одним нейроном
outputs = layers.Dense(1)(x)

# Собираем модель из входного и выходного слоёв
model = keras.Model(inputs, outputs)

# Настраиваем оптимизатор, функцию потерь и метрики для компиляции модели
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

# Обучаем модель на обучающем наборе данных, используя 10 эпох, и валидируем на валидационном наборе данных
history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=val_dataset)