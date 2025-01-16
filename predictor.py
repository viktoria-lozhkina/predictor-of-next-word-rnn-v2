# Импортируем необходимые библиотеки
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SimpleRNN
from typing import Tuple

# ШАГ 1: функция для подготовки данных для обучения рекуррентной нейросети
def load_and_prepare_data(file_path: str, max_length: int = 100) -> Tuple[np.ndarray, np.ndarray, Tokenizer]:
    """
    Загружаем текст из файла, токенизируем его и подготавливаем входные данные и метки для обучения модели RNN.

    :param file_path: Путь к файлу с текстом для обучения.
    :param max_length: Максимальная длина последовательностей.
    :return: Кортеж из массивов NumPy (X, y), где X - входные данные, y - метки.
    """
    # Загрузка текста из файла
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Токенизация текста
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])  # Обучаем токенизатор на нашем тексте

    # Преобразуем текст в последовательности токенов
    sequences = tokenizer.texts_to_sequences([text])[0]

    # Создаем пустые списки для входных данных (X) и меток (y) для обучения модели
    X = []  # Список для хранения входных данных
    y = []  # Список для хранения меток

    # Проходим по всем элементам последовательностей, начиная с индекса max_length
    for i in range(max_length, len(sequences)):
        # Добавляем в список X подмассив из последовательностей длиной max_length, заканчивающийся на текущем элементе
        X.append(sequences[i - max_length:i])  # Входные данные: последние max_length элементов перед текущим
        # Добавляем в список y текущий элемент последовательности как метку
        y.append(sequences[i])  # Метка: текущий элемент последовательности
    
    # Преобразуем X и y в массивы NumPy
    X = np.array(X)
    y = np.array(y)

    # Паддинг последовательностей при необходимости
    X = pad_sequences(X, maxlen=max_length)

    # Преобразуем метки в категориальный формат при необходимости
    y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

    return X, y, tokenizer

# ШАГ 2: Загрузка и подготовка данных

# !!!!!!! Загрузите свой файл для обучения модели вместо C:/Users/v-lozhkina/Desktop/Lozhkina-RNN/venv/povesti-belkina.txt

X, y, tokenizer = load_and_prepare_data('C:/Users/v-lozhkina/Desktop/Lozhkina-RNN/venv/povesti-belkina.txt')

# Теперь X и y готовы для обучения модели RNN
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)


# ШАГ 3: Создание модели

# Создание модели RNN с использованием последовательного подхода
model = Sequential()  # Инициализация последовательной модели

# Добавление слоя встраивания (Embedding) для преобразования токенов в векторы фиксированной длины
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=X.shape[1]))
# input_dim: количество уникальных токенов (словарь + 1 для учета нуля)
# output_dim: размерность векторов для представления токенов (100)
# input_length: длина входных последовательностей

# Добавление слоя простой рекуррентной нейронной сети (SimpleRNN) с 128 нейронами
model.add(SimpleRNN(128, return_sequences=False))
# return_sequences=False: возвращает только выход последнего временного шага, а не все временные шаги

# Добавление полносвязного слоя (Dense) для классификации с активацией softmax
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))
# len(tokenizer.word_index) + 1: количество классов для классификации (уникальные токены + 1 для нуля)
# activation='softmax': функция активации для многоклассовой классификации

# Компиляция модели
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ШАГ 4:
# Обучение модели
# ! Изменяйте кол-во эпох для контроля глубины обучения, у нас сейчас 10 эпох
model.fit(X, y, epochs=10, batch_size=64)

#ШАГ 5: Сохранение модели и токенизатора
import pickle
from tensorflow.keras.models import load_model

# Сохранение токенизатора
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Сохранение модели
model.save('my_model.h5')