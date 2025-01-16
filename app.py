# Импортируем нужные библиотеки
from flask import Flask, request, jsonify, render_template
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import numpy as np

app = Flask(__name__)

# Загрузка обученной модели
model = load_model('my_model.h5')

# Загрузка токенизатора из pickle файла
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model, tokenizer, input_text: str, max_length: int = 100) -> str:
    """
    Предсказывает следующее слово на основе введенного текста.
    
    :param model: Обученная модель для предсказания.
    :param tokenizer: Токенизатор для преобразования текста в последовательности.
    :param input_text: Введенный пользователем текст, на основе которого производится предсказание.
    :param max_length: Максимальная длина последовательности.
    :return: Предсказанное слово или None, если слово не предсказано.
    """
    # Преобразуем введенный текст в последовательности чисел
    sequence = tokenizer.texts_to_sequences([input_text])[0]
    
    # Обрезаем последовательность до максимальной длины
    if len(sequence) > max_length:
        sequence = sequence[-max_length:]
    
    # Дополняем последовательность до максимальной длины
    padded_sequence = pad_sequences([sequence], maxlen=max_length)
    
    # Делаем предсказание с помощью модели
    predicted = model.predict(padded_sequence, verbose=0)
    
    # Получаем индекс слова с максимальным значением вероятности
    predicted_index = np.argmax(predicted, axis=-1)[0]
    
    # Ищем слово по индексу в токенизаторе
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    
    return None  # Возвращаем None, если слово не найдено

@app.route('/')
def home() -> str:
    """Отображает главную страницу."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict() -> str:
    """Обрабатывает запрос на предсказание следующего слова.
    
    :return: JSON-ответ с предсказанным словом.
    """
    # Получаем текст из формы
    input_text = request.form['input_text']
    
    # Вызываем функцию предсказания
    predicted_word = predict_next_word(model, tokenizer, input_text)
    
    # Возвращаем предсказанное слово в формате JSON
    return jsonify({'predicted_word': predicted_word})

if __name__ == '__main__':
    app.run(debug=True)  # Запускаем приложение в отладочном режиме