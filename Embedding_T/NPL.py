import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import pickle
import json
from typing import List, Dict, Tuple, Set
import time
import warnings

# --- ОПТИМАЛЬНЫЕ НАСТРОЙКИ ДЛЯ ГЛУБОКОГО АНАЛИЗА ---
MODEL_NAME = 'deepvk/USER-bge-m3'  # Лучшая модель для русского языка
LOCAL_MODEL_PATH = "user-bge-m3-local"  # Папка для локального хранения модели

# Имена файлов для кэширования (рядом со скриптом)
BASE_VECTORS_CACHE_FILE = "base_vectors_cache.pkl"
NEW_VECTORS_CACHE_FILE = "new_vectors_cache.pkl"
CLUSTERS_CACHE_FILE = "clusters_cache.json"

# Пороги сходства (настроены под USER-bge-m3 для русского языка)
THRESHOLD_SIMILARITY = 0.85  # Порог для определения дубликатов
MIN_CLUSTER_SIZE = 2  # Минимальное количество предложений в кластере для считывания за дубликат


# --- ПРОДВИНУТАЯ ОБРАБОТКА МОДЕЛИ С ОФФЛАЙН-ПОДДЕРЖКОЙ ---

def load_model_offline(model_name: str, local_path: str) -> SentenceTransformer:
    """
    Загружает модель полностью оффлайн. Если локальной копии нет - предупреждает, но не пытается скачать.
    """
    if os.path.exists(local_path):
        print(f"✅ Загрузка модели из локальной папки: '{local_path}'...")
        try:
            model = SentenceTransformer(local_path)
            print("✅ Модель успешно загружена для оффлайн-работы.")
            return model
        except Exception as e:
            print(f"❌ Ошибка при загрузке локальной модели: {e}")
            print("❗ Проверьте целостность файлов в папке модели.")
            raise

    # Если локальной модели нет - информируем пользователя
    print(f"\n{'=' * 60}")
    print(f"❌ ЛОКАЛЬНАЯ МОДЕЛЬ НЕ НАЙДЕНА: '{local_path}'")
    print(f"❗ Для работы БЕЗ ИНТЕРНЕТА необходимо предварительно скачать модель:")
    print(f"   1. Подключитесь к интернету один раз")
    print(f"   2. Запустите этот скрипт с интернетом - модель скачается и сохранится локально")
    print(f"   3. После этого можно работать полностью оффлайн")
    print(f"{'=' * 60}\n")

    # Предлагаем скачать модель, если есть интернет
    try:
        print(f"⬇️ Попытка скачать модель '{model_name}' для последующей оффлайн-работы...")
        model = SentenceTransformer(model_name)
        print(f"➡️ Сохранение модели в локальную папку: '{local_path}'...")

        # Создаем директорию если не существует
        os.makedirs(local_path, exist_ok=True)
        model.save(local_path)
        print("✅ Модель успешно сохранена для оффлайн-использования.")
        return model
    except Exception as e:
        print(f"❌ Невозможно скачать модель: {e}")
        print("❗ Работа в оффлайн-режиме невозможна без предварительной загрузки модели.")
        raise


# --- УЛУЧШЕННОЕ КЭШИРОВАНИЕ ЭМБЕДДИНГОВ ---

def get_embeddings_with_cache(sentences: List[str], model: SentenceTransformer,
                              cache_file: str, description: str = "") -> np.ndarray:
    """
    Получает эмбеддинги с кэшированием на диск. Проверяет целостность кэша.
    """
    if os.path.exists(cache_file):
        try:
            print(f"\n✅ Загрузка эмбеддингов ({description}) из кэша: '{cache_file}'...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)

            # Проверяем целостность кэша
            if (isinstance(cached_data, dict) and
                    'embeddings' in cached_data and
                    'sentence_count' in cached_data and
                    cached_data['sentence_count'] == len(sentences)):

                print(f"   ✅ Кэш валиден. Загружено {len(cached_data['embeddings'])} векторов.")
                return cached_data['embeddings']
            else:
                print("   ⚠️ Кэш поврежден или не соответствует текущим данным. Пересчет...")
        except Exception as e:
            print(f"   ⚠️ Ошибка при загрузке кэша: {e}. Пересчет...")

    # Если кэша нет или он невалиден - считаем заново
    print(f"\n🧠 Векторизация {len(sentences)} предложений ({description})...")
    print("   ⏱️  Это может занять несколько минут (модель загружена в память)...")

    start_time = time.time()
    embeddings = model.encode(sentences, show_progress_bar=True, batch_size=32)
    end_time = time.time()

    print(f"   ✅ Векторизация завершена за {end_time - start_time:.2f} секунд.")

    # Сохраняем в кэш
    cache_data = {
        'embeddings': embeddings,
        'sentence_count': len(sentences),
        'model_name': MODEL_NAME,
        'timestamp': time.time()
    }

    print(f"➡️ Сохранение эмбеддингов в кэш: '{cache_file}'...")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    print("   ✅ Кэширование завершено.")

    return embeddings


# --- ПРОДВИНУТАЯ КЛАСТЕРИЗАЦИЯ ДУБЛИКАТОВ С ИСПРАВЛЕНИЕМ ТИПОВ ДАННЫХ ---

def find_duplicate_clusters(all_sentences: List[str], all_embeddings: np.ndarray,
                            new_sentence_indices: Set[int]) -> Dict[int, List[Dict]]:
    """
    Находит кластеры дубликатов с использованием DBSCAN для более точной группировки.
    Возвращает структуру: {cluster_id: [предложения с метаданными]}

    ИСПРАВЛЕНО: Все ключи словаря теперь имеют тип int (не numpy.int64)
    """
    print(f"\n🔍 Поиск кластеров дубликатов с порогом сходства {THRESHOLD_SIMILARITY:.2f}...")

    # Используем DBSCAN для кластеризации
    clustering = DBSCAN(
        metric='cosine',
        eps=1 - THRESHOLD_SIMILARITY,  # Преобразуем сходство в расстояние
        min_samples=MIN_CLUSTER_SIZE
    )

    cluster_labels = clustering.fit_predict(all_embeddings)

    # Группируем предложения по кластерам
    clusters = {}
    for idx, cluster_id in enumerate(cluster_labels):
        if cluster_id == -1:  # -1 означает шум (уникальные предложения)
            continue

        # ИСПРАВЛЕНИЕ: Преобразуем numpy.int64 в стандартный Python int
        cluster_id_int = int(cluster_id)

        if cluster_id_int not in clusters:
            clusters[cluster_id_int] = []

        sentence_info = {
            'index': int(idx),  # Преобразуем в int
            'text': all_sentences[idx],
            'is_new': idx in new_sentence_indices,
            # ИСПРАВЛЕНИЕ: оригинальный индекс должен быть корректным для новых и базовых предложений
            'original_index': idx if idx < len(new_sentence_indices) else idx - len(new_sentence_indices)
        }
        clusters[cluster_id_int].append(sentence_info)

    # Фильтруем кластеры, оставляя только те, где есть хотя бы одно новое предложение
    relevant_clusters = {}
    for cluster_id, sentences in clusters.items():
        if any(sent['is_new'] for sent in sentences):
            relevant_clusters[cluster_id] = sentences

    print(f"   ✅ Найдено {len(relevant_clusters)} релевантных кластеров с дубликатами.")
    return relevant_clusters


# --- АНАЛИЗ УНИКАЛЬНЫХ ПРЕДЛОЖЕНИЙ С ИСПРАВЛЕНИЕМ ТИПОВ ---

def find_unique_sentences(new_embeddings: np.ndarray, base_embeddings: np.ndarray,
                          new_sentences: List[str], threshold: float = 0.6) -> List[Dict]:
    """
    Находит уникальные предложения, которые не похожи ни на что в базе и среди новых.
    """
    print(f"\n✨ Поиск уникальных предложений (порог сходства < {threshold:.2f})...")

    unique_sentences = []

    # Сравнение новых предложений с базой
    if len(base_embeddings) > 0:
        base_similarity = cosine_similarity(new_embeddings, base_embeddings)
        max_base_similarity = np.max(base_similarity, axis=1)
    else:
        max_base_similarity = np.zeros(len(new_sentences))

    # Сравнение новых предложений между собой
    if len(new_embeddings) > 1:
        new_similarity = cosine_similarity(new_embeddings)
        # Для каждого предложения находим максимальное сходство с другими новыми
        np.fill_diagonal(new_similarity, 0)  # Исключаем сравнение с самим собой
        max_new_similarity = np.max(new_similarity, axis=1)
    else:
        max_new_similarity = np.zeros(len(new_sentences))

    # Определяем уникальные предложения
    for i, sentence in enumerate(new_sentences):
        if (max_base_similarity[i] < threshold and
                max_new_similarity[i] < threshold):
            # ИСПРАВЛЕНИЕ: Преобразуем numpy float в стандартный float
            unique_sentences.append({
                'index': int(i),  # Преобразуем в int
                'text': sentence,
                'max_base_similarity': float(max_base_similarity[i]),
                'max_new_similarity': float(max_new_similarity[i])
            })

    print(f"   ✅ Найдено {len(unique_sentences)} уникальных предложений.")
    return unique_sentences


# --- ФОРМАТИРОВАННЫЙ ВЫВОД РЕЗУЛЬТАТОВ ---

def print_results(clusters: Dict[int, List[Dict]], unique_sentences: List[Dict],
                  new_sentences: List[str], base_sentences: List[str]):
    """
    Красиво выводит результаты анализа.
    """
    print(f"\n{'=' * 80}")
    print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА ДУБЛИКАТОВ")
    print(f"{'=' * 80}")

    if not clusters and not unique_sentences:
        print("ℹ️  Не найдено ни дубликатов, ни уникальных предложений.")
        return

    # Вывод кластеров дубликатов
    if clusters:
        print(f"\n{'-' * 80}")
        print("🎯 НАЙДЕНЫ КЛАСТЕРЫ ДУБЛИКАТОВ:")
        print(f"{'-' * 80}")

        # Сортируем кластеры по ID для предсказуемого вывода
        for cluster_id in sorted(clusters.keys()):
            sentences = clusters[cluster_id]
            print(f"\n📋 Кластер #{cluster_id + 1} ({len(sentences)} предложений)")
            print("   " + "-" * 50)

            new_in_cluster = [s for s in sentences if s['is_new']]
            base_in_cluster = [s for s in sentences if not s['is_new']]

            if new_in_cluster:
                print(f"🆕 НОВЫЕ предложения в кластере ({len(new_in_cluster)}):")
                for sent in new_in_cluster:
                    print(f"   • [A{sent['original_index']}] {sent['text']}")

            if base_in_cluster:
                print(f"\n💾 СУЩЕСТВУЮЩИЕ предложения в базе ({len(base_in_cluster)}):")
                for sent in base_in_cluster:
                    original_idx = sent['original_index']
                    print(f"   • [B{original_idx}] {sent['text']}")

    # Вывод уникальных предложений
    if unique_sentences:
        print(f"\n{'-' * 80}")
        print("✨ УНИКАЛЬНЫЕ ПРЕДЛОЖЕНИЯ (низкое сходство с базой и новыми):")
        print(f"{'-' * 80}")

        for i, sent in enumerate(unique_sentences, 1):
            print(f"\n💎 Уникальное предложение #{i}:")
            print(f"   • [A{sent['index']}] {sent['text']}")
            print(f"   📊 Макс. сходство с базой: {sent['max_base_similarity']:.3f}")
            print(f"   📊 Макс. сходство с новыми: {sent['max_new_similarity']:.3f}")

    print(f"\n{'=' * 80}")
    print(f"✅ АНАЛИЗ ЗАВЕРШЕН")
    print(f"   • Всего кластеров с дубликатами: {len(clusters)}")
    print(f"   • Уникальных предложений: {len(unique_sentences)}")
    print(f"   • Обработано новых предложений: {len(new_sentences)}")
    print(f"   • Предложений в базе: {len(base_sentences)}")
    print(f"{'=' * 80}")


# --- ДОПОЛНИТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ ПОДГОТОВКИ РЕЗУЛЬТАТОВ К СОХРАНЕНИЮ В JSON ---

def prepare_results_for_json(clusters: Dict[int, List[Dict]], unique_sentences: List[Dict],
                             new_sentences: List[str], base_sentences: List[str]) -> Dict:
    """
    Подготавливает результаты для сохранения в JSON, преобразуя все numpy типы в стандартные Python типы.
    """
    # Создаем копию clusters с преобразованными ключами и значениями
    json_clusters = {}
    for cluster_id, sentences in clusters.items():
        # Преобразуем ключ в строку для надежности
        json_cluster_id = str(cluster_id)

        json_sentences = []
        for sent in sentences:
            json_sentences.append({
                'index': int(sent['index']),
                'text': str(sent['text']),
                'is_new': bool(sent['is_new']),
                'original_index': int(sent['original_index'])
            })
        json_clusters[json_cluster_id] = json_sentences

    # Подготавливаем уникальные предложения
    json_unique = []
    for sent in unique_sentences:
        json_unique.append({
            'index': int(sent['index']),
            'text': str(sent['text']),
            'max_base_similarity': float(sent['max_base_similarity']),
            'max_new_similarity': float(sent['max_new_similarity'])
        })

    return {
        'clusters': json_clusters,
        'unique_sentences': json_unique,
        'summary': {
            'total_clusters': len(clusters),
            'total_unique': len(unique_sentences),
            'total_new_sentences': len(new_sentences),
            'total_base_sentences': len(base_sentences),
            'threshold_used': THRESHOLD_SIMILARITY
        },
        'metadata': {
            'model_used': MODEL_NAME,
            'timestamp': time.time(),
            'analysis_version': '1.1'
        }
    }


# --- ОСНОВНАЯ ФУНКЦИЯ С ОПТИМИЗАЦИЕЙ И ИСПРАВЛЕНИЕМ ОШИБКИ ---

def analyze_sentences(new_sentences: List[str], base_sentences: List[str]):
    """
    Основная функция анализа предложений на дубликаты и уникальность.
    """
    print(f"🚀 ЗАПУСК АНАЛИЗА ПРЕДЛОЖЕНИЙ")
    print(f"   • Новых предложений: {len(new_sentences)}")
    print(f"   • Предложений в базе: {len(base_sentences)}")
    print(f"   • Модель: {MODEL_NAME}")
    print(f"   • Порог сходства: {THRESHOLD_SIMILARITY:.2f}")

    try:
        # 1. Загрузка модели (полностью оффлайн после первого запуска)
        model = load_model_offline(MODEL_NAME, LOCAL_MODEL_PATH)

        # 2. Комбинируем все предложения для общей кластеризации
        all_sentences = new_sentences + base_sentences
        new_indices = set(range(len(new_sentences)))  # Индексы новых предложений

        # 3. Получаем эмбеддинги с кэшированием
        all_embeddings = get_embeddings_with_cache(
            all_sentences,
            model,
            "all_embeddings_cache.pkl",
            "все предложения (новые + база)"
        )

        # 4. Находим кластеры дубликатов
        clusters = find_duplicate_clusters(all_sentences, all_embeddings, new_indices)

        # 5. Находим уникальные предложения среди новых
        new_embeddings = all_embeddings[:len(new_sentences)]
        base_embeddings = all_embeddings[len(new_sentences):] if base_sentences else np.array([])

        unique_sentences = find_unique_sentences(
            new_embeddings,
            base_embeddings,
            new_sentences,
            threshold=THRESHOLD_SIMILARITY - 0.25  # Более строгий порог для уникальности
        )

        # 6. Выводим результаты
        print_results(clusters, unique_sentences, new_sentences, base_sentences)

        # 7. ПОДГОТАВЛИВАЕМ РЕЗУЛЬТАТЫ ДЛЯ JSON (ИСПРАВЛЕНО)
        results_for_json = prepare_results_for_json(clusters, unique_sentences, new_sentences, base_sentences)

        # 8. Сохраняем результаты в файл для дальнейшего использования
        results_file = 'analysis_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_for_json, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Результаты анализа успешно сохранены в '{results_file}'")

        return results_for_json

    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        print("💡 Рекомендации по решению:")
        print("   • Для первого запуска требуется интернет для загрузки модели")
        print("   • Проверьте наличие места на диске (модель занимает ~1.5 GB)")
        print("   • Убедитесь, что все предложения являются строками")
        # Выводим тип ошибки для отладки
        print(f"   • Тип ошибки: {type(e).__name__}")
        if hasattr(e, '__traceback__'):
            import traceback
            print("\n📝 Детали стека вызовов:")
            traceback.print_exc()
        raise


# --- ДЕМОНСТРАЦИОННЫЕ ДАННЫЕ ДЛЯ ТЕСТИРОВАНИЯ ---

if __name__ == "__main__":
    # Пример использования с вашими данными

    # 100 НОВЫХ предложений (A) - для демонстрации берем меньше
    new_sentences_demo = [
        # Кластер дубликатов 1
        "Сделать процесс утверждения заявок быстрее.",
        "Ускорить процесс согласование запросов.",
        "Сократить время ожидания при рассмотрении новых предложений.",

        # Кластер дубликатов 2
        "Добавить новую кнопку в интерфейс.",
        "Внедрить иконку для быстрого доступа к настройкам.",

        # Уникальные предложения
        "Организовать обучающий семинар по работе с новым модулем.",
        "Внедрить двухфакторную аутентификацию для всех пользователей.",
        "Добавить возможность экспорта данных в формате CSV.",
        "Провести аудит безопасности существующей IT-инфраструктуры.",
        "Обеспечить обеды для начальников управлений бесплатно.",
        "Главного в управлении кормить за счет организации.",
    ]

    # 5000 ПРЕДЛОЖЕНИЙ БАЗЫ (B) - для демонстрации берем меньше
    base_sentences_demo = [
        "Переместить элемент 'Поиск' в верхний левый угол.",
        "Сделать согласование документов моментальным.",
        "Изменить цвет заголовка на синий.",
        "Добавить функционал быстрого доступа к настройкам через иконку.",
        "Провести обучение сотрудников по новым инструментам безопасности.",
        "для начальников организовать централизованное питание в столовой.",
    ]

    print("🧪 ЗАПУСК ДЕМОНСТРАЦИОННОГО РЕЖИМА")
    print("   (В реальном использовании замените данные на ваши)")

    results = analyze_sentences(new_sentences_demo, base_sentences_demo)

    print("\n✅ ДЕМОНСТРАЦИОННЫЙ РЕЖИМ ЗАВЕРШЕН УСПЕШНО!")