# Wdrożenie na produkcję

1. Przygotowanie danych
    1. Usunięcie ostatnich wierszy z podsumowaniem,
    2. Usunięcie niepotrzebnych kolumn z danych:
        * `id`, `url` są unikatowe dla każdego wiersza w tabeli
        * `desc`, `emp_title`, `title` mogą być dowolnym stringiem, w praktyce są prawie unikatowe
    3. Zamiana każdej kolumny z datami w formacie "May-2020" na 2 kolumny z rokiem i miesiącem.
    4. Zakodowanie zmiennych kategorycznych w sposób możliwy do użycia w modelach np. one-hot encoding
    5. Wypełnienie brakujących danych
2. Trening i testowanie modeli
    * Każdy model testujemy za pomocą cross-validation
    * W modelach z hiperparametrami szukamy najlepszy za pomocą metod takich jak grid search, random search
3. Wdrożenie na produkcje najlepszego modelu
    1. Zapisanie wytrenowanego modelu
    2. Napisanie skryptu do predykcji nowych danych