"""
Данный скрипт реализует симплекс-метод для решения задач линейного программирования, 
в частности, для максимизации и минимизации целевых функций. Он включает в себя 
функции для создания расширенной симплекс-таблицы, нормализации строк матрицы 
для максимизации и минимизации, а также для выполнения самого симплекс-метода.

Основные функции:

1. **dummy_variable(self, params, function)**:
   - Создает расширенную симплекс-таблицу, добавляя искусственные переменные и 
     нулевые значения. Это необходимо для корректного выполнения симплекс-метода, 
     особенно в случаях, когда исходная задача не имеет явных базисных решений.

2. **norm_c(matrix)**:
   - Нормализует строки матрицы симплекс-таблицы для максимизации целевой функции. 
     Находит максимальный элемент в последней строке и использует его для 
     нормализации, а затем обновляет остальные строки матрицы.

3. **norm_b(matrix)**:
   - Нормализует строки матрицы симплекс-таблицы для минимизации целевой функции. 
     Находит минимальный элемент в правом столбце и использует его для 
     нормализации, обновляя остальные строки аналогично функции norm_c.

4. **simplex_max(matrix)**:
   - Выполняет симплекс-метод для максимизации целевой функции. Итеративно 
     нормализует матрицу, пока в последней строке есть положительные элементы. 
     В конце определяет базисные переменные и возвращает их значения вместе с 
     оптимальным значением целевой функции.

5. **simplex_min(matrix)**:
   - Выполняет симплекс-метод для минимизации целевой функции. Работает аналогично 
     функции simplex_max, но с учетом отрицательных элементов в правом столбце.

6. **print_matrix(matrix)**:
   - Печатает матрицу с форматированием для удобства чтения. Использует 
     выравнивание для улучшения визуального восприятия данных.

В конце скрипта создаются примеры матриц для двух игроков (A и B), 
после чего выполняются симплексные методы для нахождения оптимальных 
значений переменных и целевых функций. Результаты выводятся на экран, 
включая оптимальные смешанные стратегии для обоих игроков и цену игры.
"""

import numpy as np


def dummy_variable(self, params, function):
    """
    Создает расширенную симплекс-таблицу, добавляя искусственные переменные и нулевые значения.
    """
    # Добавляем единичную матрицу для искусственных переменных
    A_extended = np.hstack((self, np.eye(self.shape[0])))

    # Расширяем вектор целевой функции, добавляя нули для искусственных переменных
    c_extended = np.concatenate((function, np.zeros(self.shape[0])))

    # Добавляем расширенный вектор целевой функции в матрицу
    A_extended = np.vstack((A_extended, c_extended))

    # Расширяем вектор правых частей, добавляя ноль для целевой функции
    b_extended = np.append(params, 0)

    # Объединяем матрицу с вектором правых частей
    A_extended = np.column_stack((A_extended, b_extended))

    return A_extended


def norm_c(matrix):
    """
    Нормализует строки матрицы симплекс-таблицы для максимизации.
    """
    # Находим индекс максимального элемента в последней строке (целевой функции)
    max_element_row_index = np.argmax(matrix[-1, :-1])

    # Находим максимальный элемент в соответствующем столбце
    max_element_column = np.max(matrix[:, max_element_row_index])

    # Находим индекс строки с максимальным элементом в этом столбце
    max_element_column_index = np.argmax(matrix[:, max_element_row_index])

    # Нормализуем строку с максимальным элементом
    matrix[max_element_column_index, :] /= max_element_column

    # Обновляем остальные строки
    for i in range(matrix.shape[0]):
        if i != max_element_column_index:
            # Вычисляем коэффициент для вычитания
            ratio = (
                matrix[i, max_element_row_index]
                / matrix[max_element_column_index, max_element_row_index]
            )
            # Вычитаем пропорциональную строку
            matrix[i, :] -= ratio * matrix[max_element_column_index, :]

    return matrix


def norm_b(matrix):
    """
    Нормализует строки матрицы симплекс-таблицы для минимизации.
    """
    # Находим индекс столбца с минимальным элементом в правом столбце
    max_element_column_index = np.argmin(matrix[:-1, -1])

    # Находим индекс минимального элемента в соответствующей строке
    max_element_row_index = np.argmin(matrix[max_element_column_index, :-1])

    # Получаем минимальный элемент в этой строке
    max_element_row = matrix[max_element_column_index, max_element_row_index]

    # Нормализуем строку с минимальным элементом
    matrix[max_element_column_index, :] /= max_element_row

    # Обновляем остальные строки
    for i in range(matrix.shape[0]):
        if i != max_element_column_index:
            # Вычисляем коэффициент для вычитания
            ratio = (
                matrix[i, max_element_row_index]
                / matrix[max_element_column_index, max_element_row_index]
            )
            # Вычитаем пропорциональную строку
            matrix[i, :] -= ratio * matrix[max_element_column_index, :]

    return matrix


def simplex_max(matrix):
    """
    Выполняет симплекс-метод для максимизации целевой функции.
    """
    iteration = 1  # Счетчик итераций
    # Продолжаем, пока есть положительные элементы в последней строке (целевой функции)
    while np.any(matrix[-1, :-1] > 0):
        # Нормализуем матрицу для максимизации
        norm_c(matrix)

        # Выводим текущую нормализованную матрицу
        print(f"\nШаг №{iteration}. Нормированная матрица:")
        print_matrix(matrix)
        iteration += 1

    # Определяем базисные переменные
    basic_variables = []
    for col in range(matrix.shape[1] - 1):
        if np.all(matrix[:, col] == 0) or np.count_nonzero(matrix[:, col]) != 1:
            basic_variables.append(0)  # Если столбец не является базисным
        else:
            row_index = np.argmax(
                np.abs(matrix[:, col])
            )  # Индекс строки с максимальным элементом
            basic_variables.append(
                matrix[row_index, -1]
            )  # Добавляем значение базисной переменной

    optimal_value = matrix[-1, -1]  # Оптимальное значение целевой функции
    return basic_variables, optimal_value


def simplex_min(matrix):
    """
    Выполняет симплекс-метод для минимизации целевой функции.
    """
    iteration = 1  # Счетчик итераций
    # Продолжаем, пока есть отрицательные элементы в последнем столбце (правые части)
    while np.any(matrix[:-1, -1] < 0):
        # Нормализуем матрицу для минимизации
        norm_b(matrix)

        # Выводим текущую нормализованную матрицу
        print(f"\nШаг №{iteration}. Нормированная матрица:")
        print_matrix(matrix)
        iteration += 1

    # Определяем базисные переменные
    basic_variables = []
    for col in range(matrix.shape[1] - 1):
        if np.all(matrix[:, col] == 0) or np.count_nonzero(matrix[:, col]) != 1:
            basic_variables.append(0)  # Если столбец не является базисным
        else:
            row_index = np.argmax(
                np.abs(matrix[:, col])
            )  # Индекс строки с максимальным элементом
            basic_variables.append(
                matrix[row_index, -1]
            )  # Добавляем значение базисной переменной

    optimal_value = matrix[-1, -1]  # Оптимальное значение целевой функции
    return basic_variables, optimal_value


def print_matrix(matrix, title="Матрица"):
    """
    Печатает матрицу с форматированием для удобства чтения.
    """
    print(f"\n{title}:")
    print("-" * (len(title) + 2))  # Разделитель под заголовком

    # Определяем ширину для форматирования
    width = (
        max(len(f"{value:.2f}") for row in matrix for value in row) + 2
    )  # +2 для отступа

    # Печатаем строки матрицы
    for i, row in enumerate(matrix):
        for value in row:
            print(f"{value:>{width}.2f}", end="")
        print()  # Переход на новую строку

        # Добавляем разделитель между строками
        if i < len(matrix) - 1:  # Не добавляем после последней строки
            print("-" * (width * len(row)))  # Разделитель между строками

    print()  # Пустая строка после матрицы для лучшей читаемости


def main():
    """
    Основная функция программы, которая реализует симплекс-метод для решения задач линейного программирования
    в контексте теории игр. В данной функции рассматриваются две стратегии для двух игроков (A и B) и
    вычисляются оптимальные смешанные стратегии для каждого из них.
    """

    A = np.array(
        [[18, 7, 15, 2], [0, 13, 16, 3], [1, 17, 9, 19], [3, 15, 17, 9], [5, 2, 11, 4]]
    )
    print("Матрица A:")
    print_matrix(A)

    c = np.array([1, 1, 1, 1])
    b = np.array([1, 1, 1, 1, 1])

    print("\nСимплекс-таблица для игрока A:")
    print_matrix(dummy_variable(-1 * np.transpose(A), -1 * c, -1 * b))

    result_variables_A, result_value_A = simplex_min(
        dummy_variable(-1 * np.transpose(A), -1 * c, -1 * b)
    )

    print("\nОптимальное значение переменных для игрока A:")
    for i, val in enumerate(result_variables_A, start=1):
        print(f"u{i} = {val:.4f}")
    print(f"W = {result_value_A:.4f}")

    print("\nСимплекс-таблица для игрока B:")
    print_matrix(dummy_variable(A, b, c))

    result_variables_B, result_value_B = simplex_max(dummy_variable(A, b, c))

    print("\nОптимальное значение переменных для игрока B:")
    for i, val in enumerate(result_variables_B, start=1):
        print(f"v{i} = {val:.4f}")
    print(f"Z = {-result_value_B:.4f}")

    g = 1 / result_value_A
    print("\n_____________________________")
    print(f"g = {g:.4f}")
    for i, val in enumerate(result_variables_A, start=1):
        print(f"x{i} = {g * val:.4f}")
    print(
        f"Оптимальная смешанная стратегия игрока A - ({', '.join(f'{g * val:.4f}' for val in result_variables_A)})"
    )

    h = -1 / result_value_B
    print("\n_____________________________")
    print(f"h = {h:.4f}")
    for i, val in enumerate(result_variables_B, start=1):
        print(f"y{i} = {h * val:.4f}")
    print(
        f"Оптимальная смешанная стратегия игрока B - ({', '.join(f'{h * val:.4f}' for val in result_variables_B)})"
    )

    print("\nЦена игры будет равна:\n1/W = 1/Z =", round(g, 4))

    return 0


if __name__ == "__main__":
    main()
