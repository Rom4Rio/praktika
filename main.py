import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COUNT = 1000
LEFT = -10000
RIGHT = 10000
SEED = 42


class Dataset:
    """Управляет наборами Series и DataFrame."""

    def __init__(self, count=COUNT, left=LEFT, right=RIGHT, seed=SEED):
        """Сохраняет параметры будущего набора данных."""
        self.count = count
        self.left = left
        self.right = right
        self.seed = seed
        self.series = pd.Series(dtype="int64", name="Исходные значения")
        self.frame = pd.DataFrame()

    def generate_series(self):
        """Заполняет Series воспроизводимыми случайными числами."""
        generator = np.random.default_rng(self.seed)
        values = generator.integers(
            low=self.left,
            high=self.right + 1,
            size=self.count,
        )
        self.series = pd.Series(values, name="Исходные значения")

    def print_series(self):
        """Показывает первые строки Series."""
        print("Первые 10 строк набора Series:")
        print(self.series.head(10).to_string())
        print()

    @staticmethod
    def bubble_sort(values):
        """Упорядочивает копию списка пузырьковым методом."""
        ordered = values.copy()
        last = len(ordered) - 1

        while last > 0:
            changed = False

            for position in range(last):
                if ordered[position] > ordered[position + 1]:
                    ordered[position], ordered[position + 1] = (
                        ordered[position + 1],
                        ordered[position],
                    )
                    changed = True

            if not changed:
                break

            last -= 1

        return ordered

    def build_frame(self):
        """Создает DataFrame с двумя вариантами сортировки."""
        source = self.series.tolist()
        ascending = self.bubble_sort(source)
        descending = ascending[::-1]

        self.frame = pd.DataFrame(
            {
                "Исходные значения": source,
                "Сортировка по возрастанию": ascending,
                "Сортировка по убыванию": descending,
            }
        )

    def print_frame(self):
        """Показывает первые строки DataFrame."""
        print("Первые 10 строк набора DataFrame:")
        print(self.frame.head(10).to_string(index=False))
        print()


class Statistic:
    """Вычисляет показатели числового набора."""

    def __init__(self, dataset):
        """Связывает вычисления с объектом Dataset."""
        self.dataset = dataset
        self.values = {}

    def _count_repeated(self):
        """Определяет число различных повторяющихся значений."""
        frequencies = {}

        for value in self.dataset.series:
            if value in frequencies:
                frequencies[value] += 1
            else:
                frequencies[value] = 1

        repeated = 0

        for frequency in frequencies.values():
            if frequency > 1:
                repeated += 1

        return repeated

    def calculate(self):
        """Находит стандартные числовые характеристики."""
        numbers = self.dataset.series.tolist()
        minimum = numbers[0]
        maximum = numbers[0]
        total = 0

        for number in numbers:
            if number < minimum:
                minimum = number
            if number > maximum:
                maximum = number
            total += number

        average = total / len(numbers)
        squared_total = 0.0

        for number in numbers:
            squared_total += (number - average) ** 2

        deviation = math.sqrt(squared_total / len(numbers))
        self.values = {
            "Минимальное значение": minimum,
            "Количество повторяющихся значений": self._count_repeated(),
            "Максимальное значение": maximum,
            "Сумма чисел": total,
            "Среднеквадратическое отклонение": deviation,
        }

    def print_values(self):
        """Выводит рассчитанные показатели."""
        print("Стандартные числовые характеристики набора Series:")

        for title, value in self.values.items():
            if isinstance(value, float):
                print(f"{title}: {value:.4f}")
            else:
                print(f"{title}: {value}")

        print()


class Chart:
    """Строит графики по данным объекта Dataset."""

    def __init__(self, dataset):
        """Сохраняет источник данных для визуализации."""
        self.dataset = dataset

    @staticmethod
    def _round_hundred(value):
        """Округляет допустимое целое число до сотен."""
        numeric = float(value)

        if numeric < LEFT or numeric > RIGHT or not numeric.is_integer():
            return value

        integer = int(numeric)
        rounded = ((abs(integer) + 50) // 100) * 100
        return rounded if integer >= 0 else -rounded

    def _rounded_series(self):
        """Готовит значения для гистограммы."""
        rounded = []

        for value in self.dataset.series:
            rounded.append(self._round_hundred(value))

        return pd.Series(rounded, name="Округленные значения")

    @staticmethod
    def _decorate(title, xlabel, ylabel):
        """Оформляет активную область построения."""
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()

    def draw_source(self):
        """Отображает исходную последовательность."""
        plt.figure(figsize=(12, 5))
        plt.plot(self.dataset.series.index, self.dataset.series.values)
        self._decorate(
            "Линейный график исходного набора Series",
            "Индекс",
            "Значение",
        )
        plt.show()

    def draw_histogram(self):
        """Отображает гистограмму округленных чисел."""
        rounded = self._rounded_series()
        plt.figure(figsize=(12, 5))
        plt.hist(rounded.values, bins=30, edgecolor="black")
        self._decorate(
            "Гистограмма значений, округленных до сотен",
            "Округленное значение",
            "Частота",
        )
        plt.show()

    def draw_sorted(self):
        """Сопоставляет два упорядоченных столбца."""
        plt.figure(figsize=(12, 5))
        plt.plot(
            self.dataset.frame.index,
            self.dataset.frame["Сортировка по возрастанию"],
            label="По возрастанию",
        )
        plt.plot(
            self.dataset.frame.index,
            self.dataset.frame["Сортировка по убыванию"],
            label="По убыванию",
        )
        plt.legend()
        self._decorate(
            "Графики отсортированных значений",
            "Индекс",
            "Значение",
        )
        plt.show()


def main():
    """Выполняет все этапы анализа данных."""
    dataset = Dataset()
    statistic = Statistic(dataset)
    chart = Chart(dataset)

    dataset.generate_series()
    dataset.print_series()
    statistic.calculate()
    statistic.print_values()
    chart.draw_source()
    chart.draw_histogram()
    dataset.build_frame()
    dataset.print_frame()
    chart.draw_sorted()


if __name__ == "__main__":
    main()
