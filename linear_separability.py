"""
Лабораторна робота №1
Визначення лінійної розділюваності двох наборів точок на площині.

Алгоритм: Опукла оболонка (Convex Hull) + перевірка перетину двох опуклих многокутників.
Якщо опуклі оболонки двох наборів точок не перетинаються — набори лінійно розділювані.
"""

import random
import time
import json
import sys
import os
import threading
from math import inf


# ─────────────────────────── Геометричні примітиви ───────────────────────────

def cross(O, A, B):
    """Векторний добуток OA × OB."""
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])


def convex_hull(points):
    """Алгоритм Ендрю (Andrew's monotone chain). O(n log n)."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 1:
        return pts
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def segments_intersect(p1, p2, p3, p4):
    """Перевірка перетину відрізків p1p2 та p3p4."""
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    def on_segment(p, q, r):
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    if d1 == 0 and on_segment(p3, p1, p4): return True
    if d2 == 0 and on_segment(p3, p2, p4): return True
    if d3 == 0 and on_segment(p1, p3, p2): return True
    if d4 == 0 and on_segment(p1, p4, p2): return True
    return False


def point_in_convex_polygon(point, hull):
    """Чи лежить точка всередині або на межі опуклого многокутника."""
    n = len(hull)
    if n == 0:
        return False
    if n == 1:
        return point == hull[0]
    if n == 2:
        # відрізок
        c = cross(hull[0], hull[1], point)
        if c != 0:
            return False
        return (min(hull[0][0], hull[1][0]) <= point[0] <= max(hull[0][0], hull[1][0]) and
                min(hull[0][1], hull[1][1]) <= point[1] <= max(hull[0][1], hull[1][1]))
    for i in range(n):
        if cross(hull[i], hull[(i + 1) % n], point) < 0:
            return False
    return True


def hulls_intersect(hull_a, hull_b):
    """
    Перевірка перетину двох опуклих многокутників:
    1. Перевіряємо перетин ребер.
    2. Перевіряємо вкладеність одного в інший.
    """
    na, nb = len(hull_a), len(hull_b)
    if na == 0 or nb == 0:
        return False

    # Перевірка перетину ребер
    for i in range(na):
        for j in range(nb):
            if segments_intersect(hull_a[i], hull_a[(i + 1) % na],
                                   hull_b[j], hull_b[(j + 1) % nb]):
                return True

    # Перевірка вкладеності
    if point_in_convex_polygon(hull_a[0], hull_b):
        return True
    if point_in_convex_polygon(hull_b[0], hull_a):
        return True

    return False


# ─────────────────────── Послідовний алгоритм ────────────────────────────────

def is_linearly_separable_sequential(set_a, set_b):
    """
    Повертає True, якщо два набори точок лінійно розділювані.
    Послідовна реалізація.
    """
    hull_a = convex_hull(set_a)
    hull_b = convex_hull(set_b)
    return not hulls_intersect(hull_a, hull_b)


# ─────────────────────── Паралельний алгоритм ────────────────────────────────

def is_linearly_separable_parallel(set_a, set_b):
    """
    Паралельна реалізація: опуклі оболонки будуються в окремих потоках.
    """
    hull_a = [None]
    hull_b = [None]

    def compute_a():
        hull_a[0] = convex_hull(set_a)

    def compute_b():
        hull_b[0] = convex_hull(set_b)

    t1 = threading.Thread(target=compute_a)
    t2 = threading.Thread(target=compute_b)
    t1.start(); t2.start()
    t1.join(); t2.join()

    return not hulls_intersect(hull_a[0], hull_b[0])


# ─────────────────────── Генерація даних ─────────────────────────────────────

def generate_separable(n, margin=5.0, coord_range=100.0):
    """Генерує два лінійно розділювані набори точок."""
    set_a = [(random.uniform(-coord_range, -margin),
              random.uniform(-coord_range, coord_range)) for _ in range(n)]
    set_b = [(random.uniform(margin, coord_range),
              random.uniform(-coord_range, coord_range)) for _ in range(n)]
    return set_a, set_b


def generate_nonseparable(n, coord_range=50.0):
    """Генерує два перемішані (не розділювані) набори точок."""
    set_a = [(random.uniform(-coord_range, coord_range),
              random.uniform(-coord_range, coord_range)) for _ in range(n)]
    set_b = [(random.uniform(-coord_range, coord_range),
              random.uniform(-coord_range, coord_range)) for _ in range(n)]
    return set_a, set_b


def generate_random(n, coord_range=100.0):
    """Довільна генерація (результат невизначений заздалегідь)."""
    set_a = [(random.uniform(-coord_range, coord_range),
              random.uniform(-coord_range, coord_range)) for _ in range(n)]
    set_b = [(random.uniform(-coord_range, coord_range),
              random.uniform(-coord_range, coord_range)) for _ in range(n)]
    return set_a, set_b


# ─────────────────────── Збереження / зчитування ─────────────────────────────

def save_to_file(filename, set_a, set_b):
    data = {"set_a": set_a, "set_b": set_b}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Дані збережено у файл: {filename}")


def load_from_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    set_a = [tuple(p) for p in data["set_a"]]
    set_b = [tuple(p) for p in data["set_b"]]
    print(f"Дані завантажено з файлу: {filename}  (|A|={len(set_a)}, |B|={len(set_b)})")
    return set_a, set_b


# ─────────────────────── Виведення результатів ───────────────────────────────

def print_points(label, pts, max_show=10):
    print(f"  {label} ({len(pts)} точок):")
    for p in pts[:max_show]:
        print(f"    ({p[0]:.3f}, {p[1]:.3f})")
    if len(pts) > max_show:
        print(f"    ... (показано {max_show} з {len(pts)})")


def run_and_time(func, set_a, set_b, label):
    t0 = time.perf_counter()
    result = func(set_a, set_b)
    elapsed = time.perf_counter() - t0
    status = "РОЗДІЛЮВАНІ" if result else "НЕ РОЗДІЛЮВАНІ"
    print(f"  [{label}]  Результат: {status}  |  Час: {elapsed:.6f} с")
    return result, elapsed


# ─────────────────────────────── main ────────────────────────────────────────

def demo_small():
    """Контрольний приклад з невеликим об'ємом даних."""
    print("=" * 60)
    print("КОНТРОЛЬНИЙ ПРИКЛАД (малий розмір, ручна перевірка)")
    print("=" * 60)

    # Розділювані: A ліворуч від x=-2, B праворуч від x=2
    set_a = [(-5, 0), (-3, 2), (-4, -1), (-6, 3)]
    set_b = [(3, 0),  (5, 1),  (4, -2),  (6, 2)]
    print("Набір A:", set_a)
    print("Набір B:", set_b)
    res = is_linearly_separable_sequential(set_a, set_b)
    print(f"Очікується: РОЗДІЛЮВАНІ  →  Результат: {'РОЗДІЛЮВАНІ' if res else 'НЕ РОЗДІЛЮВАНІ'}")

    print()
    # Не розділювані: точки перемішані
    set_c = [(-1, 0), (1, 1), (0, -1)]
    set_d = [(0, 0), (-1, 1), (1, -1)]
    print("Набір C:", set_c)
    print("Набір D:", set_d)
    res2 = is_linearly_separable_sequential(set_c, set_d)
    print(f"Очікується: НЕ РОЗДІЛЮВАНІ  →  Результат: {'РОЗДІЛЮВАНІ' if res2 else 'НЕ РОЗДІЛЮВАНІ'}")
    print()


def demo_performance(n=1_000_000):
    """Вимірювання часу на великому наборі (~5 секунд)."""
    print("=" * 60)
    print(f"ВИМІРЮВАННЯ ПРОДУКТИВНОСТІ  (n={n} точок у кожному наборі)")
    print("=" * 60)

    random.seed(42)
    set_a, set_b = generate_random(n)

    print("Послідовний алгоритм:")
    res_seq, t_seq = run_and_time(is_linearly_separable_sequential, set_a, set_b, "sequential")

    print("Паралельний алгоритм (2 потоки):")
    res_par, t_par = run_and_time(is_linearly_separable_parallel, set_a, set_b, "parallel  ")

    print()
    if t_seq > 0:
        speedup = t_seq / t_par
        print(f"  Прискорення (speedup): {speedup:.2f}x")
    print()


def interactive_menu():
    """Інтерактивний режим."""
    print("\n" + "=" * 60)
    print("  ПЕРЕВІРКА ЛІНІЙНОЇ РОЗДІЛЮВАНОСТІ ДВОХ НАБОРІВ ТОЧОК")
    print("=" * 60)
    print("1. Контрольний приклад (малий)")
    print("2. Згенерувати РОЗДІЛЮВАНІ набори та перевірити")
    print("3. Згенерувати НЕ РОЗДІЛЮВАНІ набори та перевірити")
    print("4. Завантажити з файлу та перевірити")
    print("5. Вимірювання продуктивності (великий набір)")
    print("0. Вийти")
    print("-" * 60)

    while True:
        choice = input("Оберіть пункт: ").strip()

        if choice == "0":
            break

        elif choice == "1":
            demo_small()

        elif choice in ("2", "3"):
            n = int(input("Кількість точок у кожному наборі: "))
            random.seed(int(time.time()))
            if choice == "2":
                set_a, set_b = generate_separable(n)
            else:
                set_a, set_b = generate_nonseparable(n)

            save = input("Зберегти у файл? (y/n): ").strip().lower()
            if save == "y":
                fname = input("Ім'я файлу (напр. data.json): ").strip()
                save_to_file(fname, set_a, set_b)

            print("\nПослідовний алгоритм:")
            run_and_time(is_linearly_separable_sequential, set_a, set_b, "sequential")
            print("Паралельний алгоритм:")
            run_and_time(is_linearly_separable_parallel, set_a, set_b, "parallel  ")
            print()

        elif choice == "4":
            fname = input("Ім'я файлу: ").strip()
            if not os.path.exists(fname):
                print(f"Файл '{fname}' не знайдено.")
                continue
            set_a, set_b = load_from_file(fname)
            print("\nПослідовний алгоритм:")
            run_and_time(is_linearly_separable_sequential, set_a, set_b, "sequential")
            print("Паралельний алгоритм:")
            run_and_time(is_linearly_separable_parallel, set_a, set_b, "parallel  ")
            print()

        elif choice == "5":
            n = int(input("Кількість точок (рекомендовано 200000+): "))
            demo_performance(n)

        else:
            print("Невідомий пункт.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        # Автоматичний запуск для звіту
        demo_small()
        demo_performance(n=1_000_000)
    else:
        interactive_menu()