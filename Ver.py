import os
import glob
import numpy as np
import pandas as pd

def load_matrix(filename):
    """Читает матрицу из файла формата: первая строка - N, далее N строк данных."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    data = []
    for line in lines[1:]:
        data.extend([float(x) for x in line.split()])
    return np.array(data, dtype=np.float64).reshape(n, n)

def main():
    c_files = glob.glob("C_*_*.txt")
    if not c_files:
        print("Ошибка: не найдены файлы результатов C_*_*.txt")
        return

    results = []

    for c_file in c_files:
        # Парсим имя файла: C_200_4.txt -> N=200, threads=4
        base = os.path.basename(c_file).replace('.txt', '')
        parts = base.split('_')
        if len(parts) != 3:
            continue
        n = int(parts[1])
        threads = int(parts[2])

        a_file = f"A_{n}.txt"
        b_file = f"B_{n}.txt"

        if not (os.path.exists(a_file) and os.path.exists(b_file)):
            print(f"Пропущен {c_file}: отсутствуют A_{n}.txt или B_{n}.txt")
            continue

        # Загрузка матриц
        A = load_matrix(a_file)
        B = load_matrix(b_file)
        C_cpp = load_matrix(c_file)

        # Эталонный расчёт (двойная точность)
        C_ref = np.dot(A, B)

        # Вычисление погрешностей
        abs_diff = np.abs(C_ref - C_cpp)
        max_abs = np.max(abs_diff)
        mean_abs = np.mean(abs_diff)

        # Относительная погрешность (защита от деления на 0)
        rel_diff = abs_diff / (np.abs(C_ref) + 1e-15)
        max_rel = np.max(rel_diff)

        results.append({
            'N': n,
            'threads': threads,
            'max_abs_error': max_abs,
            'mean_abs_error': mean_abs,
            'max_rel_error': max_rel
        })

    # Формирование таблицы
    df = pd.DataFrame(results)
    df = df.sort_values(by=['N', 'threads']).reset_index(drop=True)

    # Настройка вывода
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.4e}'.format)

    print("\n=== Таблица погрешностей ===")
    print(df.to_string(index=False))

    # Сохранение в CSV
    df.to_csv("table_res_omp.csv", index=False)
    print("\nРезультаты сохранены в table_res_omp.csv")

if __name__ == "__main__":
    main()
