#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <omp.h>
#include <windows.h>

using namespace std;
using namespace std::chrono;

void writeMatrixToFile(const float* mat, int size, const string& filename)
{
    ofstream fout(filename);
    if (!fout.is_open()) return;
    fout << size << "\n";
    fout << fixed << setprecision(10);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fout << mat[i * size + j];
            if (j < size - 1) fout << " ";
        }
        fout << "\n";
    }
    fout.close();
}

void parallelMatMul(const float* A, const float* B, float* C, int N, int numThreads)
{
#pragma omp parallel for collapse(2) schedule(dynamic) num_threads(numThreads)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

int main()
{
    cout << "OpenMP версия: " << _OPENMP << endl;
    cout << "Макс. потоков: " << omp_get_max_threads() << endl;
    SetConsoleOutputCP(1251);
    SetConsoleCP(1251);

    vector<int> sizes = { 200, 400, 800, 1200, 1600, 2000 };
    vector<int> threadCounts = { 1, 2, 4, 8 };

    ofstream timeFile("timings_omp.csv");
    if (!timeFile.is_open()) {
        cerr << "Ошибка создания timings_omp.csv" << endl;
        return 1;
    }
    timeFile << "N,threads,time_sec" << endl;

    cout << "Расчёт матриц." << endl;

    for (int N : sizes)
    {
        float* A = new float[N * N];
        float* B = new float[N * N];
        float* C = new float[N * N];

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                A[i * N + j] = (i + 1) * (j + 1);
                B[i * N + j] = (i + 1) + 2 * (j + 1);
            }

        writeMatrixToFile(A, N, "A_" + to_string(N) + ".txt");
        writeMatrixToFile(B, N, "B_" + to_string(N) + ".txt");

        for (int threads : threadCounts)
        {
            auto start = high_resolution_clock::now();
            parallelMatMul(A, B, C, N, threads);
            auto end = high_resolution_clock::now();

            double seconds = duration<double>(end - start).count();

            string cFilename = "C_" + to_string(N) + "_" + to_string(threads) + ".txt";
            writeMatrixToFile(C, N, cFilename);

            timeFile << N << "," << threads << "," << seconds << endl;
        }

        delete[] A; delete[] B; delete[] C;
        cout << "Обработка N=" << N << " завершена." << endl;
    }

    timeFile.close();
    cout << "Готово. Результаты сохранены." << endl;
    return 0;
}