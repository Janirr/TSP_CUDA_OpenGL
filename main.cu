#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <sstream>
#include <fstream>
#include <string>
#include <iterator>
#include <numeric>
#include <curand_kernel.h>

#define POPULATION_SIZE 100
#define TOURNAMENT_SIZE 5
#define GENERATIONS 1000
#define MUTATION_PROB 0.02
#define FILENAME "tsp_matrix.txt"

__device__ int getRandomInt(int min, int max, curandState* state) {
    return min + (int)(curand_uniform(state) * (max - min + 1));
}

void generateRandomMatrix(const std::string& filename, int size) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            file << rand() % 100 + 1 << " ";
        }
        file << "\n";
    }
    file.close();
}

__device__ int calculateRouteCost(const int* path, const int* matrix, int numCities) {
    int totalCost = 0;
    for (int i = 0; i < numCities - 1; ++i) {
        totalCost += matrix[path[i] * numCities + path[i + 1]];
    }
    totalCost += matrix[path[numCities - 1] * numCities + path[0]];
    return totalCost;
}

__global__ void initializePopulation(int* population, int numCities, curandState* states) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < POPULATION_SIZE) {
        curand_init(1234, idx, 0, &states[idx]);
        for (int i = 0; i < numCities; ++i) {
            population[idx * numCities + i] = i;
        }
        for (int i = 0; i < numCities; ++i) {
            int j = getRandomInt(0, numCities - 1, &states[idx]);
            int tmp = population[idx * numCities + i];
            population[idx * numCities + i] = population[idx * numCities + j];
            population[idx * numCities + j] = tmp;
        }
    }
}

__global__ void calculateCosts(int* population, int* matrix, int* costs, int numCities) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < POPULATION_SIZE) {
        costs[idx] = calculateRouteCost(&population[idx * numCities], matrix, numCities);
    }
}

void runTest(int size) {
    generateRandomMatrix(FILENAME, size);

    std::vector<int> matrix(size * size);
    std::ifstream file(FILENAME);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << FILENAME << std::endl;
        return;
    }

    for (int i = 0; i < size * size; ++i) {
        file >> matrix[i];
    }
    file.close();

    int numCities = size;
    std::vector<int> population(POPULATION_SIZE * numCities);
    std::vector<int> costs(POPULATION_SIZE);

    int* d_population;
    int* d_matrix;
    int* d_costs;
    curandState* d_states;

    cudaMalloc(&d_population, POPULATION_SIZE * numCities * sizeof(int));
    cudaMalloc(&d_matrix, numCities * numCities * sizeof(int));
    cudaMalloc(&d_costs, POPULATION_SIZE * sizeof(int));
    cudaMalloc(&d_states, POPULATION_SIZE * sizeof(curandState));

    cudaMemcpy(d_matrix, matrix.data(), numCities * numCities * sizeof(int), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    initializePopulation<<<(POPULATION_SIZE + 255) / 256, 256>>>(d_population, numCities, d_states);
    cudaDeviceSynchronize();

    calculateCosts<<<(POPULATION_SIZE + 255) / 256, 256>>>(d_population, d_matrix, d_costs, numCities);
    cudaDeviceSynchronize();

    cudaMemcpy(population.data(), d_population, POPULATION_SIZE * numCities * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(costs.data(), d_costs, POPULATION_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    for (int gen = 0; gen < GENERATIONS; ++gen) {
        std::vector<int> newPopulation(POPULATION_SIZE * numCities);
        std::vector<int> newCosts(POPULATION_SIZE);

        for (int i = 0; i < POPULATION_SIZE; ++i) {
            int bestIdx = rand() % POPULATION_SIZE;
            for (int j = 1; j < TOURNAMENT_SIZE; ++j) {
                int idx = rand() % POPULATION_SIZE;
                if (costs[idx] < costs[bestIdx]) bestIdx = idx;
            }
            std::vector<int> parent1(population.begin() + bestIdx * numCities, population.begin() + (bestIdx + 1) * numCities);

            bestIdx = rand() % POPULATION_SIZE;
            for (int j = 1; j < TOURNAMENT_SIZE; ++j) {
                int idx = rand() % POPULATION_SIZE;
                if (costs[idx] < costs[bestIdx]) bestIdx = idx;
            }
            std::vector<int> parent2(population.begin() + bestIdx * numCities, population.begin() + (bestIdx + 1) * numCities);

            int start = rand() % numCities;
            int end = rand() % numCities;
            if (start > end) std::swap(start, end);

            std::vector<int> child(numCities, -1);
            for (int k = start; k <= end; ++k) {
                child[k] = parent1[k];
            }

            int currentPos = (end + 1) % numCities;
            for (int gene : parent2) {
                if (std::find(child.begin(), child.end(), gene) == child.end()) {
                    child[currentPos] = gene;
                    currentPos = (currentPos + 1) % numCities;
                }
            }

            if ((rand() % 100) / 100.0 < MUTATION_PROB) {
                int idx1 = rand() % numCities;
                int idx2 = rand() % numCities;
                std::swap(child[idx1], child[idx2]);
            }

            std::copy(child.begin(), child.end(), newPopulation.begin() + i * numCities);
        }

        cudaMemcpy(d_population, newPopulation.data(), POPULATION_SIZE * numCities * sizeof(int), cudaMemcpyHostToDevice);
        calculateCosts<<<(POPULATION_SIZE + 255) / 256, 256>>>(d_population, d_matrix, d_costs, numCities);
        cudaDeviceSynchronize();
        cudaMemcpy(newCosts.data(), d_costs, POPULATION_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

        population = std::move(newPopulation);
        costs = std::move(newCosts);
    }

    auto bestIt = std::min_element(costs.begin(), costs.end());
    int bestIdx = std::distance(costs.begin(), bestIt);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Execution time for " << size << " cities: " << duration.count() << " ms\n";

    cudaFree(d_population);
    cudaFree(d_matrix);
    cudaFree(d_costs);
    cudaFree(d_states);
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "test") {
        std::vector<int> city_sizes = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
        for (int size : city_sizes) {
            runTest(size);
        }
    } else {
        std::ifstream file(FILENAME);
        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << FILENAME << std::endl;
            return 1;
        }

        std::vector<int> matrix;
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            matrix.insert(matrix.end(), std::istream_iterator<int>(iss), std::istream_iterator<int>());
        }
        file.close();

        int numCities = std::sqrt(matrix.size());
        std::vector<int> population(POPULATION_SIZE * numCities);
        std::vector<int> costs(POPULATION_SIZE);

        int* d_population;
        int* d_matrix;
        int* d_costs;
        curandState* d_states;

        cudaMalloc(&d_population, POPULATION_SIZE * numCities * sizeof(int));
        cudaMalloc(&d_matrix, numCities * numCities * sizeof(int));
        cudaMalloc(&d_costs, POPULATION_SIZE * sizeof(int));
        cudaMalloc(&d_states, POPULATION_SIZE * sizeof(curandState));

        cudaMemcpy(d_matrix, matrix.data(), numCities * numCities * sizeof(int), cudaMemcpyHostToDevice);

        auto start = std::chrono::high_resolution_clock::now();

        initializePopulation<<<(POPULATION_SIZE + 255) / 256, 256>>>(d_population, numCities, d_states);
        cudaDeviceSynchronize();

        calculateCosts<<<(POPULATION_SIZE + 255) / 256, 256>>>(d_population, d_matrix, d_costs, numCities);
        cudaDeviceSynchronize();

        cudaMemcpy(population.data(), d_population, POPULATION_SIZE * numCities * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(costs.data(), d_costs, POPULATION_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

        for (int gen = 0; gen < GENERATIONS; ++gen) {
            std::vector<int> newPopulation(POPULATION_SIZE * numCities);
            std::vector<int> newCosts(POPULATION_SIZE);

            for (int i = 0; i < POPULATION_SIZE; ++i) {
                int bestIdx = rand() % POPULATION_SIZE;
                for (int j = 1; j < TOURNAMENT_SIZE; ++j) {
                    int idx = rand() % POPULATION_SIZE;
                    if (costs[idx] < costs[bestIdx]) bestIdx = idx;
                }
                std::vector<int> parent1(population.begin() + bestIdx * numCities, population.begin() + (bestIdx + 1) * numCities);

                bestIdx = rand() % POPULATION_SIZE;
                for (int j = 1; j < TOURNAMENT_SIZE; ++j) {
                    int idx = rand() % POPULATION_SIZE;
                    if (costs[idx] < costs[bestIdx]) bestIdx = idx;
                }
                std::vector<int> parent2(population.begin() + bestIdx * numCities, population.begin() + (bestIdx + 1) * numCities);

                int start = rand() % numCities;
                int end = rand() % numCities;
                if (start > end) std::swap(start, end);

                std::vector<int> child(numCities, -1);
                for (int k = start; k <= end; ++k) {
                    child[k] = parent1[k];
                }

                int currentPos = (end + 1) % numCities;
                for (int gene : parent2) {
                    if (std::find(child.begin(), child.end(), gene) == child.end()) {
                        child[currentPos] = gene;
                        currentPos = (currentPos + 1) % numCities;
                    }
                }

                if ((rand() % 100) / 100.0 < MUTATION_PROB) {
                    int idx1 = rand() % numCities;
                    int idx2 = rand() % numCities;
                    std::swap(child[idx1], child[idx2]);
                }

                std::copy(child.begin(), child.end(), newPopulation.begin() + i * numCities);
            }

            cudaMemcpy(d_population, newPopulation.data(), POPULATION_SIZE * numCities * sizeof(int), cudaMemcpyHostToDevice);
            calculateCosts<<<(POPULATION_SIZE + 255) / 256, 256>>>(d_population, d_matrix, d_costs, numCities);
            cudaDeviceSynchronize();
            cudaMemcpy(newCosts.data(), d_costs, POPULATION_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

            population = std::move(newPopulation);
            costs = std::move(newCosts);
        }

        auto bestIt = std::min_element(costs.begin(), costs.end());
        int bestIdx = std::distance(costs.begin(), bestIt);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Execution time: " << duration.count() << " ms\n";

        cudaFree(d_population);
        cudaFree(d_matrix);
        cudaFree(d_costs);
        cudaFree(d_states);
    }

    return 0;
}
