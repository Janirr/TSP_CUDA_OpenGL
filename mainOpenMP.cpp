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
#include <omp.h>

#define POPULATION_SIZE 100
#define TOURNAMENT_SIZE 5
#define GENERATIONS 1000
#define MUTATION_PROB 0.02
#define FILENAME "tsp_matrix.txt"

int getRandomInt(int min, int max) {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min, max);
    return dis(gen);
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

int calculateRouteCost(const std::vector<int>& path, const std::vector<std::vector<int>>& matrix) {
    int totalCost = 0;
    for (size_t i = 0; i < path.size() - 1; ++i) {
        totalCost += matrix[path[i]][path[i + 1]];
    }
    totalCost += matrix[path.back()][path.front()];
    return totalCost;
}

void calculateRouteCost(const std::vector<std::vector<int>>& population, const std::vector<std::vector<int>>& matrix, std::vector<int>& costs) {
    int populationSize = population.size();
    #pragma omp parallel for
    for (int i = 0; i < populationSize; ++i) {
        costs[i] = calculateRouteCost(population[i], matrix);
    }
}

void runTest(int size) {
    generateRandomMatrix(FILENAME, size);

    std::vector<std::vector<int>> matrix(size, std::vector<int>(size));
    std::ifstream file(FILENAME);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << FILENAME << std::endl;
        return;
    }

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            file >> matrix[i][j];
        }
    }
    file.close();

    int numCities = size;
    std::vector<std::vector<int>> population(POPULATION_SIZE, std::vector<int>(numCities));
    std::vector<int> costs(POPULATION_SIZE);

    auto start = std::chrono::high_resolution_clock::now();

    // Initialize population
    #pragma omp parallel for
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        std::iota(population[i].begin(), population[i].end(), 0);
        std::shuffle(population[i].begin(), population[i].end(), std::default_random_engine{});
    }

    calculateRouteCost(population, matrix, costs);

    for (int gen = 0; gen < GENERATIONS; ++gen) {
        std::vector<std::vector<int>> newPopulation(POPULATION_SIZE, std::vector<int>(numCities));
        std::vector<int> newCosts(POPULATION_SIZE);

        #pragma omp parallel for
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            // Tournament selection
            int bestIdx = getRandomInt(0, POPULATION_SIZE - 1);
            for (int j = 1; j < TOURNAMENT_SIZE; ++j) {
                int idx = getRandomInt(0, POPULATION_SIZE - 1);
                if (costs[idx] < costs[bestIdx]) bestIdx = idx;
            }
            std::vector<int> parent1 = population[bestIdx];

            bestIdx = getRandomInt(0, POPULATION_SIZE - 1);
            for (int j = 1; j < TOURNAMENT_SIZE; ++j) {
                int idx = getRandomInt(0, POPULATION_SIZE - 1);
                if (costs[idx] < costs[bestIdx]) bestIdx = idx;
            }
            std::vector<int> parent2 = population[bestIdx];

            // Crossover
            int start = getRandomInt(0, numCities - 1);
            int end = getRandomInt(0, numCities - 1);
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

            // Mutation
            if (getRandomInt(0, 100) / 100.0 < MUTATION_PROB) {
                int idx1 = getRandomInt(0, numCities - 1);
                int idx2 = getRandomInt(0, numCities - 1);
                std::swap(child[idx1], child[idx2]);
            }

            newPopulation[i] = child;
        }

        calculateRouteCost(newPopulation, matrix, newCosts);
        population = std::move(newPopulation);
        costs = std::move(newCosts);
    }

    auto bestIt = std::min_element(costs.begin(), costs.end());
    int bestIdx = std::distance(costs.begin(), bestIt);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Execution time for " << size << " cities: " << duration.count() << " ms\n";
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

        std::vector<std::vector<int>> matrix;
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::vector<int> row((std::istream_iterator<int>(iss)), std::istream_iterator<int>());
            matrix.push_back(row);
        }
        file.close();

        int numCities = matrix.size();
        std::vector<std::vector<int>> population(POPULATION_SIZE, std::vector<int>(numCities));
        std::vector<int> costs(POPULATION_SIZE);

        auto start = std::chrono::high_resolution_clock::now();

        // Initialize population
        #pragma omp parallel for
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            std::iota(population[i].begin(), population[i].end(), 0);
            std::shuffle(population[i].begin(), population[i].end(), std::default_random_engine{});
        }

        calculateRouteCost(population, matrix, costs);

        for (int gen = 0; gen < GENERATIONS; ++gen) {
            std::vector<std::vector<int>> newPopulation(POPULATION_SIZE, std::vector<int>(numCities));
            std::vector<int> newCosts(POPULATION_SIZE);

            #pragma omp parallel for
            for (int i = 0; i < POPULATION_SIZE; ++i) {
                // Tournament selection
                int bestIdx = getRandomInt(0, POPULATION_SIZE - 1);
                for (int j = 1; j < TOURNAMENT_SIZE; ++j) {
                    int idx = getRandomInt(0, POPULATION_SIZE - 1);
                    if (costs[idx] < costs[bestIdx]) bestIdx = idx;
                }
                std::vector<int> parent1 = population[bestIdx];

                bestIdx = getRandomInt(0, POPULATION_SIZE - 1);
                for (int j = 1; j < TOURNAMENT_SIZE; ++j) {
                    int idx = getRandomInt(0, POPULATION_SIZE - 1);
                    if (costs[idx] < costs[bestIdx]) bestIdx = idx;
                }
                std::vector<int> parent2 = population[bestIdx];

                // Crossover
                int start = getRandomInt(0, numCities - 1);
                int end = getRandomInt(0, numCities - 1);
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

                // Mutation
                if (getRandomInt(0, 100) / 100.0 < MUTATION_PROB) {
                    int idx1 = getRandomInt(0, numCities - 1);
                    int idx2 = getRandomInt(0, numCities - 1);
                    std::swap(child[idx1], child[idx2]);
                }

                newPopulation[i] = child;
            }

            calculateRouteCost(newPopulation, matrix, newCosts);
            population = std::move(newPopulation);
            costs = std::move(newCosts);
        }

        auto bestIt = std::min_element(costs.begin(), costs.end());
        int bestIdx = std::distance(costs.begin(), bestIt);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Execution time: " << duration.count() << " ms\n";
    }

    return 0;
}
