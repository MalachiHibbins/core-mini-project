#include <time.h>
#include <vector>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <mpi.h>


int no_tree = 0;
int tree = 1;
int fire = 2;

int GenerateRandomState(float probability){
    // Generates a random state: tree with given probability, otherwise no tree
    float random_value = (float)rand() / RAND_MAX;  // Random float between 0 and 1
    if (random_value < probability) {
        return tree;
    }
    return no_tree;
}


std::vector < int > GenerateGrid(int N, int seed, float probability){
    // Creates the grid
    std::vector < int > grid(N*N, 0);
    srand(seed);  // Seed once at the start
    for (int i=0;i<N*N;i++){
        grid[i] = GenerateRandomState(probability);
    }
    return grid;
}

void DisplayGrid(const std::vector<int>& grid, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int state = grid[i * N + j];
            char display_char = (state == no_tree) ? '.' : (state == tree) ? 'T' : 'F';
            std::cout << display_char << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char* argv[]){
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <N> <seed> <probability>" << std::endl;
        return 1;
    }
    
    int N = atoi(argv[1]);
    int seed = atoi(argv[2]);
    float probability = atof(argv[3]);
    
    // Input validation
    if (N <= 0) {
        std::cerr << "Error: N must be a positive integer (got " << N << ")" << std::endl;
        return 1;
    }
    
    if (probability < 0.0 || probability > 1.0) {
        std::cerr << "Error: probability must be between 0.0 and 1.0 (got " << probability << ")" << std::endl;
        return 1;
    }
    
    std::vector<int> states = GenerateGrid(N, seed, probability);
    DisplayGrid(states, N);
    return 0;
}

