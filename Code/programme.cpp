#include <time.h>
#include <vector>
#include <stdlib.h>
#include <fstream>  
#include <iostream>
#include <cmath>
//#include <mpi.h>

int empty = 0;
int alive = 1;
int burning = 2;
int burnt = 3;


int Get1dIndex(int i, int j, int k, int N){
  return i * N + j;
}

void Model(int N, std::vector < int > & old_grid, std::vector < int > & new_grid){

  for (int i=0;i<N;i++){
    for (int j=0;j<N;j++){
       
      int ind = Get1dIndex(i, j, 0, N);
      int state = old_grid[ind];
      
      if (state == empty){
        new_grid[ind] = empty;
      }
      else if (state == alive){
        // check neighbours for fire
        bool neighbour_on_fire = false;
        if (i > 0){
          int ind_up = Get1dIndex(i-1, j, 0, N);
          if (old_grid[ind_up] == burning){
            neighbour_on_fire = true;
          }
        }
        if (i < N-1){
          int ind_down = Get1dIndex(i+1, j, 0, N);
          if (old_grid[ind_down] == burning){
            neighbour_on_fire = true;
          }
        }     
        if (j > 0){
          int ind_left = Get1dIndex(i, j-1, 0, N);
          if (old_grid[ind_left] == burning){
            neighbour_on_fire = true;
          }
        }     
        if (j < N-1){
          int ind_right = Get1dIndex(i, j+1, 0, N);
          if (old_grid[ind_right] == burning){
            neighbour_on_fire = true;
          }
        }  
        
        if (neighbour_on_fire){
          new_grid[ind] = burning;
        }
        else{
          new_grid[ind] = alive;
        }
      }
      else if (state == burning){
        new_grid[ind] = burnt;
      }
       
    }
  }

}

int GenerateRandomState(float probability){
    // Generates a random state: tree with given probability, otherwise no tree
    float random_value = (float)rand() / RAND_MAX;  // Random float between 0 and 1
    if (random_value < probability) {
        return alive;
    }
    return empty;
}


std::vector < int > GenerateGrid(int N, int seed, float probability){
    // Creates the grid
    std::vector < int > grid(N*N, 0);
    srand(seed);  // Seed once at the start
    for (int i=0;i<N*N;i++){
        grid[i] = GenerateRandomState(probability);
    }
    for (int i=0;i<N;i++){
        if (grid[i] == alive) {
            grid[i] = burning;
        }
    }
    return grid;
}

void DisplayGrid(const std::vector<int>& grid, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int state = grid[i * N + j];
            char display_char = (state == empty) ? '.' : (state == alive) ? 'T' : (state == burning) ? 'B' : 'X';
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
    
    std::vector<int> states_old = GenerateGrid(N, seed, probability);
    DisplayGrid(states_old, N);
    std::vector<int> states_new(N*N, 0);
    for (int iter=0;iter<N*N/2;iter++){
        Model(N, states_old, states_new);
        std::cout << "After iteration " << iter+1 << ":" << std::endl;
        if (states_new == states_old) {
            std::cout << "No changes in the grid, stopping simulation." << std::endl;
            break;
        }
        DisplayGrid(states_new, N);
        states_old = states_new;  // Update for the next iteration
    }
    return 0;
}

