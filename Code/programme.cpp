#include <time.h>
#include <vector>
#include <stdlib.h>
#include <fstream>  
#include <iostream>
#include <cmath>
#include <mpi.h>

int empty = 0;
int alive = 1;
int burning = 2;
int burnt = 3;


int Get1dIndex(int i, int j, int k, int N){
  // converts 2D index to 1D index
  return i * N + j;
}

void Model(int N, std::vector < int > & old_grid, std::vector < int > & new_grid, int j0, int j1, int iproc, int nproc){
  // ============================================
  // Update the grid according to the rules of the model for 1 time step.
  // ============================================
  // Initialize to zeros - each process updates only its assigned cells  
  for (int i=0;i<N;i++){
    for (int j=j0;j<j1;j++){
      // convert 2D index to 1D index
      int ind = Get1dIndex(i, j, 0, N);
      int state = old_grid[ind];
      
      if (state == empty){
        new_grid[ind] = empty;
      }
      else if (state == alive){
        // check neighbours for fire
        bool neighbour_on_fire = false;
        if (i > 0){
          // check up
          int ind_up = Get1dIndex(i-1, j, 0, N);
          if (old_grid[ind_up] == burning){
            neighbour_on_fire = true;
          }
        }
        if (i < N-1){
          // check down
          int ind_down = Get1dIndex(i+1, j, 0, N);
          if (old_grid[ind_down] == burning){
            neighbour_on_fire = true;
          }
        }     
        if (j > 0){
          // check left
          int ind_left = Get1dIndex(i, j-1, 0, N);
          if (old_grid[ind_left] == burning){
            neighbour_on_fire = true;
          }
        }     
        if (j < N-1){
          // check right
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
      else if (state == burnt){
        new_grid[ind] = burnt;
      }
       
    }
  }
  // ============================================
  // Communicate the updated grid: combine each process's
  // `new_grid` (local updates) into `old_grid` (global state).
  // ============================================
  if (nproc > 1){
    // combine all new_grid locals and store the result in old_grid
    // use SUM since each cell is updated by exactly one process (others leave it at 0)
    MPI_Allreduce(new_grid.data(), old_grid.data(), N*N, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  }
  else{
    old_grid = new_grid;
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
    // Creates the grid with each cell being a tree with given probability. Trees in the top row are set on fire.
    
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

std::vector<int> ReadGridFromFile(const std::string filename, int& N) {
    // ============================================
    // Read in grid file separated by spaces.
    // ============================================

    // check file exists
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        exit(1);
    }

    N = 6;
    // read grid values into vector
    std::vector<int> grid(N * N);
    for (int i = 0; i < N * N; ++i) {
        infile >> grid[i];
    }

    // set the first row of the grid on fire
    for (int j = 0; j < N; ++j) {
        if (grid[j] == alive) {
            grid[j] = burning;
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

void distribute_grid(int N, int iproc, int nproc, int& i0, int& i1){
  // ============================================
  // Distribute the grid among processes by dividing into 2D slices. Each process gets a contiguous block
  // ============================================
  int n_slices = N / nproc;
  int remainder = N % nproc;
  if (iproc < remainder){
    i0 = iproc * (n_slices + 1);
    i1 = i0 + n_slices + 1;
  } else {
    i0 = iproc * n_slices + remainder;
    i1 = i0 + n_slices;
  }
}

int main(int argc, char* argv[]){

    // =========================================
    // Initialise MPI
    // =========================================

    // initialise MPI
    MPI_Init(&argc, &argv);

    // Get the number of processes in MPI_COMM_WORLD
    int nproc;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // get the rank of this process in MPI_COMM_WORLD
    int iproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);

    // =========================================
    // Organise inputs and generate/load grid on root process
    // =========================================

    double start = MPI_Wtime();
    std::vector<int> states_old;
    int N;
    bool draw_grid = false;

    if (iproc == 0) {
        std::cout << "Running with " << nproc << " MPI processes" << std::endl;
    
      // Argument parsing
      if (argc != 5 && argc != 3) {
          std::cerr << "Error: Incorrect number of arguments." << std::endl;
          std::cerr << "Usage 1 (random grid mode): " << argv[0] << " <N> <seed> <probability> <draw_grid>" << std::endl;
          std::cerr << "Usage 2 (read file mode): " << argv[0] << " <filename> <draw_grid>" << std::endl;
          return 1;
      }

      else if (argc == 5){

        N = atoi(argv[1]);
        int seed = atoi(argv[2]);
        float probability = atof(argv[3]);
        draw_grid = atoi(argv[4]) != 0;
        
        // Input validation
        if (N <= 0) {
            std::cerr << "Error: N must be a positive integer (got " << N << ")" << std::endl;
            return 1;
        }
        
        if (probability < 0.0 || probability > 1.0) {
            std::cerr << "Error: probability must be between 0.0 and 1.0 (got " << probability << ")" << std::endl;
            return 1;
        }
        
        // Generate initial grid from random
        states_old = GenerateGrid(N, seed, probability);
        }

      else{
        // Read grid from file
        std::string filename = argv[1];
        states_old = ReadGridFromFile(filename, N);
        draw_grid = atoi(argv[2]) != 0;
      }
    }

    // =========================================
    // Broadcast entire grid size to all processes
    // =========================================

    // broadcast the data
    if (nproc > 1){
      // first share the image size
      MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
      // resize the vector on non-root processes
      if (iproc != 0){
        states_old.resize(N*N);
      }
      // now broadcast the actual grid data
      MPI_Bcast(states_old.data(), N*N, MPI_INT, 0, MPI_COMM_WORLD);      
    }

    // =========================================
    // Distribute the grid among processes
    // =========================================

    // divide the rows among MPI tasks
    int i0, i1;
    distribute_grid(N, iproc, nproc, i0, i1);

    // =========================================
    // Run the simulation
    // =========================================
    
    int burning_steps = 0;
    bool bottom_reached = false;

    // run simulation until no changes occur or max iterations reached
    // check that the tasks stay in sync
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> states_new(N*N, 0);
    while (true) {
        burning_steps++;
        Model(N, states_old, states_new, i0, i1, iproc, nproc);
        if (burning_steps == N) {
          break;
        }
        if (draw_grid && iproc == 0) {
          std::cout << "=========================================" << std::endl;
          std::cout << "Step " << burning_steps << ":" << std::endl;
          std::cout << "=========================================" << std::endl;
          DisplayGrid(states_old, N);
        }
    }

    // check that the tasks stay in sync
    MPI_Barrier(MPI_COMM_WORLD);

    // check if bottom row on fire
    if (iproc == 0){
      for (int j=0;j<N;j++){
          if (states_old[(N-1)*N + j] == burning || states_old[(N-1)*N + j] == burnt){
              bottom_reached = true;
              break;
          }
      }
    }

    // =========================================
    // Output results
    // =========================================

    if (iproc == 0){
      // Output results
      double end = MPI_Wtime();
      std::cout << "Burning steps: " << burning_steps << std::endl;
      std::cout << "Bottom reached: " << (bottom_reached ? "Yes" : "No") << std::endl;
      std::cout << "Execution time: " << end - start << " seconds" << std::endl;
    }
    
    // finalise MPI
    MPI_Finalize();
    return 0;
}

