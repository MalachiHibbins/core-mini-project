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

// Forward declaration
void distribute_grid(int N, int iproc, int nproc, int& i0, int& i1);

std::vector <int> CombineGrids(int N, std::vector < int > & local_grid, int i0, int i1, int iproc, int nproc){
  std::vector<int> combined_grid(N*N, 0);
  if (nproc > 1){
    // Gather all local grids to root using MPI_Gatherv
    std::vector<int> recvcounts(nproc);
    std::vector<int> displs(nproc);
    
    // Calculate receive counts and displacements for each process
    for (int p = 0; p < nproc; p++){
      int p_i0, p_i1;
      distribute_grid(N, p, nproc, p_i0, p_i1);
      recvcounts[p] = (p_i1 - p_i0) * N;
      displs[p] = p_i0 * N;
    }
    
    MPI_Allgatherv(local_grid.data(), (i1-i0)*N, MPI_INT, 
                   combined_grid.data(), recvcounts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);
    return combined_grid;
  }
  else{
    // only have 1 MPI task
    return local_grid;
  }
}

void Model(int N, std::vector < int > & old_grid, std::vector < int > & new_grid, int i0, int i1, int iproc, int nproc, std::vector<int> & ghost_row_above, std::vector<int> & ghost_row_below, bool & changes_made){
  // ============================================
  // Update the grid according to the rules of the model for 1 time step.
  // Each process updates only its assigned rows [i0, i1)
  // ============================================
  changes_made = false;
  for (int i=i0;i<i1;i++){
    for (int j=0;j<N;j++){
      // convert to local index (i-i0 for local row storage)
      int local_ind = (i - i0) * N + j;
      int state = old_grid[local_ind];
      
      if (state == empty){
        new_grid[local_ind] = empty;
      }
      else if (state == alive){
        // check neighbours for fire
        bool neighbour_on_fire = false;
        
        // Check up - use ghost row if at boundary
        if (i > 0){
          if (i == i0 && iproc > 0){
            // At top of my rows, check ghost from above
            if (ghost_row_above[j] == burning){
              neighbour_on_fire = true;
            }
          } else if (i > i0){
            // Within my rows
            int local_ind_up = (i - 1 - i0) * N + j;
            if (old_grid[local_ind_up] == burning){
              neighbour_on_fire = true;
            }
          }
        }
        
        // Check down - use ghost row if at boundary
        if (i < N-1){
          if (i == i1-1 && iproc < nproc-1){
            // At bottom of my rows, check ghost from below
            if (ghost_row_below[j] == burning){
              neighbour_on_fire = true;
            }
          } else if (i < i1-1){
            // Within my rows
            int local_ind_down = (i + 1 - i0) * N + j;
            if (old_grid[local_ind_down] == burning){
              neighbour_on_fire = true;
            }
          }
        }
        
        if (j > 0){
          // check left
          int local_ind_left = (i - i0) * N + (j - 1);
          if (old_grid[local_ind_left] == burning){
            neighbour_on_fire = true;
          }
        }     
        if (j < N-1){
          // check right
          int local_ind_right = (i - i0) * N + (j + 1);
          if (old_grid[local_ind_right] == burning){
            neighbour_on_fire = true;
          }
        }  
        
        if (neighbour_on_fire){
          new_grid[local_ind] = burning;
          changes_made = true;
        }
        else{
          new_grid[local_ind] = alive;
        }
      }
      else if (state == burning){
        new_grid[local_ind] = burnt;
        changes_made = true;
      }
      else if (state == burnt){
        new_grid[local_ind] = burnt;
      }
      
    }
  }
  // check if any changes were made across all processes
  MPI_Allreduce(MPI_IN_PLACE, &changes_made, 1, MPI_CXX_BOOL, MPI_LOR, MPI_COMM_WORLD);
}

int GenerateRandomState(float probability){
    // Generates a random state: tree with given probability, otherwise no tree
    float random_value = (float)rand() / RAND_MAX;  // Random float between 0 and 1
    if (random_value < probability) {
        return alive;
    }
    return empty;
}

void SetFirstColumnOnFire(std::vector<int>& grid, int N){
    // Sets the first column of the grid on fire
    for (int i=0;i<N;i++){
        if (grid[i * N] == alive) {
            grid[i * N] = burning;
        }
    }
}

std::vector < int > GenerateGrid(int N, int seed, float probability){
    // Creates the grid with each cell being a tree with given probability. Trees in the top row are set on fire.
    
    std::vector < int > grid(N*N, 0);
    srand(seed);  // Seed once at the start
    for (int i=0;i<N*N;i++){
        grid[i] = GenerateRandomState(probability);
    }
    // set the first column of the grid on fire
    SetFirstColumnOnFire(grid, N);
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
    // read columns in as rows (i.e. first N values are row 0, next N values are row 1 etc.)
    std::vector<int> grid(N * N);
    for (int i = 0; i < N * N; ++i) {
        int col = i / N;  // Which column 
        int row = i % N;  // Which row within that column
        infile >> grid[row * N + col];  // Store transposed (column becomes row)
    }

    // set the first column of the grid on fire
    SetFirstColumnOnFire(grid, N);
    return grid;
}

void DisplayGrid(const std::vector<int>& grid, int N, std::ostream& output = std::cout){
    // Displays the grid transposed (rows become columns, columns become rows)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int state = grid[j * N + i];  // Transpose: swap i and j indices
            char display_char = (state == empty) ? '.' : (state == alive) ? 'T' : (state == burning) ? 'B' : 'X';
            output << display_char << " ";
        }
        output << std::endl;
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
    std::vector<int> states_current;
    int N;
    bool draw_grid = false;
    std::string run_description;  // For filename generation
    int seed = 0;
    float probability = 0.0;
    std::string input_filename = "";
    std::vector<int> my_states_current;

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
        seed = atoi(argv[2]);
        probability = atof(argv[3]);
        
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
        states_current = GenerateGrid(N, seed, probability);
        run_description = "N" + std::to_string(N) + "_seed" + std::to_string(seed) + "_prob" + std::to_string((int)(probability*100));
        }

      else{
        // Read grid from file
        input_filename = argv[1];
        states_current = ReadGridFromFile(input_filename, N);
        // Extract filename without path and extension
        size_t last_slash = input_filename.find_last_of("/\\");
        size_t last_dot = input_filename.find_last_of(".");
        std::string base_name = input_filename.substr(last_slash + 1, last_dot - last_slash - 1);
        run_description = "file_" + base_name;
      }
    }

    if (argc == 5){
      draw_grid = atoi(argv[4]);
    }
    else if (argc == 3){
      draw_grid = atoi(argv[2]);
    }

    double setup_end = MPI_Wtime();
    double broadcast_start = MPI_Wtime();
    
    // =========================================
    // Broadcast entire grid size to all processes
    // =========================================

    // broadcast the data
    if (nproc > 1){
      // first share the image size
      MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // =========================================
    // Distribute the grid among processes
    // =========================================

    // divide the rows among MPI tasks - must be done AFTER N is broadcast
    int i0, i1;
    distribute_grid(N, iproc, nproc, i0, i1);

    // Send/receive the relevant rows and reshape to local storage
    if (nproc > 1){
      if (iproc == 0){
        // Process 0: keep full grid temporarily to send to others
        std::vector<int> full_grid = states_current;
        
        // Send relevant rows to each other process
        for (int p = 1; p < nproc; p++){
          int p_i0, p_i1;
          distribute_grid(N, p, nproc, p_i0, p_i1);
          MPI_Send(&full_grid[p_i0 * N], (p_i1 - p_i0) * N, MPI_INT, p, 0, MPI_COMM_WORLD);
        }
        
        // Now extract process 0's own local rows
        states_current.resize((i1 - i0) * N);
        for (int i = i0; i < i1; i++){
          for (int j = 0; j < N; j++){
            states_current[(i - i0) * N + j] = full_grid[i * N + j];
          }
        }
      } else {
        // Other processes: allocate space and receive their rows
        states_current.resize((i1 - i0) * N);
        MPI_Recv(states_current.data(), (i1 - i0) * N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }

    double broadcast_end = MPI_Wtime();
    double distribute_time_start = MPI_Wtime();

    // =========================================
    // Run the simulation
    // =========================================
    
    int burning_steps = 0;
    bool bottom_reached = false;

    double distribute_time_end = MPI_Wtime();
    double model_start = MPI_Wtime();

    
    // check that the tasks stay in sync
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> states_new((i1 - i0) * N, 0); // Only store local rows
    std::vector<int> ghost_row_above(N, 0);
    std::vector<int> ghost_row_below(N, 0);
    MPI_Status status;  // Declare status variable for MPI_Sendrecv
    bool changes_made = true;

    while (changes_made) {
        // send ghost rows 
      if (iproc > 0){
          // Send first local row, receive ghost from above
          MPI_Sendrecv(&states_current[0], N, MPI_INT, iproc-1, 0, ghost_row_above.data(), N, MPI_INT, iproc-1, 1, MPI_COMM_WORLD, &status);
      }
      if (iproc < nproc-1){
          // Send last local row, receive ghost from below
          MPI_Sendrecv(&states_current[(i1-i0-1) * N], N, MPI_INT, iproc+1, 1, ghost_row_below.data(), N, MPI_INT, iproc+1, 0, MPI_COMM_WORLD, &status);
      }
        burning_steps++;
        Model(N, states_current, states_new, i0, i1, iproc, nproc, ghost_row_above, ghost_row_below, changes_made);
        if (draw_grid){
          std::vector<int> combined_grid = CombineGrids(N, states_new, i0, i1, iproc, nproc);
          if (iproc == 0){
            // Display the grid
            if (iproc == 0){
              std::cout << "=========================================" << std::endl;
              std::cout << "Step " << burning_steps << ":" << std::endl;
              std::cout << "=========================================" << std::endl;
            }
            DisplayGrid(combined_grid, N);
          }
        }
      states_current = states_new; 
    }
    double model_end = MPI_Wtime();

    // check that the tasks stay in sync
    MPI_Barrier(MPI_COMM_WORLD);

    // Get Final Results
    std::vector<int> final_grid = CombineGrids(N, states_current, i0, i1, iproc, nproc);

    // check if bottom row on fire
    if (iproc == 0){
      for (int j=0;j<N;j++){
          if (final_grid[(N-1)*N + j] == burning || final_grid[(N-1)*N + j] == burnt){
              bottom_reached = true;
              break;
          }
      }
    }

    // =========================================
    // Output timings
    // =========================================

    if (iproc == 0){
      // Output results to console
      double end = MPI_Wtime();
      std::cout << "Burning steps: " << burning_steps << std::endl;
      std::cout << "Bottom reached: " << (bottom_reached ? "Yes" : "No") << std::endl;
      std::cout << "Execution time: " << end - start << " seconds" << std::endl;
      
      // Write timings to file
      std::string timing_filename = "timings_" + run_description + "_" + std::to_string(nproc) + "proc.txt";
      std::ofstream timing_file(timing_filename);
      if (timing_file.is_open()){
        timing_file << "Run: " << run_description << std::endl;
        timing_file << "Processors: " << nproc << std::endl;
        timing_file << "Grid size: " << N << "x" << N << std::endl;
        timing_file << "Burning steps: " << burning_steps << std::endl;
        timing_file << "Bottom reached: " << (bottom_reached ? "Yes" : "No") << std::endl;
        timing_file << "Total execution time: " << end - start << " seconds" << std::endl;
        timing_file << "Setup time: " << setup_end - start << " seconds" << std::endl;
        timing_file << "Broadcast time: " << broadcast_end - broadcast_start << " seconds" << std::endl;
        timing_file << "Distribute time: " << distribute_time_end - distribute_time_start << " seconds" << std::endl;
        timing_file << "Model time: " << model_end - model_start << " seconds" << std::endl;
        timing_file.close();
        std::cout << "Timings written to " << timing_filename << std::endl;
      }

      // =========================================
      // Output Grid
      // =========================================
      
      // Write grid data to file
      std::string data_filename = "data_" + run_description + "_" + std::to_string(nproc) + "proc.txt";
      std::ofstream data_file(data_filename);
      if (data_file.is_open()){
        data_file << "Final grid (" << N << "x" << N << ")" << std::endl;
        // Write grid in transposed format
        DisplayGrid(final_grid, N, data_file);
        data_file.close();
        std::cout << "Grid data written to " << data_filename << std::endl;
      }
    }

    // =========================================
    // Display final grid to console if requested
    // =========================================
    
    if (iproc == 0 && draw_grid){
      std::cout << "Final grid:" << std::endl;
      DisplayGrid(final_grid, N);
    }

    // finalise MPI
    MPI_Finalize();
    return 0;
}

