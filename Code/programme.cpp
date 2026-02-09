#include <time.h>
#include <vector>
#include <stdlib.h>
#include <fstream>  
#include <iostream>
#include <sstream>
#include <cmath>
#include <mpi.h>

// Cell state constants
enum CellState {
    EMPTY = 0,
    ALIVE = 1,
    BURNING = 2,
    BURNT = 3
};


void distribute_grid(const int N, const int iproc, const int nproc, int& row_start, int& row_end){
  // ============================================
  // Distribute the grid among processes by dividing into 2D slices. Each process gets a contiguous block
  // ============================================
  const int n_slices = N / nproc;
  const int remainder = N % nproc;
  if (iproc < remainder){
    row_start = iproc * (n_slices + 1);
    row_end = row_start + n_slices + 1;
  } else {
    row_start = iproc * n_slices + remainder;
    row_end = row_start + n_slices;
  }
}

std::vector <int> CombineGrids(const int N, const std::vector < int > & local_grid, const int row_start, const int row_end, const int iproc, const int nproc){
  std::vector<int> combined_grid(N*N, 0);
  if (nproc > 1){
    // Gather all local grids to root using MPI_Gatherv
    std::vector<int> recvcounts(nproc);
    std::vector<int> displs(nproc);
    
    // Calculate receive counts and displacements for each process
    for (int p = 0; p < nproc; p++){
      int p_row_start, p_row_end;
      distribute_grid(N, p, nproc, p_row_start, p_row_end);
      recvcounts[p] = (p_row_end - p_row_start) * N;
      displs[p] = p_row_start * N;
    }
    
    MPI_Allgatherv(local_grid.data(), (row_end-row_start)*N, MPI_INT, 
                   combined_grid.data(), recvcounts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);
    return combined_grid;
  }
  else{
    // only have 1 MPI task
    return local_grid;
  }
}

void Model(const int N, const std::vector < int > & old_grid, std::vector < int > & new_grid, const int row_start, const int row_end, const int iproc, const int nproc, const std::vector<int> & ghost_row_above, const std::vector<int> & ghost_row_below, bool & changes_made){
  // ============================================
  // Update the grid according to the rules of the model for 1 time step.
  // Each process updates only its assigned rows 
  // - converts to 1D local indexing for easier storage of local rows
  // - checks if cell is alive and if so checks neighbours for fire (using ghost rows if at boundary)
  // - if cell is burning, becomes burnt
  // - empty and burnt cells remain unchanged
  // ============================================
  changes_made = false;
  for (int i=row_start;i<row_end;i++){
    for (int j=0;j<N;j++){
      // convert to local index (i-row_start for local row storage)
      int local_ind = (i - row_start) * N + j;
      int state = old_grid[local_ind];
      
      if (state == ALIVE){
        // check neighbours for fire
        bool neighbour_on_fire = false;
        
        // Check up - use ghost row if at boundary
        if (!neighbour_on_fire && i > 0){
          if (i == row_start && iproc > 0){
            // At top of my rows, check ghost from above
            if (ghost_row_above[j] == BURNING){
              neighbour_on_fire = true;
            }
          } else if (i > row_start){
            // Within my rows
            int local_ind_up = (i - 1 - row_start) * N + j;
            if (old_grid[local_ind_up] == BURNING){
              neighbour_on_fire = true;
            }
          }
        }
        
        // Check down - use ghost row if at boundary
        if (!neighbour_on_fire && i < N-1){
          if (i == row_end-1 && iproc < nproc-1){
            // At bottom of my rows, check ghost from below
            if (ghost_row_below[j] == BURNING){
              neighbour_on_fire = true;
            }
          } else if (i < row_end-1){
            // Within my rows
            int local_ind_down = (i + 1 - row_start) * N + j;
            if (old_grid[local_ind_down] == BURNING){
              neighbour_on_fire = true;
            }
          }
        }
        
        if (!neighbour_on_fire && j > 0){
          // check left
          int local_ind_left = (i - row_start) * N + (j - 1);
          if (old_grid[local_ind_left] == BURNING){
            neighbour_on_fire = true;
          }
        }     
        if (!neighbour_on_fire && j < N-1){
          // check right
          int local_ind_right = (i - row_start) * N + (j + 1);
          if (old_grid[local_ind_right] == BURNING){
            neighbour_on_fire = true;
          }
        }  
        
        if (neighbour_on_fire){
          new_grid[local_ind] = BURNING;
          changes_made = true;
        }
        else{
          new_grid[local_ind] = ALIVE;
        }
      }
      else if (state == BURNING){
        new_grid[local_ind] = BURNT;
        changes_made = true;
      }
      else {
        // EMPTY and BURNT states remain unchanged
        new_grid[local_ind] = state;
      }
      
    }
  }
  // check if any changes were made across all processes
  MPI_Allreduce(MPI_IN_PLACE, &changes_made, 1, MPI_CXX_BOOL, MPI_LOR, MPI_COMM_WORLD);
}

int GenerateRandomState(const float probability){
    // Generates a random state: tree with given probability, otherwise no tree
    float random_value = (float)rand() / RAND_MAX;  // Random float between 0 and 1
    if (random_value < probability) {
        return ALIVE;
    }
    return EMPTY;
}

void SetFirstColumnOnFire(std::vector<int>& grid, const int N){
    // Sets the first column of the grid on fire
    for (int i=0;i<N;i++){
        if (grid[i * N] == ALIVE) {
            grid[i * N] = BURNING;
        }
    }
}

std::vector < int > GenerateGrid(const int N, const int seed, const float probability){
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
    // - checks file exists
    // - auto-detects grid size (must be square)
    // - reads in values in transposed order (columns become rows) to ensure more even
    // ============================================

    // check file exists
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        exit(1);
    }

    // Auto-detect grid size: count values in first line to get number of columns
    // count cols
    std::string first_line;
    std::getline(infile, first_line);
    std::istringstream iss(first_line);
    int num_cols = 0;
    int temp;
    while (iss >> temp) {
        num_cols++;
    }
    
    // Count rows
    int num_rows = 1; // Already read first line
    std::string line;
    while (std::getline(infile, line)) {
        if (!line.empty()) num_rows++;
    }
    
    // validate grid is square
    if (num_cols != num_rows) {
        std::cerr << "Error: Grid must be square. Found " << num_rows << " rows and " << num_cols << " columns" << std::endl;
        exit(1);
    }
    
    N = num_cols;
    
    // Reset file to beginning and read all values
    infile.clear();
    infile.seekg(0);
    
    // read grid values into vector
    // read columns in as rows (i.e. first N values are row 0, next N values are row 1 etc.)
    std::vector<int> grid(N * N);
    for (int i = 0; i < N * N; ++i) {
        int col = i / N;   
        int row = i % N;  
        infile >> grid[row * N + col];  // Store transposed (column becomes row)
    }

    // set the first column of the grid on fire
    SetFirstColumnOnFire(grid, N);
    return grid;
}

void DisplayGrid(const std::vector<int>& grid, const int N, std::ostream& output = std::cout){
    // ============================================
    // Displays the grid transposed (rows become columns, columns become rows)
    // Different processors are more likely to have even workloads
    // ============================================
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int state = grid[j * N + i];  // Transpose: swap i and j indices
            char display_char = (state == EMPTY) ? '.' : (state == ALIVE) ? 'T' : (state == BURNING) ? 'B' : 'X';
            output << display_char << " ";
        }
        output << std::endl;
    }
}

void parse_arguments_and_generate_grids(const int argc, char* argv[], const int iproc, const int nproc, int& N, int& num_runs, int& seed, float& probability, bool& draw_grid, std::string& run_description, std::vector<std::vector<int>>& full_grids){
  // ============================================
  // Parse command line arguments and generate grids on root process
  // ============================================
  if (iproc == 0){
    // Argument parsing
    if (argc != 6 && argc != 3) {
        std::cerr << "Error: Incorrect number of arguments." << std::endl;
        std::cerr << "Usage 1 (random grid mode): " << argv[0] << " <grid_size> <num_runs> <seed> <probability> <draw_grid>" << std::endl;
        std::cerr << "Usage 2 (read file mode): " << argv[0] << " <filename> <draw_grid>" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // reading arguments for random grid mode
    if (argc == 6){
      N = atoi(argv[1]);
      num_runs = atoi(argv[2]);
      seed = atoi(argv[3]);
      probability = atof(argv[4]);
      
      // Input validation
      if (N <= 0) {
          std::cerr << "Error: grid_size must be a positive integer (got " << N << ")" << std::endl;
          MPI_Abort(MPI_COMM_WORLD, 1);
      }
      
      if (probability < 0.0 || probability > 1.0) {
          std::cerr << "Error: probability must be between 0.0 and 1.0 (got " << probability << ")" << std::endl;
          MPI_Abort(MPI_COMM_WORLD, 1);
      }
      
      // Generate initial grids
      for (int run=0; run<num_runs; run++){
        full_grids.push_back(GenerateGrid(N, seed+run, probability));
      }
      run_description = "N" + std::to_string(N) + "_seed" + std::to_string(seed) + "_prob" + std::to_string((int)(probability*100));
    }

    // reading arguments for read file mode
    else if (argc == 3) {
      std::cerr << "Note: Reading grid from file mode. Only 1 run will be executed." << std::endl;
      // Read grid from file
      std::string input_filename = argv[1];
      full_grids.push_back(ReadGridFromFile(input_filename, N));
      num_runs = 1;
      
      // Extract filename without path and extension
      size_t last_slash = input_filename.find_last_of("/\\");
      size_t last_dot = input_filename.find_last_of(".");
      std::string base_name = input_filename.substr(last_slash + 1, last_dot - last_slash - 1);
      run_description = "file_" + base_name;
    }

    draw_grid = (argc == 6) ? atoi(argv[5]) : atoi(argv[2]);

    // check that the number of rows is at least as large as the number of processes (otherwise some processes would have no work to do)
    if (N < nproc){
          std::cerr << "Error: grid_size must be at least as large as the number of processes (grid_size=" << N << ", processes=" << nproc << ")" << std::endl;
          MPI_Abort(MPI_COMM_WORLD, 1);
      }
  }
}

void output_statistics_and_results(const int iproc, const int nproc, const int N, const bool draw_grid, const std::string& run_description, const double start, const double setup_end, const std::vector<int>& burning_steps_list, const std::vector<bool>& bottom_reached_list, const std::vector<double>& execution_time_list, const std::vector<double>& broadcast_allocate_time_list, const std::vector<int>& final_grid){
  
  // ============================================
  // Output statistics and results
  // ============================================
  
  // Only root process outputs results
  if (iproc != 0) {
    return;
  }

  // ============================================
  // Calculate statistics across runs
  // ============================================
  const int runs = burning_steps_list.size();
  double avg_burning_steps = 0.0;
  int runs_with_bottom_reached = 0;
  double avg_execution_time = 0.0;
  double avg_broadcast_allocate_time = 0.0;
  
  for (int i = 0; i < runs; i++){
    avg_burning_steps += burning_steps_list[i];
    if (bottom_reached_list[i]) runs_with_bottom_reached++;
    avg_execution_time += execution_time_list[i];
    avg_broadcast_allocate_time += broadcast_allocate_time_list[i];
  }
  avg_burning_steps /= runs;
  avg_execution_time /= runs;
  avg_broadcast_allocate_time /= runs;
  const double probability_bottom_reached = static_cast<double>(runs_with_bottom_reached) / runs;

  // ============================================
  // Output results to console
  // ============================================
  std::cout << "Mean Burning steps: " << avg_burning_steps << std::endl;
  std::cout << "Estimated Probability Bottom reached: " << probability_bottom_reached << std::endl;
  std::cout << "Mean Execution time (seconds): " << avg_execution_time << std::endl;
  std::cout << "Mean Broadcast/Allocate time (seconds): " << avg_broadcast_allocate_time << std::endl;
  
  // ============================================
  // Write timings to file
  // ============================================
  const std::string timing_filename = "timings_" + run_description + "_" + std::to_string(nproc) + "proc.txt"; // generate filename based on run description and number of processes
  std::ofstream timing_file(timing_filename);
  if (!timing_file.is_open()) {
    std::cerr << "Error: Unable to write to file " << timing_filename << std::endl;
    return;
  }
    // Write summary statistics to timing file
    timing_file << "Run: " << run_description << std::endl;
    timing_file << "Number of runs: " << runs << std::endl;
    timing_file << "Processors: " << nproc << std::endl;
    timing_file << "Grid size: " << N << "x" << N << std::endl;
    timing_file << "Mean Burning steps: " << avg_burning_steps << std::endl;
    timing_file << "Estimated Probability Bottom reached: " << probability_bottom_reached << std::endl;
    timing_file << "Mean Execution time (seconds): " << avg_execution_time << std::endl;
    timing_file << "Mean Broadcast/Allocate time (seconds): " << avg_broadcast_allocate_time << std::endl;
    timing_file << "Setup time: " << setup_end - start << " seconds" << std::endl;
    
    timing_file << "\nBurning steps per run: ";
    for (size_t i = 0; i < burning_steps_list.size(); i++) {
      timing_file << burning_steps_list[i];
      if (i < burning_steps_list.size() - 1) timing_file << ", ";
    }
    timing_file << std::endl;
    
    timing_file << "Execution time per run (seconds): ";
    for (size_t i = 0; i < execution_time_list.size(); i++) {
      timing_file << execution_time_list[i];
      if (i < execution_time_list.size() - 1) timing_file << ", ";
    }
    timing_file << std::endl;
  timing_file.close();
  
  std::cout << "Timings written to " << timing_filename << std::endl;

  // ============================================
  // Write out grid to data file (for last run)
  // ============================================
  const std::string data_filename = "data_" + run_description + "_" + std::to_string(nproc) + "proc.txt"; // generate filename based on run description and number of processes
  std::ofstream data_file(data_filename);
  if (data_file.is_open()){
    data_file << "Final grid (" << N << "x" << N << ")" << std::endl;
    DisplayGrid(final_grid, N, data_file);
    data_file.close();
    std::cout << "Grid data written to " << data_filename << std::endl;
  }

  if (draw_grid){
    std::cout << "Final grid:" << std::endl;
    DisplayGrid(final_grid, N);
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

    // Program parameters
    std::vector<int> states_current;
    std::vector<int> final_grid;
    int N;
    int num_runs;
    bool draw_grid = false;
    int seed;
    float probability;
    std::vector<std::vector<int>> full_grids;
    std::vector<int> full_grid;

    // Storage for results
    std::string run_description;  // For filename generation
    std::vector<int> burning_steps_list;
    std::vector<bool> bottom_reached_list;
    std::vector<double> execution_time_list;
    std::vector<double> broadcast_allocate_time_list;

    if (iproc == 0) {
        std::cout << "Running with " << nproc << " MPI processes" << std::endl;
    }
    
    // Parse arguments and generate grids
    parse_arguments_and_generate_grids(argc, argv, iproc, nproc, N, num_runs, seed, probability, 
                                       draw_grid, run_description, full_grids);
    
    // =========================================
    // Broadcast entire grid size to all processes
    // =========================================

    // broadcast the data
    if (nproc > 1){
      // first share the image size
      MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&num_runs, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&draw_grid, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    }

    // =========================================
    // Distribute the grid among processes
    // =========================================

    // divide the rows among MPI tasks - must be done AFTER N is broadcast
    int row_start, row_end;
    distribute_grid(N, iproc, nproc, row_start, row_end);
    double setup_end = MPI_Wtime();

    // =========================================
    // loop over runs (if multiple)
    // =========================================

    for (int irun = 0; irun < num_runs; irun++){
      double broadcast_allocate_start = MPI_Wtime();
      // Send/receive the relevant rows and reshape to local storage
      if (nproc > 1){
        if (iproc == 0){
          full_grid = full_grids[irun];        
          // Send relevant rows to each other process
          for (int p = 1; p < nproc; p++){
            int p_row_start, p_row_end;
            distribute_grid(N, p, nproc, p_row_start, p_row_end);
            MPI_Send(&full_grid[p_row_start * N], (p_row_end - p_row_start) * N, MPI_INT, p, 0, MPI_COMM_WORLD);
          }
          
          // Now extract process 0's own local rows
          states_current.resize((row_end - row_start) * N);
          for (int i = row_start; i < row_end; i++){
            for (int j = 0; j < N; j++){
              states_current[(i - row_start) * N + j] = full_grid[i * N + j];
            }
          }
        } else {
          // Other processes: allocate space and receive their rows
          states_current.resize((row_end - row_start) * N);
          MPI_Recv(states_current.data(), (row_end - row_start) * N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
      } else {
        // Single process: just use the full grid directly
        states_current = full_grids[irun];
      }

      double broadcast_allocate_end = MPI_Wtime();

      // =========================================
      // Run the simulation
      // =========================================
      
      int burning_steps = 0;
      bool bottom_reached = false;
      double model_start = MPI_Wtime();

      
      // check that the tasks stay in sync
      MPI_Barrier(MPI_COMM_WORLD);

      std::vector<int> states_new((row_end - row_start) * N, 0); // Only store local rows
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
            MPI_Sendrecv(&states_current[(row_end-row_start-1) * N], N, MPI_INT, iproc+1, 1, ghost_row_below.data(), N, MPI_INT, iproc+1, 0, MPI_COMM_WORLD, &status);
        }
          burning_steps++;
          Model(N, states_current, states_new, row_start, row_end, iproc, nproc, ghost_row_above, ghost_row_below, changes_made);
          if (draw_grid){
            std::vector<int> combined_grid = CombineGrids(N, states_new, row_start, row_end, iproc, nproc);
            if (iproc == 0){
              // Display the grid
              std::cout << "=========================================" << std::endl;
              std::cout << "Step " << burning_steps << ":" << std::endl;
              std::cout << "=========================================" << std::endl;
              DisplayGrid(combined_grid, N);
            }
          }
        states_current = states_new; 
      }
      double model_end = MPI_Wtime();

      // check that the tasks stay in sync
      MPI_Barrier(MPI_COMM_WORLD);

      // Get Final Results
      final_grid = CombineGrids(N, states_current, row_start, row_end, iproc, nproc);

      // check if bottom row on fire
      if (iproc == 0){
        for (int j=0;j<N;j++){
            if (final_grid[(N-1)*N + j] == BURNING || final_grid[(N-1)*N + j] == BURNT){
                bottom_reached = true;
                break;
            }
        }
      }
      // =========================================
      // Store results for this run
      // =========================================
      if (iproc == 0){
        burning_steps_list.push_back(burning_steps);
        bottom_reached_list.push_back(bottom_reached);
        execution_time_list.push_back(model_end - model_start);
        broadcast_allocate_time_list.push_back(broadcast_allocate_end - broadcast_allocate_start);
      }
    }
    // =========================================
    // Output statistics and results
    // =========================================
    output_statistics_and_results(iproc, nproc, N, draw_grid, run_description, start, setup_end,
                                  burning_steps_list, bottom_reached_list, 
                                  execution_time_list, broadcast_allocate_time_list, final_grid);

    // finalise MPI
    MPI_Finalize();
    return 0;
}

