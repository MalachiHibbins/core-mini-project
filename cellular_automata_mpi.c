/*
 * MPI-based Cellular Automata - Conway's Game of Life
 * 
 * This program implements Conway's Game of Life using MPI for parallelization.
 * The grid is partitioned row-wise among processes, with ghost cells used for
 * communication between neighboring processes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <mpi.h>

#define ALIVE 1
#define DEAD 0

/* Function prototypes */
void initialize_grid(int **grid, int rows, int cols, const char *filename);
void initialize_random_grid(int **grid, int rows, int cols);
int count_neighbors(int **grid, int row, int col, int rows, int cols);
void update_cell(int **current_grid, int **next_grid, int rows, int cols);
void print_grid(int **grid, int rows, int cols);
void write_grid_to_file(int **grid, int rows, int cols, const char *filename);
int **allocate_grid(int rows, int cols);
void free_grid(int **grid, int rows);
void exchange_ghost_rows(int **grid, int local_rows, int cols, int rank, int size);

int main(int argc, char *argv[]) {
    int rank, size;
    int total_rows = 100;  // Total grid rows
    int cols = 100;        // Grid columns
    int iterations = 100;  // Number of iterations
    char input_file[256] = "";
    char output_file[256] = "output.txt";
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            total_rows = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            cols = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            strncpy(input_file, argv[++i], sizeof(input_file) - 1);
            input_file[sizeof(input_file) - 1] = '\0';
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            strncpy(output_file, argv[++i], sizeof(output_file) - 1);
            output_file[sizeof(output_file) - 1] = '\0';
        }
    }
    
    // Calculate rows per process
    int rows_per_process = total_rows / size;
    int remainder = total_rows % size;
    
    // Adjust for processes that get an extra row
    int local_rows = rows_per_process;
    if (rank < remainder) {
        local_rows++;
    }
    
    // Allocate local grid with ghost rows
    // We need 2 extra rows: one at top and one at bottom for ghost cells
    int **local_grid = allocate_grid(local_rows + 2, cols);
    int **next_grid = allocate_grid(local_rows + 2, cols);
    
    // Initialize the grid
    if (rank == 0) {
        // Seed random number generator
        srand(time(NULL));
        
        // Rank 0 initializes the full grid
        int **full_grid = allocate_grid(total_rows, cols);
        
        if (strlen(input_file) > 0) {
            initialize_grid(full_grid, total_rows, cols, input_file);
        } else {
            initialize_random_grid(full_grid, total_rows, cols);
        }
        
        // Distribute rows to all processes
        int offset = 0;
        for (int p = 0; p < size; p++) {
            int p_rows = rows_per_process;
            if (p < remainder) {
                p_rows++;
            }
            
            if (p == 0) {
                // Copy to rank 0's local grid (skip the top ghost row)
                for (int i = 0; i < p_rows; i++) {
                    memcpy(local_grid[i + 1], full_grid[i], cols * sizeof(int));
                }
            } else {
                // Send to other processes
                for (int i = 0; i < p_rows; i++) {
                    MPI_Send(full_grid[offset + i], cols, MPI_INT, p, 0, MPI_COMM_WORLD);
                }
            }
            offset += p_rows;
        }
        
        free_grid(full_grid, total_rows);
    } else {
        // Other processes receive their portion
        for (int i = 0; i < local_rows; i++) {
            MPI_Recv(local_grid[i + 1], cols, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    // Main simulation loop
    for (int iter = 0; iter < iterations; iter++) {
        // Exchange ghost rows with neighbors
        exchange_ghost_rows(local_grid, local_rows, cols, rank, size);
        
        // Update cells
        update_cell(local_grid, next_grid, local_rows, cols);
        
        // Swap grids
        int **temp = local_grid;
        local_grid = next_grid;
        next_grid = temp;
    }
    
    // Gather results back to rank 0
    if (rank == 0) {
        int **full_grid = allocate_grid(total_rows, cols);
        
        // Copy rank 0's portion
        int offset = 0;
        for (int i = 0; i < local_rows; i++) {
            memcpy(full_grid[i], local_grid[i + 1], cols * sizeof(int));
        }
        offset += local_rows;
        
        // Receive from other processes
        for (int p = 1; p < size; p++) {
            int p_rows = rows_per_process;
            if (p < remainder) {
                p_rows++;
            }
            
            for (int i = 0; i < p_rows; i++) {
                MPI_Recv(full_grid[offset + i], cols, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            offset += p_rows;
        }
        
        // Write result to file
        write_grid_to_file(full_grid, total_rows, cols, output_file);
        printf("Simulation completed. Results written to %s\n", output_file);
        printf("Grid size: %d x %d, Iterations: %d, Processes: %d\n", 
               total_rows, cols, iterations, size);
        
        free_grid(full_grid, total_rows);
    } else {
        // Other processes send their results
        for (int i = 0; i < local_rows; i++) {
            MPI_Send(local_grid[i + 1], cols, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
    
    // Clean up
    free_grid(local_grid, local_rows + 2);
    free_grid(next_grid, local_rows + 2);
    
    MPI_Finalize();
    return 0;
}

/* Allocate a 2D grid */
int **allocate_grid(int rows, int cols) {
    int **grid = (int **)malloc(rows * sizeof(int *));
    if (grid == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for grid rows\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    for (int i = 0; i < rows; i++) {
        grid[i] = (int *)calloc(cols, sizeof(int));
        if (grid[i] == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for grid row %d\n", i);
            // Free previously allocated rows
            for (int j = 0; j < i; j++) {
                free(grid[j]);
            }
            free(grid);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    return grid;
}

/* Free a 2D grid */
void free_grid(int **grid, int rows) {
    for (int i = 0; i < rows; i++) {
        free(grid[i]);
    }
    free(grid);
}

/* Initialize grid from file */
void initialize_grid(int **grid, int rows, int cols, const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Cannot open file %s. Using random initialization.\n", filename);
        initialize_random_grid(grid, rows, cols);
        return;
    }
    
    char line[1024];
    int row = 0;
    
    while (fgets(line, sizeof(line), file) && row < rows) {
        // Skip comment lines starting with #
        if (line[0] == '#') {
            continue;
        }
        
        // Parse integers from the line
        char *ptr = line;
        int col = 0;
        while (col < cols && *ptr != '\0') {
            // Skip whitespace
            while (*ptr == ' ' || *ptr == '\t') ptr++;
            
            // Read integer
            if (*ptr >= '0' && *ptr <= '9') {
                grid[row][col] = (*ptr - '0');
                col++;
                ptr++;
            } else if (*ptr == '\n' || *ptr == '\r') {
                break;
            } else {
                ptr++;
            }
        }
        
        // Check if we read enough columns
        if (col < cols) {
            fprintf(stderr, "Error: Not enough values in row %d. Using random initialization.\n", row);
            fclose(file);
            initialize_random_grid(grid, rows, cols);
            return;
        }
        
        row++;
    }
    
    // Check if we read enough rows
    if (row < rows) {
        fprintf(stderr, "Error: Not enough rows in file. Using random initialization.\n");
        fclose(file);
        initialize_random_grid(grid, rows, cols);
        return;
    }
    
    fclose(file);
}

/* Initialize grid randomly */
void initialize_random_grid(int **grid, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            grid[i][j] = (rand() % 100 < 30) ? ALIVE : DEAD;  // 30% chance of being alive
        }
    }
}

/* Count live neighbors of a cell */
int count_neighbors(int **grid, int row, int col, int rows, int cols) {
    int count = 0;
    
    // Check all 8 neighbors
    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            if (di == 0 && dj == 0) continue;  // Skip the cell itself
            
            int ni = row + di;
            int nj = col + dj;
            
            // Periodic boundary conditions
            if (nj < 0) nj = cols - 1;
            if (nj >= cols) nj = 0;
            
            // For rows, we rely on ghost rows, so just check bounds
            if (ni >= 0 && ni < rows) {
                count += grid[ni][nj];
            }
        }
    }
    
    return count;
}

/* Update all cells according to Game of Life rules */
void update_cell(int **current_grid, int **next_grid, int rows, int cols) {
    // Update only the non-ghost rows (from 1 to rows, inclusive)
    for (int i = 1; i <= rows; i++) {
        for (int j = 0; j < cols; j++) {
            int neighbors = count_neighbors(current_grid, i, j, rows + 2, cols);
            
            // Conway's Game of Life rules:
            // 1. Any live cell with 2-3 live neighbors survives
            // 2. Any dead cell with exactly 3 live neighbors becomes alive
            // 3. All other cells die or stay dead
            if (current_grid[i][j] == ALIVE) {
                next_grid[i][j] = (neighbors == 2 || neighbors == 3) ? ALIVE : DEAD;
            } else {
                next_grid[i][j] = (neighbors == 3) ? ALIVE : DEAD;
            }
        }
    }
}

/* Exchange ghost rows between neighboring processes */
void exchange_ghost_rows(int **grid, int local_rows, int cols, int rank, int size) {
    MPI_Status status;
    
    // Exchange with upper neighbor
    if (rank > 0) {
        // Send top row to upper neighbor, receive into top ghost row
        MPI_Sendrecv(grid[1], cols, MPI_INT, rank - 1, 0,
                     grid[0], cols, MPI_INT, rank - 1, 0,
                     MPI_COMM_WORLD, &status);
    }
    
    // Exchange with lower neighbor
    if (rank < size - 1) {
        // Send bottom row to lower neighbor, receive into bottom ghost row
        MPI_Sendrecv(grid[local_rows], cols, MPI_INT, rank + 1, 0,
                     grid[local_rows + 1], cols, MPI_INT, rank + 1, 0,
                     MPI_COMM_WORLD, &status);
    }
}

/* Print grid to console */
void print_grid(int **grid, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%c ", grid[i][j] ? '#' : '.');
        }
        printf("\n");
    }
    printf("\n");
}

/* Write grid to file */
void write_grid_to_file(int **grid, int rows, int cols, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error: Cannot open file %s for writing.\n", filename);
        return;
    }
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%d ", grid[i][j]);
        }
        fprintf(file, "\n");
    }
    
    fclose(file);
}
