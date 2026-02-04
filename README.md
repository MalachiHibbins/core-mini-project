# MPI-based Cellular Automata

This project implements Conway's Game of Life using MPI (Message Passing Interface) for parallel computation. The simulation divides the grid among multiple processes using row-wise domain decomposition, with ghost cells for communication between neighboring processes.

## Features

- **Parallel Processing**: Uses MPI to distribute computation across multiple processes
- **Domain Decomposition**: Row-wise partitioning of the grid with ghost cell exchange
- **Configurable Parameters**: Command-line options for grid size, iterations, and I/O
- **File I/O**: Support for reading initial states and writing final results
- **Conway's Game of Life**: Classic cellular automata rules:
  - Any live cell with 2-3 neighbors survives
  - Any dead cell with exactly 3 neighbors becomes alive
  - All other cells die or stay dead

## Prerequisites

- MPI implementation (OpenMPI or MPICH)
- GCC or compatible C compiler
- Make build system

### Installing MPI

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
```

**macOS:**
```bash
brew install open-mpi
```

**Fedora/RHEL:**
```bash
sudo yum install openmpi openmpi-devel
```

## Building

To compile the program:

```bash
make
```

To clean build artifacts:

```bash
make clean
```

## Usage

### Basic Usage

Run with default parameters (100x100 grid, 100 iterations, 4 processes):

```bash
make run
```

Or manually:

```bash
mpirun -np 4 ./cellular_automata_mpi
```

### Command-Line Options

```bash
mpirun -np <num_processes> ./cellular_automata_mpi [options]
```

**Options:**
- `-r <rows>`: Number of rows in the grid (default: 100)
- `-c <cols>`: Number of columns in the grid (default: 100)
- `-i <iterations>`: Number of iterations to simulate (default: 100)
- `-f <filename>`: Input file with initial grid state
- `-o <filename>`: Output file for final grid state (default: output.txt)

### Examples

**Run with custom grid size:**
```bash
mpirun -np 4 ./cellular_automata_mpi -r 200 -c 200 -i 500
```

**Run with input file:**
```bash
mpirun -np 2 ./cellular_automata_mpi -f examples/glider.txt -r 5 -c 5 -i 10
```

**Run test simulation:**
```bash
make run-test
```

## File Format

Input files should contain space-separated integers (0 for dead, 1 for alive), one row per line.

Example (3x3 grid):
```
0 0 0
0 1 0
0 0 0
```

## Implementation Details

### Parallelization Strategy

1. **Domain Decomposition**: The grid is divided row-wise among processes
2. **Ghost Rows**: Each process maintains 2 extra rows (top and bottom) for neighbor data
3. **Communication**: Processes exchange boundary rows using `MPI_Sendrecv`
4. **Load Balancing**: Extra rows are distributed to lower-ranked processes if grid size is not evenly divisible

### Algorithm Flow

1. Rank 0 initializes the full grid (from file or randomly)
2. Grid rows are distributed to all processes
3. For each iteration:
   - Exchange ghost rows with neighboring processes
   - Update cells based on Game of Life rules
   - Swap current and next grid buffers
4. Results are gathered back to rank 0
5. Rank 0 writes the final grid to output file

## Performance Considerations

- Grid size should be significantly larger than the number of processes for good speedup
- Communication overhead increases with more processes
- Best performance typically with grid_rows >> num_processes

## Example Patterns

The `examples/` directory contains sample patterns:
- `glider.txt`: A simple moving pattern
- `blinker.txt`: An oscillator pattern
- `test_10x10.txt`: A test grid with multiple patterns

## License

This is an educational project for demonstrating MPI parallelization of cellular automata.