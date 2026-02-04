# Makefile for MPI Cellular Automata

# Compiler and flags
MPICC = mpicc
CFLAGS = -Wall -O2
TARGET = cellular_automata_mpi

# Source files
SOURCES = cellular_automata_mpi.c
OBJECTS = $(SOURCES:.c=.o)

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJECTS)
	$(MPICC) $(CFLAGS) -o $(TARGET) $(OBJECTS)

# Compile source files
%.o: %.c
	$(MPICC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET) output.txt

# Run with default parameters (4 processes)
run: $(TARGET)
	mpirun --allow-run-as-root --oversubscribe -np 4 $(PWD)/$(TARGET)

# Run with custom parameters
run-test: $(TARGET)
	mpirun --allow-run-as-root -np 2 $(PWD)/$(TARGET) -r 20 -c 20 -i 10

# Help target
help:
	@echo "Available targets:"
	@echo "  all       - Build the program (default)"
	@echo "  clean     - Remove build artifacts"
	@echo "  run       - Run with 4 processes (default parameters)"
	@echo "  run-test  - Run test with 2 processes (small grid)"
	@echo ""
	@echo "Usage: mpirun -np <num_processes> ./$(TARGET) [options]"
	@echo "Options:"
	@echo "  -r <rows>       Number of rows in the grid (default: 100)"
	@echo "  -c <cols>       Number of columns in the grid (default: 100)"
	@echo "  -i <iterations> Number of iterations to run (default: 100)"
	@echo "  -f <filename>   Input file with initial grid state"
	@echo "  -o <filename>   Output file for final grid state (default: output.txt)"

.PHONY: all clean run run-test help
