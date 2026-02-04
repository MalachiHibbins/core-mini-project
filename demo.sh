#!/bin/bash
# Demo script to show the MPI cellular automata in action

echo "=================================="
echo "MPI Cellular Automata Demo"
echo "=================================="
echo ""

echo "1. Building the project..."
make clean > /dev/null 2>&1
make > /dev/null 2>&1
echo "   ✓ Build successful"
echo ""

echo "2. Running with 1 process (serial)..."
mpirun --allow-run-as-root -np 1 ./cellular_automata_mpi -r 20 -c 20 -i 50 -o demo_np1.txt > /dev/null 2>&1
echo "   ✓ Completed"
echo ""

echo "3. Running with 2 processes..."
mpirun --allow-run-as-root -np 2 ./cellular_automata_mpi -r 20 -c 20 -i 50 -o demo_np2.txt > /dev/null 2>&1
echo "   ✓ Completed"
echo ""

echo "4. Running with 4 processes..."
mpirun --allow-run-as-root --oversubscribe -np 4 ./cellular_automata_mpi -r 20 -c 20 -i 50 -o demo_np4.txt > /dev/null 2>&1
echo "   ✓ Completed"
echo ""

echo "5. Running with glider pattern..."
mpirun --allow-run-as-root -np 2 ./cellular_automata_mpi -f examples/glider.txt -r 5 -c 5 -i 10 -o demo_glider.txt
echo ""

echo "=================================="
echo "Demo Complete!"
echo "=================================="
echo ""
echo "Output files created:"
ls -lh demo_*.txt 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}'

# Clean up
rm -f demo_*.txt
