#!/bin/bash
# Test script for MPI cellular automata

echo "Testing MPI Cellular Automata Implementation"
echo "=============================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Track test results
PASSED=0
FAILED=0

# Test 1: Single process run
echo "Test 1: Single process run (baseline)"
mpirun --allow-run-as-root -np 1 ./cellular_automata_mpi -r 10 -c 10 -i 5 -o test1_np1.txt > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Test 1 passed${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ Test 1 failed${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 2: Two process run with same parameters
echo "Test 2: Two process run"
mpirun --allow-run-as-root -np 2 ./cellular_automata_mpi -r 10 -c 10 -i 5 -o test2_np2.txt > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Test 2 passed${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ Test 2 failed${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 3: Four process run
echo "Test 3: Four process run"
mpirun --allow-run-as-root --oversubscribe -np 4 ./cellular_automata_mpi -r 20 -c 20 -i 10 -o test3_np4.txt > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Test 3 passed${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ Test 3 failed${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 4: Large grid test
echo "Test 4: Large grid (100x100)"
mpirun --allow-run-as-root --oversubscribe -np 4 ./cellular_automata_mpi -r 100 -c 100 -i 50 -o test4_large.txt > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Test 4 passed${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ Test 4 failed${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 5: Input file test
echo "Test 5: Input file (glider pattern)"
mpirun --allow-run-as-root -np 2 ./cellular_automata_mpi -f examples/glider.txt -r 5 -c 5 -i 10 -o test5_glider.txt > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Test 5 passed${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ Test 5 failed${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 6: Verify output dimensions
echo "Test 6: Output dimensions verification"
LINES=$(wc -l < test1_np1.txt)
if [ "$LINES" -eq 10 ]; then
    echo -e "${GREEN}✓ Test 6 passed${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ Test 6 failed (expected 10 lines, got $LINES)${NC}"
    FAILED=$((FAILED + 1))
fi

# Clean up test files
rm -f test*.txt

echo ""
echo "=============================================="
echo "Test Results: $PASSED passed, $FAILED failed"
echo "=============================================="

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
