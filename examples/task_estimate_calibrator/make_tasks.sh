#!/usr/bin/env bash
# Create a benchmark corpus of ~40 tasks for the harness.
# Usage: ./make_tasks.sh [TARGET_DIR]   (default: ./tasks)
set -euo pipefail

TASKS_DIR="${1:-./tasks}"

new_task() {
  mkdir -p "$TASKS_DIR/$1/workspace"
  cat > "$TASKS_DIR/$1/task.json"
}

put_file() {
  mkdir -p "$TASKS_DIR/$1/workspace/$(dirname "$2")"
  cat > "$TASKS_DIR/$1/workspace/$2"
}

put_da_csv() {
  put_file "$1" data.csv <<'EOF'
name,value,category
alpha,10,A
beta,20,B
gamma,30,A
delta,40,C
epsilon,50,A
EOF
}

mkdir -p "$TASKS_DIR"

# ============================================================
# CODING (10)
# ============================================================

new_task coding_0001 <<'EOF'
{"task_id": "coding_0001", "task_type": "coding", "check": "exact_int", "reward": 1.0, "expected": "12", "instruction": "Compute 5 + 7. Write ONLY the resulting integer to answer.txt via write_file."}
EOF

new_task coding_0002 <<'EOF'
{"task_id": "coding_0002", "task_type": "coding", "check": "exact_int", "reward": 1.0, "expected": "30", "instruction": "Compute 10 * 3. Write ONLY the resulting integer to answer.txt via write_file."}
EOF

new_task coding_0003 <<'EOF'
{"task_id": "coding_0003", "task_type": "coding", "check": "exact_str", "reward": 1.0, "expected": "olleh", "instruction": "Reverse the string hello. Write ONLY the reversed string to answer.txt via write_file."}
EOF

new_task coding_0004 <<'EOF'
{"task_id": "coding_0004", "task_type": "coding", "check": "exact_int", "reward": 1.0, "expected": "11", "instruction": "How many characters are in the string programming? Write ONLY the number to answer.txt via write_file."}
EOF

new_task coding_0005 <<'EOF'
{"task_id": "coding_0005", "task_type": "coding", "check": "exact_int", "reward": 1.0, "expected": "256", "instruction": "Compute 2 to the power of 8. Write ONLY the integer to answer.txt via write_file."}
EOF

new_task coding_0006 <<'EOF'
{"task_id": "coding_0006", "task_type": "coding", "check": "exact_str", "reward": 1.0, "expected": "foobar", "instruction": "Concatenate the strings foo and bar. Write ONLY the result to answer.txt via write_file."}
EOF

new_task coding_0007 <<'EOF'
{"task_id": "coding_0007", "task_type": "coding", "check": "exact_int", "reward": 1.0, "expected": "3", "instruction": "How many times does the letter a appear in the string banana? Write ONLY the number to answer.txt via write_file."}
EOF

new_task coding_0008 <<'EOF'
{"task_id": "coding_0008", "task_type": "coding", "check": "exact_int", "reward": 1.0, "expected": "55", "instruction": "Compute the sum of all integers from 1 to 10 inclusive. Write ONLY the integer to answer.txt via write_file."}
EOF

new_task coding_0009 <<'EOF'
{"task_id": "coding_0009", "task_type": "coding", "check": "exact_str", "reward": 1.0, "expected": "hello", "instruction": "Convert the string HELLO to lowercase. Write ONLY the result to answer.txt via write_file."}
EOF

new_task coding_0010 <<'EOF'
{"task_id": "coding_0010", "task_type": "coding", "check": "unittest", "reward": 2.0, "expected": "passed", "instruction": "Create a file solution.py in the workspace containing a function add(a, b) that returns a + b. Tests are in test_solution.py. When all tests pass, write done to answer.txt via write_file."}
EOF

put_file coding_0010 test_solution.py <<'EOF'
import unittest

from solution import add


class TestAdd(unittest.TestCase):
    def test_add_positive(self):
        self.assertEqual(add(1, 2), 3)

    def test_add_negative(self):
        self.assertEqual(add(-1, 1), 0)

    def test_add_zero(self):
        self.assertEqual(add(0, 0), 0)


if __name__ == "__main__":
    unittest.main()
EOF

# ============================================================
# MATH (10)
# ============================================================

new_task math_0001 <<'EOF'
{"task_id": "math_0001", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "391", "instruction": "Compute 17 multiplied by 23. Write ONLY the integer to answer.txt via write_file."}
EOF

new_task math_0002 <<'EOF'
{"task_id": "math_0002", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "12", "instruction": "What is the square root of 144? Write ONLY the integer to answer.txt via write_file."}
EOF

new_task math_0003 <<'EOF'
{"task_id": "math_0003", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "5050", "instruction": "Compute the sum of all integers from 1 to 100 inclusive. Write ONLY the integer to answer.txt via write_file."}
EOF

new_task math_0004 <<'EOF'
{"task_id": "math_0004", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "6", "instruction": "Solve the equation 2x + 5 = 17 for x. Write ONLY the integer value of x to answer.txt via write_file."}
EOF

new_task math_0005 <<'EOF'
{"task_id": "math_0005", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "1024", "instruction": "Compute 2 to the power of 10. Write ONLY the integer to answer.txt via write_file."}
EOF

new_task math_0006 <<'EOF'
{"task_id": "math_0006", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "120", "instruction": "Compute 5 factorial (5!). Write ONLY the integer to answer.txt via write_file."}
EOF

new_task math_0007 <<'EOF'
{"task_id": "math_0007", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "6", "instruction": "Compute the greatest common divisor of 48 and 18. Write ONLY the integer to answer.txt via write_file."}
EOF

new_task math_0008 <<'EOF'
{"task_id": "math_0008", "task_type": "math", "check": "contains", "reward": 1.0, "expected": "28", "instruction": "Compute the area of a circle with radius 3 using pi approximated as 3.14. Write the numeric result to answer.txt via write_file."}
EOF

new_task math_0009 <<'EOF'
{"task_id": "math_0009", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "8", "instruction": "How many prime numbers are strictly less than 20? Write ONLY the number to answer.txt via write_file."}
EOF

new_task math_0010 <<'EOF'
{"task_id": "math_0010", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "66", "instruction": "Compute the binomial coefficient 12 choose 2. Write ONLY the integer to answer.txt via write_file."}
EOF

# ============================================================
# DATA ANALYSIS (10)
# ============================================================

new_task data_analysis_0001 <<'EOF'
{"task_id": "data_analysis_0001", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "5", "instruction": "Read data.csv in the workspace. How many data rows does it have, excluding the header? Write ONLY the number to answer.txt via write_file."}
EOF
put_da_csv data_analysis_0001

new_task data_analysis_0002 <<'EOF'
{"task_id": "data_analysis_0002", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "150", "instruction": "Read data.csv in the workspace. Compute the sum of the value column. Write ONLY the integer to answer.txt via write_file."}
EOF
put_da_csv data_analysis_0002

new_task data_analysis_0003 <<'EOF'
{"task_id": "data_analysis_0003", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "50", "instruction": "Read data.csv in the workspace. What is the maximum value in the value column? Write ONLY the integer to answer.txt via write_file."}
EOF
put_da_csv data_analysis_0003

new_task data_analysis_0004 <<'EOF'
{"task_id": "data_analysis_0004", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "3", "instruction": "Read data.csv in the workspace. How many distinct values are in the category column? Write ONLY the integer to answer.txt via write_file."}
EOF
put_da_csv data_analysis_0004

new_task data_analysis_0005 <<'EOF'
{"task_id": "data_analysis_0005", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "30", "instruction": "Read data.csv in the workspace. Compute the arithmetic mean of the value column. Write ONLY the integer (rounded) to answer.txt via write_file."}
EOF
put_da_csv data_analysis_0005

new_task data_analysis_0006 <<'EOF'
{"task_id": "data_analysis_0006", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "3", "instruction": "Read data.csv in the workspace. How many rows have category equal to A? Write ONLY the integer to answer.txt via write_file."}
EOF
put_da_csv data_analysis_0006

new_task data_analysis_0007 <<'EOF'
{"task_id": "data_analysis_0007", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "30", "instruction": "Read data.csv in the workspace. Compute the median of the value column. Write ONLY the integer to answer.txt via write_file."}
EOF
put_da_csv data_analysis_0007

new_task data_analysis_0008 <<'EOF'
{"task_id": "data_analysis_0008", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "4", "instruction": "Read data.csv in the workspace. How many rows have a name with length strictly greater than 4? Write ONLY the integer to answer.txt via write_file."}
EOF
put_da_csv data_analysis_0008

new_task data_analysis_0009 <<'EOF'
{"task_id": "data_analysis_0009", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "3", "instruction": "Read data.csv in the workspace. How many rows have value strictly greater than 20? Write ONLY the integer to answer.txt via write_file."}
EOF
put_da_csv data_analysis_0009

new_task data_analysis_0010 <<'EOF'
{"task_id": "data_analysis_0010", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "90", "instruction": "Read data.csv in the workspace. Compute the sum of the value column over rows where category is A. Write ONLY the integer to answer.txt via write_file."}
EOF
put_da_csv data_analysis_0010

# ============================================================
# SEARCH (10)
# ============================================================

new_task search_0001 <<'EOF'
{"task_id": "search_0001", "task_type": "search", "check": "exact_str", "reward": 1.0, "expected": "treasure", "instruction": "Read text.txt in the workspace. Which word names the hidden object? Write ONLY that word to answer.txt via write_file."}
EOF
put_file search_0001 text.txt <<'EOF'
The treasure is hidden somewhere in the old castle.
EOF

new_task search_0002 <<'EOF'
{"task_id": "search_0002", "task_type": "search", "check": "exact_int", "reward": 1.0, "expected": "42", "instruction": "Read file.txt in the workspace. What is the number mentioned in the file? Write ONLY the number to answer.txt via write_file."}
EOF
put_file search_0002 file.txt <<'EOF'
The answer to everything is 42.
EOF

new_task search_0003 <<'EOF'
{"task_id": "search_0003", "task_type": "search", "check": "contains", "reward": 1.0, "expected": "Error", "instruction": "Read log.txt in the workspace. Find the line that contains the word Error and write that line to answer.txt via write_file."}
EOF
put_file search_0003 log.txt <<'EOF'
2024-01-15 Info: system started
2024-01-16 Error: something broke
2024-01-17 Info: recovery attempted
2024-01-18 Info: system stopped
EOF

new_task search_0004 <<'EOF'
{"task_id": "search_0004", "task_type": "search", "check": "exact_int", "reward": 1.0, "expected": "3", "instruction": "Read text.txt in the workspace. How many times does the word the appear? Write ONLY the number to answer.txt via write_file."}
EOF
put_file search_0004 text.txt <<'EOF'
the cat the dog the bird
EOF

new_task search_0005 <<'EOF'
{"task_id": "search_0005", "task_type": "search", "check": "contains", "reward": 1.0, "expected": "https://example.com", "instruction": "Read text.txt in the workspace. Find the URL in the file and write it to answer.txt via write_file."}
EOF
put_file search_0005 text.txt <<'EOF'
Visit https://example.com for more information.
EOF

new_task search_0006 <<'EOF'
{"task_id": "search_0006", "task_type": "search", "check": "exact_str", "reward": 1.0, "expected": "2024-01-15", "instruction": "Read log.txt in the workspace. What is the date on the first line? Write ONLY the date in YYYY-MM-DD format to answer.txt via write_file."}
EOF
put_file search_0006 log.txt <<'EOF'
2024-01-15 Event happened
2024-01-16 Another event
2024-01-17 More events
EOF

new_task search_0007 <<'EOF'
{"task_id": "search_0007", "task_type": "search", "check": "exact_str", "reward": 1.0, "expected": "Alice", "instruction": "Read text.txt in the workspace. What is the name of the person mentioned? Write ONLY the name to answer.txt via write_file."}
EOF
put_file search_0007 text.txt <<'EOF'
Alice went to the store to buy bread.
EOF

new_task search_0008 <<'EOF'
{"task_id": "search_0008", "task_type": "search", "check": "contains", "reward": 1.0, "expected": "192.168.1.1", "instruction": "Read log.txt in the workspace. Find the IP address in the log and write it to answer.txt via write_file."}
EOF
put_file search_0008 log.txt <<'EOF'
2024-01-15 Info: server started
2024-01-16 Connection from 192.168.1.1 accepted
2024-01-17 Info: connection closed
EOF

new_task search_0009 <<'EOF'
{"task_id": "search_0009", "task_type": "search", "check": "exact_str", "reward": 1.0, "expected": "Bob", "instruction": "Read data.json in the workspace. What is the value of the name field? Write ONLY that value to answer.txt via write_file."}
EOF
put_file search_0009 data.json <<'EOF'
{"name": "Bob", "age": 30, "city": "London"}
EOF

new_task search_0010 <<'EOF'
{"task_id": "search_0010", "task_type": "search", "check": "exact_str", "reward": 1.0, "expected": "Paris", "instruction": "Read text.txt in the workspace. What is the capital of France according to the file? Write ONLY the name to answer.txt via write_file."}
EOF
put_file search_0010 text.txt <<'EOF'
The capital of France is Paris.
EOF

echo "Created tasks in: $TASKS_DIR"
find "$TASKS_DIR" -maxdepth 1 -mindepth 1 -type d | sort
echo "Total: $(find "$TASKS_DIR" -maxdepth 1 -mindepth 1 -type d | wc -l) tasks"