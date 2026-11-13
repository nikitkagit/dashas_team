#!/usr/bin/env bash
# Create a benchmark corpus of 100 tasks for the harness.
# Usage: ./make_tasks_100.sh [TARGET_DIR]   (default: ./tasks)
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

put_base_csv() {
  put_file "$1" data.csv <<'EOF'
name,value,category
alpha,10,A
beta,20,B
gamma,30,A
delta,40,C
epsilon,50,A
EOF
}

put_sales_csv() {
  put_file "$1" sales.csv <<'EOF'
region,units,price
north,10,5
south,20,3
east,15,4
west,25,2
EOF
}

put_students_csv() {
  put_file "$1" students.csv <<'EOF'
name,score,grade
ann,85,A
bob,72,B
cid,91,A
dan,68,D
eve,78,B
fay,88,A
EOF
}

mkdir -p "$TASKS_DIR"

# ============================================================
# CODING (25)
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

new_task coding_0011 <<'EOF'
{"task_id": "coding_0011", "task_type": "coding", "check": "exact_int", "reward": 1.0, "expected": "42", "instruction": "Compute 6 multiplied by 7. Write ONLY the integer to answer.txt via write_file."}
EOF

new_task coding_0012 <<'EOF'
{"task_id": "coding_0012", "task_type": "coding", "check": "exact_int", "reward": 1.0, "expected": "5", "instruction": "How many elements are in the list [1, 2, 3, 4, 5]? Write ONLY the number to answer.txt via write_file."}
EOF

new_task coding_0013 <<'EOF'
{"task_id": "coding_0013", "task_type": "coding", "check": "exact_str", "reward": 1.0, "expected": "dlrow", "instruction": "Reverse the string world. Write ONLY the reversed string to answer.txt via write_file."}
EOF

new_task coding_0014 <<'EOF'
{"task_id": "coding_0014", "task_type": "coding", "check": "exact_int", "reward": 1.0, "expected": "2", "instruction": "How many times does the letter a appear in the string abcabc? Write ONLY the number to answer.txt via write_file."}
EOF

new_task coding_0015 <<'EOF'
{"task_id": "coding_0015", "task_type": "coding", "check": "exact_int", "reward": 1.0, "expected": "60", "instruction": "Compute the sum of the list [10, 20, 30]. Write ONLY the integer to answer.txt via write_file."}
EOF

new_task coding_0016 <<'EOF'
{"task_id": "coding_0016", "task_type": "coding", "check": "exact_str", "reward": 1.0, "expected": "PYTHON", "instruction": "Convert the string Python to uppercase. Write ONLY the result to answer.txt via write_file."}
EOF

new_task coding_0017 <<'EOF'
{"task_id": "coding_0017", "task_type": "coding", "check": "exact_int", "reward": 1.0, "expected": "81", "instruction": "Compute 3 to the power of 4. Write ONLY the integer to answer.txt via write_file."}
EOF

new_task coding_0018 <<'EOF'
{"task_id": "coding_0018", "task_type": "coding", "check": "exact_int", "reward": 1.0, "expected": "1", "instruction": "Compute the minimum of the list [4, 2, 8, 1, 9]. Write ONLY the integer to answer.txt via write_file."}
EOF

new_task coding_0019 <<'EOF'
{"task_id": "coding_0019", "task_type": "coding", "check": "exact_int", "reward": 1.0, "expected": "9", "instruction": "Compute the maximum of the list [4, 2, 8, 1, 9]. Write ONLY the integer to answer.txt via write_file."}
EOF

new_task coding_0020 <<'EOF'
{"task_id": "coding_0020", "task_type": "coding", "check": "exact_str", "reward": 1.0, "expected": "cot", "instruction": "Replace every occurrence of the letter a with the letter o in the string cat. Write ONLY the result to answer.txt via write_file."}
EOF

new_task coding_0021 <<'EOF'
{"task_id": "coding_0021", "task_type": "coding", "check": "exact_str", "reward": 1.0, "expected": "1 2 3", "instruction": "Sort the list [3, 1, 2] in ascending order and write the result space-separated to answer.txt via write_file."}
EOF

new_task coding_0022 <<'EOF'
{"task_id": "coding_0022", "task_type": "coding", "check": "exact_int", "reward": 1.0, "expected": "9", "instruction": "How many characters are in the string benchmark? Write ONLY the number to answer.txt via write_file."}
EOF

new_task coding_0023 <<'EOF'
{"task_id": "coding_0023", "task_type": "coding", "check": "exact_int", "reward": 1.0, "expected": "5", "instruction": "How many vowels (a, e, i, o, u) are in the string education? Write ONLY the number to answer.txt via write_file."}
EOF

new_task coding_0024 <<'EOF'
{"task_id": "coding_0024", "task_type": "coding", "check": "exact_int", "reward": 1.0, "expected": "3", "instruction": "Split the string a,b,c on commas. How many parts are there? Write ONLY the number to answer.txt via write_file."}
EOF

new_task coding_0025 <<'EOF'
{"task_id": "coding_0025", "task_type": "coding", "check": "unittest", "reward": 2.0, "expected": "passed", "instruction": "Create a file solution.py in the workspace containing a function is_even(n) that returns True if n is even and False otherwise. Tests are in test_solution.py. When all tests pass, write done to answer.txt via write_file."}
EOF

put_file coding_0025 test_solution.py <<'EOF'
import unittest

from solution import is_even


class TestIsEven(unittest.TestCase):
    def test_zero(self):
        self.assertTrue(is_even(0))

    def test_two(self):
        self.assertTrue(is_even(2))

    def test_three(self):
        self.assertFalse(is_even(3))

    def test_negative_even(self):
        self.assertTrue(is_even(-4))

    def test_negative_odd(self):
        self.assertFalse(is_even(-7))


if __name__ == "__main__":
    unittest.main()
EOF

# ============================================================
# MATH (25)
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

new_task math_0011 <<'EOF'
{"task_id": "math_0011", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "5040", "instruction": "Compute 7 factorial (7!). Write ONLY the integer to answer.txt via write_file."}
EOF

new_task math_0012 <<'EOF'
{"task_id": "math_0012", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "63", "instruction": "Compute 100 minus 37. Write ONLY the integer to answer.txt via write_file."}
EOF

new_task math_0013 <<'EOF'
{"task_id": "math_0013", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "12", "instruction": "Compute 144 divided by 12. Write ONLY the integer to answer.txt via write_file."}
EOF

new_task math_0014 <<'EOF'
{"task_id": "math_0014", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "128", "instruction": "Compute 2 to the power of 7. Write ONLY the integer to answer.txt via write_file."}
EOF

new_task math_0015 <<'EOF'
{"task_id": "math_0015", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "36", "instruction": "Compute the least common multiple of 12 and 18. Write ONLY the integer to answer.txt via write_file."}
EOF

new_task math_0016 <<'EOF'
{"task_id": "math_0016", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "16", "instruction": "What is the square root of 256? Write ONLY the integer to answer.txt via write_file."}
EOF

new_task math_0017 <<'EOF'
{"task_id": "math_0017", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "55", "instruction": "Compute the sum of squares 1^2 + 2^2 + 3^2 + 4^2 + 5^2. Write ONLY the integer to answer.txt via write_file."}
EOF

new_task math_0018 <<'EOF'
{"task_id": "math_0018", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "4", "instruction": "How many prime numbers are strictly between 10 and 20 (exclusive of the endpoints)? Write ONLY the number to answer.txt via write_file."}
EOF

new_task math_0019 <<'EOF'
{"task_id": "math_0019", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "120", "instruction": "Compute the binomial coefficient 10 choose 3. Write ONLY the integer to answer.txt via write_file."}
EOF

new_task math_0020 <<'EOF'
{"task_id": "math_0020", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "7", "instruction": "Solve the equation 3x - 7 = 14 for x. Write ONLY the integer value of x to answer.txt via write_file."}
EOF

new_task math_0021 <<'EOF'
{"task_id": "math_0021", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "63", "instruction": "Compute the area of a rectangle with width 7 and height 9. Write ONLY the integer to answer.txt via write_file."}
EOF

new_task math_0022 <<'EOF'
{"task_id": "math_0022", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "30", "instruction": "Compute 15 percent of 200. Write ONLY the integer to answer.txt via write_file."}
EOF

new_task math_0023 <<'EOF'
{"task_id": "math_0023", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "4096", "instruction": "Compute 2 to the power of 12. Write ONLY the integer to answer.txt via write_file."}
EOF

new_task math_0024 <<'EOF'
{"task_id": "math_0024", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "100", "instruction": "Compute 6 squared plus 8 squared. Write ONLY the integer to answer.txt via write_file."}
EOF

new_task math_0025 <<'EOF'
{"task_id": "math_0025", "task_type": "math", "check": "exact_int", "reward": 1.0, "expected": "5", "instruction": "Compute the median of the list [1, 3, 5, 7, 9]. Write ONLY the integer to answer.txt via write_file."}
EOF

# ============================================================
# DATA ANALYSIS (25)
# ============================================================

new_task data_analysis_0001 <<'EOF'
{"task_id": "data_analysis_0001", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "5", "instruction": "Read data.csv in the workspace. How many data rows does it have, excluding the header? Write ONLY the number to answer.txt via write_file."}
EOF
put_base_csv data_analysis_0001

new_task data_analysis_0002 <<'EOF'
{"task_id": "data_analysis_0002", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "150", "instruction": "Read data.csv in the workspace. Compute the sum of the value column. Write ONLY the integer to answer.txt via write_file."}
EOF
put_base_csv data_analysis_0002

new_task data_analysis_0003 <<'EOF'
{"task_id": "data_analysis_0003", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "50", "instruction": "Read data.csv in the workspace. What is the maximum value in the value column? Write ONLY the integer to answer.txt via write_file."}
EOF
put_base_csv data_analysis_0003

new_task data_analysis_0004 <<'EOF'
{"task_id": "data_analysis_0004", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "3", "instruction": "Read data.csv in the workspace. How many distinct values are in the category column? Write ONLY the integer to answer.txt via write_file."}
EOF
put_base_csv data_analysis_0004

new_task data_analysis_0005 <<'EOF'
{"task_id": "data_analysis_0005", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "30", "instruction": "Read data.csv in the workspace. Compute the arithmetic mean of the value column. Write ONLY the integer (rounded) to answer.txt via write_file."}
EOF
put_base_csv data_analysis_0005

new_task data_analysis_0006 <<'EOF'
{"task_id": "data_analysis_0006", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "3", "instruction": "Read data.csv in the workspace. How many rows have category equal to A? Write ONLY the integer to answer.txt via write_file."}
EOF
put_base_csv data_analysis_0006

new_task data_analysis_0007 <<'EOF'
{"task_id": "data_analysis_0007", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "30", "instruction": "Read data.csv in the workspace. Compute the median of the value column. Write ONLY the integer to answer.txt via write_file."}
EOF
put_base_csv data_analysis_0007

new_task data_analysis_0008 <<'EOF'
{"task_id": "data_analysis_0008", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "4", "instruction": "Read data.csv in the workspace. How many rows have a name with length strictly greater than 4? Write ONLY the integer to answer.txt via write_file."}
EOF
put_base_csv data_analysis_0008

new_task data_analysis_0009 <<'EOF'
{"task_id": "data_analysis_0009", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "3", "instruction": "Read data.csv in the workspace. How many rows have value strictly greater than 20? Write ONLY the integer to answer.txt via write_file."}
EOF
put_base_csv data_analysis_0009

new_task data_analysis_0010 <<'EOF'
{"task_id": "data_analysis_0010", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "90", "instruction": "Read data.csv in the workspace. Compute the sum of the value column over rows where category is A. Write ONLY the integer to answer.txt via write_file."}
EOF
put_base_csv data_analysis_0010

new_task data_analysis_0011 <<'EOF'
{"task_id": "data_analysis_0011", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "70", "instruction": "Read sales.csv in the workspace. Compute the sum of the units column. Write ONLY the integer to answer.txt via write_file."}
EOF
put_sales_csv data_analysis_0011

new_task data_analysis_0012 <<'EOF'
{"task_id": "data_analysis_0012", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "220", "instruction": "Read sales.csv in the workspace. Compute the total revenue = sum over rows of (units * price). Write ONLY the integer to answer.txt via write_file."}
EOF
put_sales_csv data_analysis_0012

new_task data_analysis_0013 <<'EOF'
{"task_id": "data_analysis_0013", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "5", "instruction": "Read sales.csv in the workspace. What is the maximum value in the price column? Write ONLY the integer to answer.txt via write_file."}
EOF
put_sales_csv data_analysis_0013

new_task data_analysis_0014 <<'EOF'
{"task_id": "data_analysis_0014", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "4", "instruction": "Read sales.csv in the workspace. How many distinct regions are in the region column? Write ONLY the integer to answer.txt via write_file."}
EOF
put_sales_csv data_analysis_0014

new_task data_analysis_0015 <<'EOF'
{"task_id": "data_analysis_0015", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "18", "instruction": "Read sales.csv in the workspace. Compute the mean of the units column, rounded to the nearest integer. Write ONLY the integer to answer.txt via write_file."}
EOF
put_sales_csv data_analysis_0015

new_task data_analysis_0016 <<'EOF'
{"task_id": "data_analysis_0016", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "50", "instruction": "Read sales.csv in the workspace. Compute the total revenue (units * price) for the north region. Write ONLY the integer to answer.txt via write_file."}
EOF
put_sales_csv data_analysis_0016

new_task data_analysis_0017 <<'EOF'
{"task_id": "data_analysis_0017", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "6", "instruction": "Read students.csv in the workspace. How many data rows does it have, excluding the header? Write ONLY the integer to answer.txt via write_file."}
EOF
put_students_csv data_analysis_0017

new_task data_analysis_0018 <<'EOF'
{"task_id": "data_analysis_0018", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "80", "instruction": "Read students.csv in the workspace. Compute the mean of the score column, rounded to the nearest integer. Write ONLY the integer to answer.txt via write_file."}
EOF
put_students_csv data_analysis_0018

new_task data_analysis_0019 <<'EOF'
{"task_id": "data_analysis_0019", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "91", "instruction": "Read students.csv in the workspace. What is the maximum value in the score column? Write ONLY the integer to answer.txt via write_file."}
EOF
put_students_csv data_analysis_0019

new_task data_analysis_0020 <<'EOF'
{"task_id": "data_analysis_0020", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "3", "instruction": "Read students.csv in the workspace. How many rows have grade equal to A? Write ONLY the integer to answer.txt via write_file."}
EOF
put_students_csv data_analysis_0020

new_task data_analysis_0021 <<'EOF'
{"task_id": "data_analysis_0021", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "3", "instruction": "Read students.csv in the workspace. How many students have score strictly greater than 80? Write ONLY the integer to answer.txt via write_file."}
EOF
put_students_csv data_analysis_0021

new_task data_analysis_0022 <<'EOF'
{"task_id": "data_analysis_0022", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "482", "instruction": "Read students.csv in the workspace. Compute the sum of the score column. Write ONLY the integer to answer.txt via write_file."}
EOF
put_students_csv data_analysis_0022

new_task data_analysis_0023 <<'EOF'
{"task_id": "data_analysis_0023", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "2", "instruction": "Read students.csv in the workspace. How many rows have grade equal to B? Write ONLY the integer to answer.txt via write_file."}
EOF
put_students_csv data_analysis_0023

new_task data_analysis_0024 <<'EOF'
{"task_id": "data_analysis_0024", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "3", "instruction": "Read students.csv in the workspace. How many students have score greater than or equal to 85? Write ONLY the integer to answer.txt via write_file."}
EOF
put_students_csv data_analysis_0024

new_task data_analysis_0025 <<'EOF'
{"task_id": "data_analysis_0025", "task_type": "data_analysis", "check": "exact_int", "reward": 1.0, "expected": "68", "instruction": "Read students.csv in the workspace. What is the minimum value in the score column? Write ONLY the integer to answer.txt via write_file."}
EOF
put_students_csv data_analysis_0025

# ============================================================
# SEARCH (25)
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

new_task search_0011 <<'EOF'
{"task_id": "search_0011", "task_type": "search", "check": "exact_int", "reward": 1.0, "expected": "8080", "instruction": "Read config.txt in the workspace. What is the value assigned to port? Write ONLY the number to answer.txt via write_file."}
EOF
put_file search_0011 config.txt <<'EOF'
host=localhost
port=8080
debug=false
EOF

new_task search_0012 <<'EOF'
{"task_id": "search_0012", "task_type": "search", "check": "contains", "reward": 1.0, "expected": "admin@example.com", "instruction": "Read text.txt in the workspace. Find the email address and write it to answer.txt via write_file."}
EOF
put_file search_0012 text.txt <<'EOF'
For any questions, contact us at admin@example.com or call the office.
EOF

new_task search_0013 <<'EOF'
{"task_id": "search_0013", "task_type": "search", "check": "exact_str", "reward": 1.0, "expected": "doctor", "instruction": "Read bio.txt in the workspace. What is Maria's profession? Write ONLY the profession to answer.txt via write_file."}
EOF
put_file search_0013 bio.txt <<'EOF'
Maria is a doctor from Spain.
EOF

new_task search_0014 <<'EOF'
{"task_id": "search_0014", "task_type": "search", "check": "exact_int", "reward": 1.0, "expected": "3", "instruction": "Read text.txt in the workspace. How many times does the word apple appear? Write ONLY the number to answer.txt via write_file."}
EOF
put_file search_0014 text.txt <<'EOF'
apple orange apple banana apple
EOF

new_task search_0015 <<'EOF'
{"task_id": "search_0015", "task_type": "search", "check": "exact_int", "reward": 1.0, "expected": "42", "instruction": "Read text.txt in the workspace. What number is mentioned as the weight? Write ONLY the number to answer.txt via write_file."}
EOF
put_file search_0015 text.txt <<'EOF'
The package weighs 42 kg and ships tomorrow.
EOF

new_task search_0016 <<'EOF'
{"task_id": "search_0016", "task_type": "search", "check": "exact_str", "reward": 1.0, "expected": "alice_99", "instruction": "Read data.json in the workspace. What is the value of the user field? Write ONLY that value to answer.txt via write_file."}
EOF
put_file search_0016 data.json <<'EOF'
{"user": "alice_99", "role": "admin"}
EOF

new_task search_0017 <<'EOF'
{"task_id": "search_0017", "task_type": "search", "check": "contains", "reward": 1.0, "expected": "+7-999-123-45-67", "instruction": "Read text.txt in the workspace. Find the phone number and write it to answer.txt via write_file."}
EOF
put_file search_0017 text.txt <<'EOF'
Call us at +7-999-123-45-67 during business hours.
EOF

new_task search_0018 <<'EOF'
{"task_id": "search_0018", "task_type": "search", "check": "exact_str", "reward": 1.0, "expected": "Japan", "instruction": "Read text.txt in the workspace. Which country is Tokyo the capital of? Write ONLY the country name to answer.txt via write_file."}
EOF
put_file search_0018 text.txt <<'EOF'
Tokyo is the capital of Japan.
EOF

new_task search_0019 <<'EOF'
{"task_id": "search_0019", "task_type": "search", "check": "exact_str", "reward": 1.0, "expected": "3.14.1", "instruction": "Read version.txt in the workspace. What version is listed? Write ONLY the version string to answer.txt via write_file."}
EOF
put_file search_0019 version.txt <<'EOF'
version: 3.14.1
released: 2024-01-10
EOF

new_task search_0020 <<'EOF'
{"task_id": "search_0020", "task_type": "search", "check": "exact_int", "reward": 1.0, "expected": "1499", "instruction": "Read price.txt in the workspace. What is the product price in rubles? Write ONLY the number to answer.txt via write_file."}
EOF
put_file search_0020 price.txt <<'EOF'
The product costs 1499 rubles.
EOF

new_task search_0021 <<'EOF'
{"task_id": "search_0021", "task_type": "search", "check": "exact_int", "reward": 1.0, "expected": "1998", "instruction": "Read text.txt in the workspace. In what year was the company founded? Write ONLY the year to answer.txt via write_file."}
EOF
put_file search_0021 text.txt <<'EOF'
Founded in 1998, the company has grown steadily ever since.
EOF

new_task search_0022 <<'EOF'
{"task_id": "search_0022", "task_type": "search", "check": "exact_str", "reward": 1.0, "expected": "deployer", "instruction": "Read config.txt in the workspace. What is the value assigned to user? Write ONLY that value to answer.txt via write_file."}
EOF
put_file search_0022 config.txt <<'EOF'
user=deployer
password=secret
EOF

new_task search_0023 <<'EOF'
{"task_id": "search_0023", "task_type": "search", "check": "exact_str", "reward": 1.0, "expected": "Berlin", "instruction": "Read text.txt in the workspace. Which city did she move to? Write ONLY the city name to answer.txt via write_file."}
EOF
put_file search_0023 text.txt <<'EOF'
She moved to Berlin last year.
EOF

new_task search_0024 <<'EOF'
{"task_id": "search_0024", "task_type": "search", "check": "contains", "reward": 1.0, "expected": "E5023", "instruction": "Read log.txt in the workspace. Find the error code in the file and write it to answer.txt via write_file."}
EOF
put_file search_0024 log.txt <<'EOF'
2024-02-01 Info: starting service
2024-02-02 CRITICAL: error code E5023 occurred
2024-02-03 Info: service restarted
EOF

new_task search_0025 <<'EOF'
{"task_id": "search_0025", "task_type": "search", "check": "exact_str", "reward": 1.0, "expected": "blue", "instruction": "Read text.txt in the workspace. What color is the sky according to the file? Write ONLY the color name to answer.txt via write_file."}
EOF
put_file search_0025 text.txt <<'EOF'
The sky is blue today.
EOF

echo "Created tasks in: $TASKS_DIR"
find "$TASKS_DIR" -maxdepth 1 -mindepth 1 -type d | sort
echo "Total: $(find "$TASKS_DIR" -maxdepth 1 -mindepth 1 -type d | wc -l) tasks"