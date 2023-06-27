#!/bin/bash
rm -f results.txt
timeout 1200 python3 learning_test.py
test -f results.txt && grade=$(cat results.txt) || grade=0
mypy --strict --disallow-any-explicit ./*.py && ((grade = grade + 10)) || echo "mypy fail"
black --check ./*.py && ((grade = grade + 10)) || echo "black fail"
wd_count=$(wc -w <summary.md)
(( wd_count > 300 )) && ((grade = grade + 10)) || echo "wc should be >300, was: $wd_count"
echo "$grade" >results.txt
echo "Your grade is currently: $grade"
if ((grade > 70)); then
    echo "You're passing."
    echo "Use tuning, control, and meta-optimization to find better solutions."
    echo "If you have questions about how grading happens, just look at this script."
    exit 0
else
    exit 1
fi
