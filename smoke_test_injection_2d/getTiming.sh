#! /bin/bash

file="mirge-1.out"
if [[ $# -gt "0" ]]; then
  file=$1
fi
echo "Getting timing from ${file}"

avg_time=`grep "step walltime" $file | awk 'NR>1 && NR!=20{ sum += $10 } END { if (NR > 0) print sum / NR }'`
first_time=`grep "step walltime" $file | awk 'NR==1{ sum += $10 } END { if (NR > 0) print sum }'`
last_time=`grep "step walltime" $file | awk 'NR==20{ sum += $10 } END { if (NR > 0) print sum }'`
echo "first step time: $first_time"
echo "last step time: $last_time"
echo "average (2-20) step time: $avg_time"
