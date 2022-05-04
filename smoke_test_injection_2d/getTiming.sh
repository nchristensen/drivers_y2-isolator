#! /bin/bash

avg_time=`grep "step walltime" mirge-1.out | awk 'NR>1{ sum += $10 } END { if (NR > 0) print sum / NR }'`
first_time=`grep "step walltime" mirge-1.out | awk 'NR==1{ sum += $10 } END { if (NR > 0) print sum }'`
echo "first step time: $first_time"
echo "average (2-20) step time: $avg_time"
