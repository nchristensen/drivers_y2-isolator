#! /bin/bash

dir_list=("eigthX" "quarterX" "halfX" "oneX" "1p5X" "twoX" "threeX" "fourX")

for dir in ${dir_list[@]}; do
  echo "--- $dir ---"
  avg_time=`grep "step walltime" ${dir}/mirge-1.out | awk 'NR>1{ sum += $4 } END { if (NR > 0) print sum / NR }'`
  first_time=`grep "step walltime" ${dir}/mirge-1.out | awk 'NR==1{ sum += $4 } END { if (NR > 0) print sum }'`
  echo "first step time: $first_time"
  echo "average (2-50) step time: $avg_time"
done

