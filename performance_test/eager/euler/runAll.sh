#! /bin/bash

dirs=("eigthX" "quarterX" "halfX" "oneX" "1p5X" "twoX" "threeX" "fourX")

for run in ${dirs[@]}; do
	echo "================"
  echo "running $run"
	echo "================/n"
	cd $run
	./run.sh
	cd ..
	echo "================"
  echo "done running $run"
	echo "================/n"

done

