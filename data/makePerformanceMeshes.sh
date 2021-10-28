#! /bin/bash

dirs=("eigthX" "quarterX" "halfX" "oneX" "1p5X" "twoX" "threeX" "fourX")

for run in ${dirs[@]}; do
	echo "================"
  echo "creating mesh in $run"
	echo "================/n"
	cd $run
	./make_mesh.sh
	cd ..
	echo "================"
  echo "done creating mesh in $run"
	echo "================/n"

done

