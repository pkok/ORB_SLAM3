#!/bin/bash
DATASET_DIR="/mnt/e/EuRoC/"
PLOT_TYPE="png"


## Monocular
echo "Monocular"
for i in MH01 MH02 MH03 MH04 MH05 V101 V102 V103 V201 V202 V203
do
for j in {0..5}
do
../Monocular/mono_euroc ../Vocabulary/ORBvoc.txt ../Monocular/EuRoC.yaml ${DATASET_DIR}/${i} ../Monocular/EuRoC_TimeStamps/${i}.txt dataset-${i}_mono-${j}
done
python3 ../evaluation/evaluate_ate_scale.py ../evaluation/Ground_truth/EuRoC_left_cam/${i}_GT.txt ../data/f_dataset-${i}_mono-{0..5}.txt --plot mono-${i}.${PLOT_TYPE} --export-ate ../data/errors-${i}_mono.csv --verbose >> ./eval-mono-${i}.md
done
echo -e "\a"

exit

## Stereo
echo "Stereo"
for i in MH01 MH02 MH03 MH04 MH05 V101 V102 V103 V201 V202 V203
do
for j in {0..5}
do
../Examples/Stereo/stereo_euroc ../Vocabulary/ORBvoc.txt ../Examples/Stereo/EuRoC.yaml ${DATASET_DIR}/${i} ../Examples/Stereo/EuRoC_TimeStamps/${i}.txt dataset-${i}_stereo-${j}
done
python3 ../evaluation/evaluate_ate_scale.py ../evaluation/Ground_truth/EuRoC_left_cam/${i}_GT.txt ../data/f_dataset-${i}_stereo-{0..5}.txt --plot stereo-${i}.${PLOT_TYPE} --export-ate ../data/errors-${i}_stereo.csv --verbose >> ./eval-stereo-${i}.md
done
echo -e "\a"


## Monocular inertial
echo "Monocular inertial"
for i in MH01 MH02 MH03 MH04 MH05 V101 V102 V103 V201 V202 V203
do
for j in {0..5}
do
../Monocular-Inertial/mono_inertial_euroc ../Vocabulary/ORBvoc.txt ../Monocular-Inertial/EuRoC.yaml ${DATASET_DIR}/${i} ../Monocular-Inertial/EuRoC_TimeStamps/${i}.txt dataset-${i}_monoi-${j}
#done
python3 ../evaluation/evaluate_ate_scale.py ${DATASET_DIR}/${i}/mav0/state_groundtruth_estimate0/data.csv ../data/f_dataset-${i}_monoi-{0..5}.txt --plot monoi-${i}.${PLOT_TYPE} --export-ate ../data/errors-${i}_monoi.csv --verbose >> ./eval-monoi-${i}.md
done
echo -e "\a"


## Stereo inertial
echo "Stereo inertial"
for i in MH01 MH02 MH03 MH04 MH05 V101 V102 V103 V201 V202 V203
do
for j in {0..5}
do
../Stereo-Inertial/stereo_inertial_euroc ../Vocabulary/ORBvoc.txt ../Stereo-Inertial/EuRoC.yaml ${DATASET_DIR}/${i} ../tereo-Inertial/EuRoC_TimeStamps/${i}.txt dataset-${i}_stereoi-${j}
done
python3 ../evaluation/evaluate_ate_scale.py ${DATASET_DIR}/${i}/mav0/state_groundtruth_estimate0/data.csv ../data/f_dataset-${i}_stereoi-{0..5}.txt --plot stereoi-${i}.${PLOT_TYPE} --export-ate ../data/errors-${i}_stereoi.csv --verbose >> ./eval-stereoi-${i}.md
done
echo -e "\a"
