#!/bin/sh
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/emmanuel/car/car_test1_data.csv -m ./dl/dlmodels/quantise/q6/emmanuel/car/car_train1.csv.cn2 -x abd -k ../rules/size/q6_car_train1.csv_size5.json > ../logs/dl/dl_q6_car_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/emmanuel/cancer/cancer_test1_data.csv -m ./dl/dlmodels/quantise/q6/emmanuel/cancer/cancer_train1.csv.cn2 -x abd -k ../rules/size/q6_cancer_train1.csv_size5.json > ../logs/dl/dl_q6_cancer_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/other/mlic/parkinsons_test1_data.csv -m ./dl/dlmodels/quantise/q6/other/mlic/parkinsons_train1.csv.cn2 -x abd -k ../rules/size/q6_parkinsons_train1.csv_size5.json > ../logs/dl/dl_q6_parkinsons_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/other/mlic/pima_test1_data.csv -m ./dl/dlmodels/quantise/q6/other/mlic/pima_train1.csv.cn2 -x abd -k ../rules/size/q6_pima_train1.csv_size5.json > ../logs/dl/dl_q6_pima_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/other/house-votes-84/house-votes-84_test1_data.csv -m ./dl/dlmodels/quantise/q6/other/house-votes-84/house-votes-84_train1.csv.cn2 -x abd -k ../rules/size/q6_house-votes-84_train1.csv_size5.json > ../logs/dl/dl_q6_house-votes-84_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/other/zoo/zoo_test1_data.csv -m ./dl/dlmodels/quantise/q6/other/zoo/zoo_train1.csv.cn2 -x abd -k ../rules/size/q6_zoo_train1.csv_size5.json > ../logs/dl/dl_q6_zoo_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/other/anchor/lending/lending_test1_data.csv -m ./dl/dlmodels/quantise/q6/other/anchor/lending/lending_train1.csv.cn2 -x abd -k ../rules/size/q6_lending_train1.csv_size5.json > ../logs/dl/dl_q6_lending_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/other/anchor/recidivism/recidivism_test1_data.csv -m ./dl/dlmodels/quantise/q6/other/anchor/recidivism/recidivism_train1.csv.cn2 -x abd -k ../rules/size/q6_recidivism_train1.csv_size5.json > ../logs/dl/dl_q6_recidivism_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/other/anchor/adult/adult_test1_data.csv -m ./dl/dlmodels/quantise/q6/other/anchor/adult/adult_train1.csv.cn2 -x abd -k ../rules/size/q6_adult_train1.csv_size5.json > ../logs/dl/dl_q6_adult_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/other/fairml/compas/compas_test1_data.csv -m ./dl/dlmodels/quantise/q6/other/fairml/compas/compas_train1.csv.cn2 -x abd -k ../rules/size/q6_compas_train1.csv_size5.json > ../logs/dl/dl_q6_compas_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/penn-ml/liver-disorder/liver-disorder_test1_data.csv -m ./dl/dlmodels/quantise/q6/penn-ml/liver-disorder/liver-disorder_train1.csv.cn2 -x abd -k ../rules/size/q6_liver-disorder_train1.csv_size5.json > ../logs/dl/dl_q6_liver-disorder_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/penn-ml/australian/australian_test1_data.csv -m ./dl/dlmodels/quantise/q6/penn-ml/australian/australian_train1.csv.cn2 -x abd -k ../rules/size/q6_australian_train1.csv_size5.json > ../logs/dl/dl_q6_australian_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/penn-ml/hungarian/hungarian_test1_data.csv -m ./dl/dlmodels/quantise/q6/penn-ml/hungarian/hungarian_train1.csv.cn2 -x abd -k ../rules/size/q6_hungarian_train1.csv_size5.json > ../logs/dl/dl_q6_hungarian_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/penn-ml/heart-c/heart-c_test1_data.csv -m ./dl/dlmodels/quantise/q6/penn-ml/heart-c/heart-c_train1.csv.cn2 -x abd -k ../rules/size/q6_heart-c_train1.csv_size5.json > ../logs/dl/dl_q6_heart-c_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/penn-ml/heart-statlog/heart-statlog_test1_data.csv -m ./dl/dlmodels/quantise/q6/penn-ml/heart-statlog/heart-statlog_train1.csv.cn2 -x abd -k ../rules/size/q6_heart-statlog_train1.csv_size5.json > ../logs/dl/dl_q6_heart-statlog_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/penn-ml/diabetes/diabetes_test1_data.csv -m ./dl/dlmodels/quantise/q6/penn-ml/diabetes/diabetes_train1.csv.cn2 -x abd -k ../rules/size/q6_diabetes_train1.csv_size5.json > ../logs/dl/dl_q6_diabetes_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/penn-ml/appendicitis/appendicitis_test1_data.csv -m ./dl/dlmodels/quantise/q6/penn-ml/appendicitis/appendicitis_train1.csv.cn2 -x abd -k ../rules/size/q6_appendicitis_train1.csv_size5.json > ../logs/dl/dl_q6_appendicitis_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/penn-ml/schizo/schizo_test1_data.csv -m ./dl/dlmodels/quantise/q6/penn-ml/schizo/schizo_train1.csv.cn2 -x abd -k ../rules/size/q6_schizo_train1.csv_size5.json > ../logs/dl/dl_q6_schizo_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/penn-ml/cleve/cleve_test1_data.csv -m ./dl/dlmodels/quantise/q6/penn-ml/cleve/cleve_train1.csv.cn2 -x abd -k ../rules/size/q6_cleve_train1.csv_size5.json > ../logs/dl/dl_q6_cleve_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/penn-ml/biomed/biomed_test1_data.csv -m ./dl/dlmodels/quantise/q6/penn-ml/biomed/biomed_train1.csv.cn2 -x abd -k ../rules/size/q6_biomed_train1.csv_size5.json > ../logs/dl/dl_q6_biomed_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/penn-ml/bupa/bupa_test1_data.csv -m ./dl/dlmodels/quantise/q6/penn-ml/bupa/bupa_train1.csv.cn2 -x abd -k ../rules/size/q6_bupa_train1.csv_size5.json > ../logs/dl/dl_q6_bupa_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/penn-ml/glass2/glass2_test1_data.csv -m ./dl/dlmodels/quantise/q6/penn-ml/glass2/glass2_train1.csv.cn2 -x abd -k ../rules/size/q6_glass2_train1.csv_size5.json > ../logs/dl/dl_q6_glass2_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/penn-ml/hepatitis/hepatitis_test1_data.csv -m ./dl/dlmodels/quantise/q6/penn-ml/hepatitis/hepatitis_train1.csv.cn2 -x abd -k ../rules/size/q6_hepatitis_train1.csv_size5.json > ../logs/dl/dl_q6_hepatitis_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/penn-ml/heart-h/heart-h_test1_data.csv -m ./dl/dlmodels/quantise/q6/penn-ml/heart-h/heart-h_train1.csv.cn2 -x abd -k ../rules/size/q6_heart-h_train1.csv_size5.json > ../logs/dl/dl_q6_heart-h_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q5/emmanuel/cancer/cancer_test1_data.csv -m ./dl/dlmodels/quantise/q5/emmanuel/cancer/cancer_train1.csv.cn2 -x abd -k ../rules/size/q5_cancer_train1.csv_size5.json > ../logs/dl/dl_q5_cancer_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q5/other/mlic/parkinsons_test1_data.csv -m ./dl/dlmodels/quantise/q5/other/mlic/parkinsons_train1.csv.cn2 -x abd -k ../rules/size/q5_parkinsons_train1.csv_size5.json > ../logs/dl/dl_q5_parkinsons_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q5/other/mlic/pima_test1_data.csv -m ./dl/dlmodels/quantise/q5/other/mlic/pima_train1.csv.cn2 -x abd -k ../rules/size/q5_pima_train1.csv_size5.json > ../logs/dl/dl_q5_pima_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q5/other/zoo/zoo_test1_data.csv -m ./dl/dlmodels/quantise/q5/other/zoo/zoo_train1.csv.cn2 -x abd -k ../rules/size/q5_zoo_train1.csv_size5.json > ../logs/dl/dl_q5_zoo_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q5/other/fairml/compas/compas_test1_data.csv -m ./dl/dlmodels/quantise/q5/other/fairml/compas/compas_train1.csv.cn2 -x abd -k ../rules/size/q5_compas_train1.csv_size5.json > ../logs/dl/dl_q5_compas_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q5/penn-ml/liver-disorder/liver-disorder_test1_data.csv -m ./dl/dlmodels/quantise/q5/penn-ml/liver-disorder/liver-disorder_train1.csv.cn2 -x abd -k ../rules/size/q5_liver-disorder_train1.csv_size5.json > ../logs/dl/dl_q5_liver-disorder_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q5/penn-ml/australian/australian_test1_data.csv -m ./dl/dlmodels/quantise/q5/penn-ml/australian/australian_train1.csv.cn2 -x abd -k ../rules/size/q5_australian_train1.csv_size5.json > ../logs/dl/dl_q5_australian_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q5/penn-ml/hungarian/hungarian_test1_data.csv -m ./dl/dlmodels/quantise/q5/penn-ml/hungarian/hungarian_train1.csv.cn2 -x abd -k ../rules/size/q5_hungarian_train1.csv_size5.json > ../logs/dl/dl_q5_hungarian_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q5/penn-ml/heart-c/heart-c_test1_data.csv -m ./dl/dlmodels/quantise/q5/penn-ml/heart-c/heart-c_train1.csv.cn2 -x abd -k ../rules/size/q5_heart-c_train1.csv_size5.json > ../logs/dl/dl_q5_heart-c_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q5/penn-ml/heart-statlog/heart-statlog_test1_data.csv -m ./dl/dlmodels/quantise/q5/penn-ml/heart-statlog/heart-statlog_train1.csv.cn2 -x abd -k ../rules/size/q5_heart-statlog_train1.csv_size5.json > ../logs/dl/dl_q5_heart-statlog_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q5/penn-ml/diabetes/diabetes_test1_data.csv -m ./dl/dlmodels/quantise/q5/penn-ml/diabetes/diabetes_train1.csv.cn2 -x abd -k ../rules/size/q5_diabetes_train1.csv_size5.json > ../logs/dl/dl_q5_diabetes_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q5/penn-ml/appendicitis/appendicitis_test1_data.csv -m ./dl/dlmodels/quantise/q5/penn-ml/appendicitis/appendicitis_train1.csv.cn2 -x abd -k ../rules/size/q5_appendicitis_train1.csv_size5.json > ../logs/dl/dl_q5_appendicitis_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q5/penn-ml/schizo/schizo_test1_data.csv -m ./dl/dlmodels/quantise/q5/penn-ml/schizo/schizo_train1.csv.cn2 -x abd -k ../rules/size/q5_schizo_train1.csv_size5.json > ../logs/dl/dl_q5_schizo_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q5/penn-ml/cleve/cleve_test1_data.csv -m ./dl/dlmodels/quantise/q5/penn-ml/cleve/cleve_train1.csv.cn2 -x abd -k ../rules/size/q5_cleve_train1.csv_size5.json > ../logs/dl/dl_q5_cleve_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q5/penn-ml/biomed/biomed_test1_data.csv -m ./dl/dlmodels/quantise/q5/penn-ml/biomed/biomed_train1.csv.cn2 -x abd -k ../rules/size/q5_biomed_train1.csv_size5.json > ../logs/dl/dl_q5_biomed_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q5/penn-ml/bupa/bupa_test1_data.csv -m ./dl/dlmodels/quantise/q5/penn-ml/bupa/bupa_train1.csv.cn2 -x abd -k ../rules/size/q5_bupa_train1.csv_size5.json > ../logs/dl/dl_q5_bupa_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q5/penn-ml/glass2/glass2_test1_data.csv -m ./dl/dlmodels/quantise/q5/penn-ml/glass2/glass2_train1.csv.cn2 -x abd -k ../rules/size/q5_glass2_train1.csv_size5.json > ../logs/dl/dl_q5_glass2_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q5/penn-ml/hepatitis/hepatitis_test1_data.csv -m ./dl/dlmodels/quantise/q5/penn-ml/hepatitis/hepatitis_train1.csv.cn2 -x abd -k ../rules/size/q5_hepatitis_train1.csv_size5.json > ../logs/dl/dl_q5_hepatitis_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q5/penn-ml/heart-h/heart-h_test1_data.csv -m ./dl/dlmodels/quantise/q5/penn-ml/heart-h/heart-h_train1.csv.cn2 -x abd -k ../rules/size/q5_heart-h_train1.csv_size5.json > ../logs/dl/dl_q5_heart-h_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q4/emmanuel/cancer/cancer_test1_data.csv -m ./dl/dlmodels/quantise/q4/emmanuel/cancer/cancer_train1.csv.cn2 -x abd -k ../rules/size/q4_cancer_train1.csv_size5.json > ../logs/dl/dl_q4_cancer_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q4/other/mlic/parkinsons_test1_data.csv -m ./dl/dlmodels/quantise/q4/other/mlic/parkinsons_train1.csv.cn2 -x abd -k ../rules/size/q4_parkinsons_train1.csv_size5.json > ../logs/dl/dl_q4_parkinsons_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q4/other/mlic/pima_test1_data.csv -m ./dl/dlmodels/quantise/q4/other/mlic/pima_train1.csv.cn2 -x abd -k ../rules/size/q4_pima_train1.csv_size5.json > ../logs/dl/dl_q4_pima_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q4/other/zoo/zoo_test1_data.csv -m ./dl/dlmodels/quantise/q4/other/zoo/zoo_train1.csv.cn2 -x abd -k ../rules/size/q4_zoo_train1.csv_size5.json > ../logs/dl/dl_q4_zoo_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q4/other/fairml/compas/compas_test1_data.csv -m ./dl/dlmodels/quantise/q4/other/fairml/compas/compas_train1.csv.cn2 -x abd -k ../rules/size/q4_compas_train1.csv_size5.json > ../logs/dl/dl_q4_compas_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q4/penn-ml/liver-disorder/liver-disorder_test1_data.csv -m ./dl/dlmodels/quantise/q4/penn-ml/liver-disorder/liver-disorder_train1.csv.cn2 -x abd -k ../rules/size/q4_liver-disorder_train1.csv_size5.json > ../logs/dl/dl_q4_liver-disorder_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q4/penn-ml/australian/australian_test1_data.csv -m ./dl/dlmodels/quantise/q4/penn-ml/australian/australian_train1.csv.cn2 -x abd -k ../rules/size/q4_australian_train1.csv_size5.json > ../logs/dl/dl_q4_australian_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q4/penn-ml/hungarian/hungarian_test1_data.csv -m ./dl/dlmodels/quantise/q4/penn-ml/hungarian/hungarian_train1.csv.cn2 -x abd -k ../rules/size/q4_hungarian_train1.csv_size5.json > ../logs/dl/dl_q4_hungarian_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q4/penn-ml/heart-c/heart-c_test1_data.csv -m ./dl/dlmodels/quantise/q4/penn-ml/heart-c/heart-c_train1.csv.cn2 -x abd -k ../rules/size/q4_heart-c_train1.csv_size5.json > ../logs/dl/dl_q4_heart-c_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q4/penn-ml/heart-statlog/heart-statlog_test1_data.csv -m ./dl/dlmodels/quantise/q4/penn-ml/heart-statlog/heart-statlog_train1.csv.cn2 -x abd -k ../rules/size/q4_heart-statlog_train1.csv_size5.json > ../logs/dl/dl_q4_heart-statlog_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q4/penn-ml/diabetes/diabetes_test1_data.csv -m ./dl/dlmodels/quantise/q4/penn-ml/diabetes/diabetes_train1.csv.cn2 -x abd -k ../rules/size/q4_diabetes_train1.csv_size5.json > ../logs/dl/dl_q4_diabetes_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q4/penn-ml/appendicitis/appendicitis_test1_data.csv -m ./dl/dlmodels/quantise/q4/penn-ml/appendicitis/appendicitis_train1.csv.cn2 -x abd -k ../rules/size/q4_appendicitis_train1.csv_size5.json > ../logs/dl/dl_q4_appendicitis_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q4/penn-ml/schizo/schizo_test1_data.csv -m ./dl/dlmodels/quantise/q4/penn-ml/schizo/schizo_train1.csv.cn2 -x abd -k ../rules/size/q4_schizo_train1.csv_size5.json > ../logs/dl/dl_q4_schizo_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q4/penn-ml/cleve/cleve_test1_data.csv -m ./dl/dlmodels/quantise/q4/penn-ml/cleve/cleve_train1.csv.cn2 -x abd -k ../rules/size/q4_cleve_train1.csv_size5.json > ../logs/dl/dl_q4_cleve_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q4/penn-ml/biomed/biomed_test1_data.csv -m ./dl/dlmodels/quantise/q4/penn-ml/biomed/biomed_train1.csv.cn2 -x abd -k ../rules/size/q4_biomed_train1.csv_size5.json > ../logs/dl/dl_q4_biomed_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q4/penn-ml/bupa/bupa_test1_data.csv -m ./dl/dlmodels/quantise/q4/penn-ml/bupa/bupa_train1.csv.cn2 -x abd -k ../rules/size/q4_bupa_train1.csv_size5.json > ../logs/dl/dl_q4_bupa_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q4/penn-ml/glass2/glass2_test1_data.csv -m ./dl/dlmodels/quantise/q4/penn-ml/glass2/glass2_train1.csv.cn2 -x abd -k ../rules/size/q4_glass2_train1.csv_size5.json > ../logs/dl/dl_q4_glass2_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q4/penn-ml/hepatitis/hepatitis_test1_data.csv -m ./dl/dlmodels/quantise/q4/penn-ml/hepatitis/hepatitis_train1.csv.cn2 -x abd -k ../rules/size/q4_hepatitis_train1.csv_size5.json > ../logs/dl/dl_q4_hepatitis_test1.csv_abd_bg_size5.log
python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q4/penn-ml/heart-h/heart-h_test1_data.csv -m ./dl/dlmodels/quantise/q4/penn-ml/heart-h/heart-h_train1.csv.cn2 -x abd -k ../rules/size/q4_heart-h_train1.csv_size5.json > ../logs/dl/dl_q4_heart-h_test1.csv_abd_bg_size5.log