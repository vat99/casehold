#!/bin/bash

#public
type="pile"
seeds=(42 125380 160800 22758 176060 193228)

start=`date +%s`
for folder in ~/casehold/checkpoints/downstream_eval/${type}/*/; do
    echo "${folder}"
    values=$(echo "${folder}" | cut -d'/' -f8)
    #echo "${values}"
    wd=$(echo "${values}" | cut -d'_' -f2 | cut -f2 -d"y") # decay0.005
    lr=$(echo "${values}" | cut -d'_' -f3 | cut -f2 -d"r") # lr5e-6
    echo "wd: ${wd}, lr: ${lr}"
    for seed in ${seeds[@]}; do
        run_folder="new_downstream_eval_gridsearch/${type}/downstream_eval_grid_search_${lr}_${wd}/"
        echo "${run_folder}"
        mkdir ${run_folder}
        log_file="new_downstream_eval_gridsearch/${type}/downstream_eval_grid_search_${lr}_${wd}/downstream_eval_grid_search_${lr}_${wd}_${seed}.txt"
        echo "${log_file}"
        accelerate launch --config_file run.yaml multiple_choice/mc_no_trainer.py \
            --config_name "bert-large-uncased" \
            --tokenizer_name "bert-large-uncased" \
            --model_name_or_path "${folder}" \
            --max_seq_length 128 \
            --per_device_train_batch_size=16 \
            --learning_rate=1e-5 \
            --num_train_epochs=6 \
            --output_dir ${run_folder} \
            --weight_decay=0.0005 \
            --seed=${seed} 2>&1 | tee ${log_file}
    done
done
end=`date +%s`
runtime=$((end-start))
echo "took ${runtime}"
sudo poweroff