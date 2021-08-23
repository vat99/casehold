#!/bin/bash

# myArray=("/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_public/hfrun_decay0.0005_lr5e-5_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/" "pretraining_grid_public/hfrun_decay0.0005_lr5e-6_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_public/hfrun_decay0.0005_lr5e-7_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_public/hfrun_decay0.001_lr1e-5_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_public/hfrun_decay0.001_lr1e-6_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_public/hfrun_decay0.001_lr5e-5_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_public/hfrun_decay0.001_lr5e-6_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_public/hfrun_decay0.001_lr5e-7_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_public/hfrun_decay0.005_lr1e-5_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_public/hfrun_decay0.005_lr1e-6_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_public/hfrun_decay0.005_lr5e-5_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_public/hfrun_decay0.005_lr5e-6_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_public/hfrun_decay0.005_lr5e-7_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_public/hfrun_decay0.01_lr1e-5_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_public/hfrun_decay0.01_lr1e-6_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_public/hfrun_decay0.01_lr5e-5_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_public/hfrun_decay0.01_lr5e-6_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_public/hfrun_decay0.01_lr5e-7_ss512_bs256_results_finbert_dir")

# #for folder in /scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_public/*; do
# for folder in ${myArray[@]}; do
#     echo "${folder}"
#     run_folder=$(echo "${folder}/" | cut -d'/' -f8)
#     echo "${run_folder}"
#     log_file="downstream_eval_gridsearch_results/public/downstreamgrid_${run_folder}.txt"
#     echo "${log_file}"
#     CUDA_VISIBLE_DEVICES=0 python3 -u run_seeds.py --model_name_or_path ${folder} 2>&1 --start_seed 0 --end_seed 4 --type public | tee ${log_file}
#     rm -rf /scratch/varunt/finbert_clpath/public
# done

#weightDecays=(0.0005 0.001 0.005 0.01)
weightDecays=(0.1 0.05)
#weightDecays=(0.1)
#learningRates=(5e-4 5e-5 1e-5 5e-6 5e-7)
learningRates=(1e-5 5e-6)
#learningRates=(5e-4)
#seeds=(42)
seeds=(42 125380 160800 22758 176060 193228)

# public
start=`date +%s`
for lr in ${learningRates[@]}; do
    for wd in ${weightDecays[@]}; do
        for seed in ${seeds[@]}; do
            run_folder="downstream_config_gridsearch/public/downstream_config_grid_search_${lr}_${wd}/"
            echo "${run_folder}"
            mkdir ${run_folder}
            log_file="downstream_config_gridsearch/public/downstream_config_grid_search_${lr}_${wd}/downstream_config_grid_search_${lr}_${wd}_${seed}.txt"
            echo "${log_file}"
            #export DATASET_NAME=casehold
            # CUDA_VISIBLE_DEVICES=0 python
            # multiple_choice/run_multiple_choice.py \
            #     --task_name casehold \
            #     --model_name_or_path "bert-large-uncased" \
            #     --data_dir data_processed/ \
            #     --do_train \
            #     --do_eval \
            #     --evaluation_strategy steps \
            #     --max_seq_length 128 \
            #     --per_device_train_batch_size=4 \
            #     --learning_rate=${lr} \
            #     --num_train_epochs=1.0 \
            #     --output_dir ${run_folder} \
            #     --overwrite_output_dir \
            #     --save_strategy "no" \
            #     --logging_strategy "epoch" \
            #     --weight_decay=${wd} \
            #     --cache_dir "cache/" \
            #     --seed=${seed} 2>&1 | tee ${log_file}
            accelerate launch --config_file run.yaml multiple_choice/mc_no_trainer.py \
                --config_name "bert-large-uncased" \
                --tokenizer_name "bert-large-uncased" \
                --model_name_or_path "bert-large-uncased" \
                --max_seq_length 128 \
                --per_device_train_batch_size=16 \
                --learning_rate=${lr} \
                --num_train_epochs=6 \
                --output_dir ${run_folder} \
                --weight_decay=${wd} \
                --seed=${seed} 2>&1 | tee ${log_file}
        done
    done
done
end=`date +%s`
runtime=$((end-start))
echo "took ${runtime}"

#pile

start=`date +%s`
for lr in ${learningRates[@]}; do
    for wd in ${weightDecays[@]}; do
        for seed in ${seeds[@]}; do
            run_folder="downstream_config_gridsearch/pile/downstream_config_grid_search_${lr}_${wd}/"
            echo "${run_folder}"
            mkdir ${run_folder}
            log_file="downstream_config_gridsearch/pile/downstream_config_grid_search_${lr}_${wd}/downstream_config_grid_search_${lr}_${wd}_${seed}.txt"
            echo "${log_file}"
            #export DATASET_NAME=casehold
            # CUDA_VISIBLE_DEVICES=0 python
            # multiple_choice/run_multiple_choice.py \
            #     --task_name casehold \
            #     --model_name_or_path "bert-large-uncased" \
            #     --data_dir data_processed/ \
            #     --do_train \
            #     --do_eval \
            #     --evaluation_strategy steps \
            #     --max_seq_length 128 \
            #     --per_device_train_batch_size=4 \
            #     --learning_rate=${lr} \
            #     --num_train_epochs=1.0 \
            #     --output_dir ${run_folder} \
            #     --overwrite_output_dir \
            #     --save_strategy "no" \
            #     --logging_strategy "epoch" \
            #     --weight_decay=${wd} \
            #     --cache_dir "cache/" \
            #     --seed=${seed} 2>&1 | tee ${log_file}
            accelerate launch --config_file run.yaml multiple_choice/mc_no_trainer.py \
                --config_name "bert-large-uncased" \
                --tokenizer_name "bert-large-uncased" \
                --model_name_or_path "checkpoints/downstream_config/pile/" \
                --max_seq_length 128 \
                --per_device_train_batch_size=16 \
                --learning_rate=${lr} \
                --num_train_epochs=6 \
                --output_dir ${run_folder} \
                --weight_decay=${wd} \
                --seed=${seed} 2>&1 | tee ${log_file}
        done
    done
done
end=`date +%s`
runtime=$((end-start))
echo "took ${runtime}"