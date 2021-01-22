# train_pos_pth="/home/sdf/xujiping/DataSet_bladder_12w/zzc_data/0"
# train_neg_pth="/home/sdf/xujiping/DataSet_bladder_12w/zzc_data/1"

# train_pos_pth="/home/sdc/xujiping_sdf/data/train_shila/1"
# train_neg_pth="/home/sdc/xujiping_sdf/data/train_shila/0"

data_pth="/home/sdg/4class_shila_patch"

model_sav_pth="/home/sdc/xujiping_sde/saved_model_resnet_four"
log_dir="/home/sdc/xujiping_sde/log_resnet_four"

gpu_number="9"

# python train_batch_64.py --epochs 50 --train_positive_pth ${train_pos_pth} --train_negative_pth ${train_neg_pth} --val_positive_pth ${val_pos_pth} --val_negative_pth ${val_neg_pth} --model_save_pth ${model_sav_pth} --log_dir ${log_dir}
python train_resnet_64_four.py --epochs 20 --data_pth ${data_pth}  --model_save_pth ${model_sav_pth} --log_dir ${log_dir} --gpu_number ${gpu_number}
