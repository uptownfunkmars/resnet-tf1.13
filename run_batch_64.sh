# train_pos_pth="/home/sdf/xujiping/DataSet_bladder_12w/zzc_data/0"
# train_neg_pth="/home/sdf/xujiping/DataSet_bladder_12w/zzc_data/1"

train_pos_pth="/home/sdc/xujiping/train/1"
train_neg_pth="/home/sdc/xujiping/train/0"

model_sav_pth="/home/sde/xujiping/saved_model"
log_dir="/home/sde/xujiping/log"

gpu_number="7"

# python train_batch_64.py --epochs 50 --train_positive_pth ${train_pos_pth} --train_negative_pth ${train_neg_pth} --val_positive_pth ${val_pos_pth} --val_negative_pth ${val_neg_pth} --model_save_pth ${model_sav_pth} --log_dir ${log_dir}
python train_batch_64.py --epochs 50 --train_positive_pth ${train_pos_pth} --train_negative_pth ${train_neg_pth} --model_save_pth ${model_sav_pth} --log_dir ${log_dir} --gpu_number ${gpu_number}
