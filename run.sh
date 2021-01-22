train_pos_pth="/home/sdf/xujiping/tmb_bladder/data/HL_data/train/0"
train_neg_pth="/home/sdf/xujiping/tmb_bladder/data/HL_data/train/1"

val_pos_pth="/home/sdf/xujiping/tmb_bladder/data/HL_data/val/0"
val_neg_pth="/home/sdf/xujiping/tmb_bladder/data/HL_data/val/1"

model_sav_pth="/home/sdf/xujiping/tmb_bladder/zzc/saved_model"
log_dir="/home/sdf/xujiping/tmb_bladder/zzc/log"

python train.py --epochs 50 --train_positive_pth ${train_pos_pth} --train_negative_pth ${train_neg_pth} --val_positive_pth ${val_pos_pth} --val_negative_pth ${val_neg_pth} --model_save_pth ${model_sav_pth} --log_dir ${log_dir}
