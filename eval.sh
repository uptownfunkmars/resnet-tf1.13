# test_data_pth="/home/sdf/xujiping/tmb_bladder/data/HL_data/train"
positive_pth="/home/sdc/xujiping/train/1"
negative_pth="/home/sdc/xujiping/train/0"
model_sav_pth="/home/sde/xujiping/saved_model/epoch_1.ckpt"

python eval.py --positive_pth ${positive_pth} --negative_pth ${negative_pth} --model_saved_pth ${model_sav_pth}




