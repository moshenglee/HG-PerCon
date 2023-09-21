# pre-train
#given example
# CUDA_VISIBLE_DEVICES=0 python3 pretrain.py --lr 8e-4 --dropout 0.5 --weight-decay 5e-4 --batch X --patience 20 --feature-size 32,256 --data-dir input/youtube_gat --pkl-dir model_save/youtube_0318_pretrain_rmse --pkl-dir2 model_save/youtube_0318_pretrain_fine_rmse --loss-fn rmse
## joint
# CUDA_VISIBLE_DEVICES=2 python3 traincljoint.py --lr 8e-4 --dropout 0.5 --weight-decay 5e-4 --batch X --patience 20 --feature-size 32,256 --data-dir input/youtube_gat --pkl-dir model_save/youtube_0318_joint_rmse --loss-fn rmse