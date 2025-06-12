cd $PWD/scripts
export PYTHONPATH=$PWD/../..:$PYTHONPATH

# ==============Average Kernel==========================================================
# python run_MKL.py --kernel_level early --kernel_type rbf \
#     --dataset AD_CN  --modality PET GM CSF \
#     --kernel_combination AverageMKL 

# python run_MKL.py --kernel_level early --kernel_type rbf \
#     --dataset ROSMAP  --modality meth mRNA miRNA\
#     --kernel_combination AverageMKL 



# # ==============EasyMKL==========================================================
# python run_MKL.py --kernel_level early --kernel_type rbf \
#     --dataset AD_CN  --modality PET GM CSF \
#     --kernel_combination EasyMKL 

# python run_MKL.py --kernel_level early --kernel_type rbf \
#     --dataset ROSMAP  --modality meth mRNA miRNA\
#     --kernel_combination EasyMKL 


# # ==============EarlyFusionATK==========================================================
# python run_ATKQK.py --dataset AD_CN --modality concat_ADNI \
#     --lr 0.0005 --patience 30 --epochs 5000 \
#     --num_layers 4 --num_head 3 --top_k 1 \

# python run_ATKQK.py --dataset ROSMAP --modality concat_ROSMAP \
#     --lr 0.0005 --patience 30 --epochs 5000 \
#     --num_layers 4 --num_head 3 --top_k 1 \