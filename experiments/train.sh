cd $PWD/scripts
export PYTHONPATH=$PWD/../..:$PYTHONPATH

# #===== EARLY KERNELS ======
# python run_early_kernel.py --kernel_level early --dataset AD_CN\
#     --kernel_type rbf --modality PET \
#     --version 4

# python run_GP.py  --dataset ROSMAP\
#     --modality meth \
#     --version gaussian_process
#
# python run_DKL.py --dataset ROSMAP --modality meth  \
#     --epochs 100 --batch_size 64 \
#     --version DKLs

#
# python run_DKL.py --dataset AD_CN --modality PET  \
#     --epochs 1000 --batch_size 64 \
#     --version DKL


# ===== COMBINATION KERNELS ======
# python run_MKL.py --kernel_level early --dataset AD_CN\
#     --kernel_type rbf --modality GM PET CSF\
#     --kernel_combination sum --version 1

# python run_MKL.py --kernel_level early --dataset ROSMAP\
#     --kernel_type linear --modality meth miRNA mRNA\
#     --kernel_combination AverageMKL --version average_mkl
#



# # SMKN FOR APPROACHE 1    # --use_fe
# python run_MKL.py --kernel_level early --dataset AD_CN\
#     --kernel_type rbf --modality PET\
#     --kernel_combination SKL \
#     --pretrain_epochs 500 --use_fe \
#     --num_kernels 8 --lr 0.001 --epochs 30  \
#     --patience 100 --batch_size 64 --lambda_reg 1e-3  \
#     --version 1


# # SMKN FOR APPROACHE 2
# python run_MKL.py --kernel_level early --dataset AD_CN\
#     --kernel_type rbf --modality GM PET CSF\
#     --kernel_combination AMKL \
#     --num_kernels 8 --lr 0.001 \
#     --epochs 30 --pretrain_epochs 700 --use_fe \
#     --patience 5 --batch_size 512 --lambda_reg 1e-3\
#     --version akl


#Deep Kernel Learning
# python run_MKL.py --kernel_level early --dataset AD_CN\
#     --kernel_type rbf --modality PET\
#     --kernel_combination DKL \
#     --num_kernels 15 --lr 0.001 --epochs 200\
#     --batch_size 64 \
#     --version 1


# # MOGONET
# python main_mogonet.py --kernel_level early --dataset AD_CN\
#     --kernel_type rbf --modality PET\
#     --kernel_combination SKL --classifier MOGONET \
#     --num_kernels 8 --lr 0.001 --epochs 30 --patience 10 --batch_size 64 --lambda_reg 1e-3  \
#     --version 1


#  GCT miRNA mRNA
# python run_ATK.py --kernel_level early --dataset ROSMAP\
#     --modality meth miRNA mRNA\
#     --lr 0.003 --epochs 300 --patience 10 \
#     --top_k 2 \
#     --version 3

# python run_ATK.py --kernel_level early --dataset AD_CN\
#     --modality PET\
#     --lr 0.001 --epochs 1000 --patience 50 \
#     --top_k 2 \
#     --version 5

# ============== LATE FUSION ==========================================================
# python run_late_FusionATK.py --kernel_level early --dataset ROSMAP\
#     --modality meth miRNA mRNA\
#     --lr 0.001 --epochs 300 --patience 50 \
#     --top_k 2 \
#     --version 3

# python run_late_FusionATK.py --kernel_level early --dataset AD_CN\
#     --modality PET GM \
#     --lr 0.003 --epochs 300 --patience 10 \
#     --top_k 2 \
#     --version 3
