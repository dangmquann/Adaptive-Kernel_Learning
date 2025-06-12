cd $PWD/scripts
export PYTHONPATH=$PWD/../..:$PYTHONPATH

# ##============= KERNELS SVM=======================================================================
# python run_early_kernel.py --kernel_level early --kernel_type rbf\
#    --dataset AD_CN  --modality PET

# python run_early_kernel.py --kernel_level early --kernel_type rbf\
#    --dataset AD_CN  --modality GM

# python run_early_kernel.py --kernel_level early --kernel_type linear\
#    --dataset AD_CN  --modality CSF

# python run_early_kernel.py --kernel_level early --kernel_type rbf\
#    --dataset AD_CN  --modality MRI

# python run_early_kernel.py --kernel_level early --kernel_type rbf\
#    --dataset ROSMAP  --modality meth

# python run_early_kernel.py --kernel_level early --kernel_type rbf\
#    --dataset ROSMAP  --modality mRNA

# python run_early_kernel.py --kernel_level early --kernel_type rbf\
#    --dataset ROSMAP  --modality miRNA





# # ================== Gaussian Process ==========================================================
   
# python run_GP.py  --dataset AD_CN  --modality PET
# python run_GP.py  --dataset AD_CN  --modality GM
# python run_GP.py  --dataset AD_CN  --modality CSF
# python run_GP.py  --dataset ROSMAP --modality meth
# python run_GP.py  --dataset ROSMAP --modality miRNA
# python run_GP.py  --dataset ROSMAP --modality mRNA
# #



# # ================== Deep Kernel Learning ========================================================================
# ## AD_CN with PET modality
# python run_DKL.py --dataset AD_CN --modality PET \
#    --epochs 10000 --batch_size 64 \
#    --patience 20 --lr 0.01 \

# python run_DKL.py --dataset AD_CN --modality GM \
#    --epochs 10000 --batch_size 32 \
#    --patience 20 --lr 0.01 \

# # Need to fine tune hidden dimensions of feature extractor [3]
# python run_DKL.py --dataset AD_CN --modality CSF \
#    --epochs 10000 --batch_size 32 \
#    --patience 20 --lr 0.01 \

# # ROSMAP with meth modality
# python run_DKL.py --dataset ROSMAP --modality meth \
#    --epochs 10000 --batch_size 32 \
#    --patience 20 --lr 0.01 \

# # ROSMAP with miRNA modality
# python run_DKL.py --dataset ROSMAP --modality miRNA \
#    --epochs 10000 --batch_size 32 \
#    --patience 20 --lr 0.01 \

# # ROSMAP with mRNA modality
# python run_DKL.py --dataset ROSMAP --modality mRNA \
#    --epochs 10000 --batch_size 32 \
#    --patience 20 --lr 0.01 \
# #






# #================= Self-Attention ========================================================================
# python run_MHA.py --dataset AD_CN --modality PET \
#    --lr 0.001 --patience 20 --epochs 5000 \
#    --num_layers 5 --num_head 4

# python run_MHA.py --dataset AD_CN --modality GM \
#    --lr 0.001 --patience 20 --epochs 5000 \
#    --num_layers 5 --num_head 4

# python run_MHA.py --dataset AD_CN --modality CSF \
#    --lr 0.001 --patience 20 --epochs 5000 \
#    --num_layers 5 --num_head 4

# python run_MHA.py --dataset AD_CN --modality MRI \
#    --lr 0.001 --patience 20 --epochs 5000 \
#    --num_layers 5 --num_head 4

# python run_MHA.py --dataset ROSMAP --modality meth \
#    --lr 0.001 --patience 20 --epochs 5000 \
#    --num_layers 3 --num_head 4

# python run_MHA.py --dataset ROSMAP --modality mRNA \
#    --lr 0.001 --patience 20 --epochs 5000 \
#    --num_layers 3 --num_head 4

# python run_MHA.py --dataset ROSMAP --modality miRNA \
#    --lr 0.001 --patience 20 --epochs 5000 \
#    --num_layers 3 --num_head 4


#================== Linformer ========================================================================
# python run_linformer.py --dataset AD_CN --modality PET \
#    --lr 0.001 --patience 20 --epochs 5000 \
#    --num_layers 5 --num_head 4
# python run_linformer.py --dataset AD_CN --modality PET \
#    --lr 0.001 --patience 20 --epochs 5000 \
#    --num_layers 5 --num_head 4

# python run_linformer.py --dataset AD_CN --modality GM \
#    --lr 0.001 --patience 20 --epochs 5000 \
#    --num_layers 5 --num_head 4

# python run_linformer.py --dataset AD_CN --modality CSF \
#    --lr 0.001 --patience 20 --epochs 5000 \
#    --num_layers 5 --num_head 4

# python run_linformer.py --dataset ROSMAP --modality meth \
#    --lr 0.001 --patience 20 --epochs 5000 \
#    --num_layers 3 --num_head 4

# python run_linformer.py --dataset ROSMAP --modality mRNA \
#    --lr 0.001 --patience 20 --epochs 5000 \
#    --num_layers 3 --num_head 4

# python run_linformer.py --dataset ROSMAP --modality miRNA \
#    --lr 0.001 --patience 20 --epochs 5000 \
#    --num_layers 3 --num_head 4




# #================== Performer ========================================================================
# python run_performer.py --dataset AD_CN --modality PET \
#    --lr 0.001 --patience 20 --epochs 5000 \
#    --num_layers 5 --num_head 4

# python run_performer.py --dataset AD_CN --modality GM \
#    --lr 0.001 --patience 20 --epochs 5000 \
#    --num_layers 5 --num_head 4

# python run_performer.py --dataset AD_CN --modality CSF \
#    --lr 0.001 --patience 20 --epochs 5000 \
#    --num_layers 5 --num_head 4

# python run_performer.py --dataset ROSMAP --modality meth \
#    --lr 0.001 --patience 20 --epochs 5000 \
#    --num_layers 3 --num_head 4

# python run_performer.py --dataset ROSMAP --modality mRNA \
#    --lr 0.001 --patience 20 --epochs 5000 \
#    --num_layers 3 --num_head 4

# python run_performer.py --dataset ROSMAP --modality miRNA \
#    --lr 0.001 --patience 20 --epochs 5000 \
#    --num_layers 3 --num_head 4


# # #================ ATK with QK ========================================================================
# python run_ATKQK.py --dataset AD_CN --modality PET \
#    --lr 0.0004 --patience 30 --epochs 5000 --reg_coef 0.1 \
#    --num_layers 4 --num_head 3 --top_k 2 \

# python run_ATKQK.py --dataset AD_CN --modality GM \
#    --lr 0.0004 --patience 30 --epochs 5000 --reg_coef 0.1\
#    --num_layers 4 --num_head 3 --top_k 2 \

# python run_ATKQK.py --dataset AD_CN --modality CSF \
#    --lr 0.0004 --patience 30 --epochs 5000 --reg_coef 0.1\
#    --num_layers 4 --num_head 3 --top_k 2 \

# python run_ATKQK.py --dataset AD_CN --modality MRI \
#    --lr 0.0004 --patience 30 --epochs 5000 --reg_coef 0.1\
#    --num_layers 4 --num_head 3 --top_k 2 \

# python run_ATKQK.py --dataset ROSMAP --modality mRNA \
#    --lr 0.0004 --patience 30 --epochs 5000 --reg_coef 0.1\
#    --num_layers 4 --num_head 3 --top_k 2 \

# python run_ATKQK.py --dataset ROSMAP --modality miRNA \
#    --lr 0.0004 --patience 30 --epochs 5000 --reg_coef 0.1\
#    --num_layers 4 --num_head 3 --top_k 2 \

# python run_ATKQK.py --dataset ROSMAP --modality meth \
#    --lr 0.0004 --patience 30 --epochs 5000 --reg_coef 0.1\
#    --num_layers 4 --num_head 3 --top_k 2 \





# #================ ATK without QK ========================================================================
# #prior method  1 is the combined kernel, 2 is the prior kernel

# #==== RANDOM ======================

# python run_ATK_noQK.py --dataset AD_CN --modality PET GM CSF --selected_modality PET \
#    --use_prior False --reg_coef 0.1 \
#    --lr 0.001 --patience 20 --epochs 2000 \
#    --num_layers 5 --num_head 4 --top_k 2 \

# python run_ATK_noQK.py --dataset AD_CN --modality PET GM CSF --selected_modality GM \
#    --use_prior False --reg_coef 0.1\
#    --lr 0.0005 --patience 20 --epochs 2000 \
#    --num_layers 5 --num_head 4 --top_k 2 \

# python run_ATK_noQK.py --dataset AD_CN --modality PET GM CSF --selected_modality CSF \
#    --use_prior False --reg_coef 0.1\
#    --lr 0.001 --patience 20 --epochs 2000 \
#    --num_layers 5 --num_head 4 --top_k 2 \

# python run_ATK_noQK.py --dataset ROSMAP --modality meth mRNA miRNA --selected_modality meth \
#     --use_prior False --reg_coef 0.1\
#     --lr 0.001 --patience 20 --epochs 2000 \
#     --num_layers 3 --num_head 4 --top_k 2 \

# python run_ATK_noQK.py --dataset ROSMAP --modality meth mRNA miRNA --selected_modality mRNA \
#    --use_prior False --reg_coef 0.1\
#    --lr 0.001 --patience 20 --epochs 200 \
#    --num_layers 3 --num_head 4 --top_k 2 \

# python run_ATK_noQK.py --dataset ROSMAP --modality meth mRNA miRNA --selected_modality miRNA \
#    --use_prior False --reg_coef 0.1\
#    --lr 0.001 --patience 20 --epochs 200 \
#    --num_layers 3 --num_head 4 --top_k 2 \




# # #==== PRIOR KERNEL ================

# python run_ATK_noQK.py --dataset AD_CN --modality PET GM CSF --selected_modality PET \
#    --use_prior True --prior_method 2 --reg_coef 0.1 \
#    --lr 0.001 --patience 20 --epochs 2000 \
#    --num_layers 5 --num_head 4 --top_k 2 \

# python run_ATK_noQK.py --dataset AD_CN --modality PET GM CSF --selected_modality GM \
#    --use_prior True --prior_method 2 --reg_coef 0.1\
#    --lr 0.0005 --patience 20 --epochs 2000 \
#    --num_layers 5 --num_head 4 --top_k 2 \

# python run_ATK_noQK.py --dataset AD_CN --modality PET GM CSF --selected_modality CSF \
#    --use_prior True --prior_method 2 --reg_coef 0.1\
#    --lr 0.001 --patience 20 --epochs 2000 \
#    --num_layers 5 --num_head 4 --top_k 2 \

# python run_ATK_noQK.py --dataset ROSMAP --modality meth mRNA miRNA --selected_modality meth \
#     --use_prior True --prior_method 2 --reg_coef 0.1\
#     --lr 0.001 --patience 20 --epochs 2000 \
#     --num_layers 3 --num_head 4 --top_k 2 \

# python run_ATK_noQK.py --dataset ROSMAP --modality meth mRNA miRNA --selected_modality mRNA \
#    --use_prior True --prior_method 2 --reg_coef 0.1\
#    --lr 0.001 --patience 20 --epochs 200 \
#    --num_layers 3 --num_head 4 --top_k 2 \

# python run_ATK_noQK.py --dataset ROSMAP --modality meth mRNA miRNA --selected_modality miRNA \
#    --use_prior True --prior_method 2 --reg_coef 0.1\
#    --lr 0.001 --patience 20 --epochs 200 \
#    --num_layers 3 --num_head 4 --top_k 2 \


# #==== COMBINED KERNEL =============
# python run_ATK_noQK.py --dataset AD_CN --modality PET GM CSF --selected_modality PET \
#    --use_prior True --prior_method 1 --reg_coef 0.1 \
#    --lr 0.001 --patience 20 --epochs 2000 \
#    --num_layers 5 --num_head 4 --top_k 2 \

# python run_ATK_noQK.py --dataset AD_CN --modality PET GM CSF --selected_modality GM \
#    --use_prior True --prior_method 1 --reg_coef 0.1\
#    --lr 0.0005 --patience 20 --epochs 2000 \
#    --num_layers 5 --num_head 4 --top_k 2 \

# python run_ATK_noQK.py --dataset AD_CN --modality PET GM CSF --selected_modality CSF \
#    --use_prior True --prior_method 1 --reg_coef 0.1\
#    --lr 0.001 --patience 20 --epochs 2000 \
#    --num_layers 5 --num_head 4 --top_k 2 \

# python run_ATK_noQK.py --dataset ROSMAP --modality meth mRNA miRNA --selected_modality meth \
#     --use_prior True --prior_method 1 --reg_coef 0.1\
#     --lr 0.001 --patience 20 --epochs 2000 \
#     --num_layers 3 --num_head 4 --top_k 2 \

# python run_ATK_noQK.py --dataset ROSMAP --modality meth mRNA miRNA --selected_modality mRNA \
#    --use_prior True --prior_method 1 --reg_coef 0.1\
#    --lr 0.001 --patience 20 --epochs 200 \
#    --num_layers 3 --num_head 4 --top_k 2 \

# python run_ATK_noQK.py --dataset ROSMAP --modality meth mRNA miRNA --selected_modality miRNA \
#    --use_prior True --prior_method 1 --reg_coef 0.1\
#    --lr 0.001 --patience 20 --epochs 200 \
#    --num_layers 3 --num_head 4 --top_k 2 \
