import os
from datetime import datetime

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix, f1_score, roc_curve,
    classification_report
)

from ..model.ATK_noQK import ATK
from ..model.ATK_QK import ATKQK

def train_test(
        data,
        embedding_size=128,
        atk_params=None,
        num_epochs=100,
        lr=1e-3,
        random_seed=42,
        checkpoint_dir="checkpoints",
        early_stopping_patience=40,
        early_stopping_delta=0.001, 
        logger=None,
        use_QK=False,  # Flag to determine which model to use
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Log training configuration
    model_type = "ATKQK" if use_QK else "ATK"
    logger.info(f"Training Configuration for {model_type}:")
    logger.info(f"Embedding size: {embedding_size}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Random seed: {random_seed}")
    logger.info(f"ATK parameters: {atk_params}")
    logger.info(f"Early stopping patience: {early_stopping_patience}")

    # Reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    Y_test = data.y[test_mask].cpu().numpy()
    
    # Initialize the appropriate model
    if use_QK:
        model = ATKQK(
            num_nodes=data.num_nodes,
            in_dim=data.num_features,
            embedding_size=embedding_size,
            atk_params=atk_params,
        )
    else:
        model = ATK(
            num_nodes=data.num_nodes,
            in_dim=data.num_features,
            embedding_size=embedding_size,
            atk_params=atk_params,
        )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'lr': []}

    best_val_loss = float('inf')
    best_loss_epoch = 0
    best_loss_model_state = None

    model = model.to(device)
    data = data.to(device)

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits, attentions = model(data)
        loss = model.get_loss(logits[train_mask], data.y[train_mask], attentions,train_mask)
        loss.backward()
        optimizer.step()

        # Skip validation for early epochs only if not using QK or if explicitly configured
        if not use_QK and epoch < 50:
            continue

        model.eval()
        with torch.no_grad():
            logits, attentions = model(data)
            val_loss = model.get_loss(logits[val_mask], data.y[val_mask], attentions,val_mask).item()

        current_lr = optimizer.param_groups[0]['lr']
        history['loss'].append(loss.item())
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        logger.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"LR: {current_lr:.6f}"
        )

        scheduler.step(val_loss)

        loss_improved = val_loss < best_val_loss - early_stopping_delta

        if loss_improved:
            best_val_loss = val_loss
            best_loss_epoch = epoch
            best_loss_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }
            logger.info(f"Saved checkpoint with best val_loss: {best_val_loss:.4f} at epoch {best_loss_epoch}")
            no_improve_count = 0
        else:
            no_improve_count += 1
            logger.info(f"No val_loss improvement for {no_improve_count} epochs (best loss: {best_val_loss:.4f})")

        if no_improve_count >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch} epochs due to no val_loss improvement")
            break

    logger.info("Training completed!")

    if best_loss_model_state is not None:
        model.load_state_dict(best_loss_model_state['model_state_dict'])
        logger.info(f"Loaded best loss model from epoch {best_loss_epoch} with val_loss {best_val_loss:.4f}")

    best_models_state = {
        'best_loss': {
            'epoch': best_loss_epoch,
            'val_loss': best_val_loss,
            'model_state': best_loss_model_state
        }
    }

    result = {
        'best_models': {
            'best_loss_model': model
        },
        'history': history,
        'best_models_state': best_models_state
    }

    model.eval()
    with torch.no_grad():
        logits, attentions = model(data)
        test_logits = logits[test_mask]
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        fpr, tpr, thresholds = roc_curve(Y_test, test_probs)
        youden_index = np.argmax(tpr - fpr)
        optimal_threshold_youden = thresholds[youden_index]
        f1_scores = [f1_score(Y_test, (test_probs > t).astype(int)) for t in thresholds]
        optimal_threshold_f1 = thresholds[np.argmax(f1_scores)]
        
        # Use logger for both model types
        logger.info(f"Optimal threshold (Youden's J): {optimal_threshold_youden}")
        logger.info(f"Optimal threshold (F1 score): {optimal_threshold_f1}")
        test_preds = (test_probs > optimal_threshold_f1).astype(int)

        attention_maps = []
        for layer_attn in attentions:
            if len(layer_attn.shape) > 2:
                layer_attn = layer_attn.mean(dim=0)
            test_attends_to_all = layer_attn[test_mask, :].cpu().numpy()
            test_attends_to_test = layer_attn[test_mask, :][:, test_mask].cpu().numpy()
            attention_maps.append({
                'test_attends_to_all': test_attends_to_all,
                'test_attends_to_test': test_attends_to_test
            })

    prediction_results = {
        'probabilities': test_probs,
        'predictions': test_preds,
        'attention_maps': attention_maps
    }

    if Y_test is not None:
        accuracy = accuracy_score(Y_test, test_preds)
        cm = confusion_matrix(Y_test, test_preds)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        auc = roc_auc_score(Y_test, test_probs)

        prediction_results['metrics'] = {
            'confusion_matrix': cm.tolist(),
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'auc': auc,
            'classification_report': classification_report(Y_test, test_preds, output_dict=True),
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
        prediction_results['y_test'] = Y_test
        prediction_results['y_pred_prob'] = test_probs

        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"Test sensitivity: {sensitivity:.4f}")
        logger.info(f"Test specificity: {specificity:.4f}")
        logger.info(f"Test AUC: {auc:.4f}")
        logger.info(f"Confusion matrix:\n{cm}")

    result['prediction_results'] = prediction_results

    return result

# def train_test(
#         data,
#         embedding_size=128,
#         atk_params=None,
#         num_epochs=100,
#         lr=1e-3,
#         random_seed=42,
#         checkpoint_dir="checkpoints",
#         early_stopping_patience=40,
#         early_stopping_delta=0.001, 
#         logger=None,
#         fusion=False,  # Add fusion flag to handle multiple modalities
# ):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

#     # Log training configuration
#     logger.info("Training Configuration:")
#     logger.info(f"Embedding size: {embedding_size}")
#     logger.info(f"Number of epochs: {num_epochs}")
#     logger.info(f"Learning rate: {lr}")
#     logger.info(f"Random seed: {random_seed}")
#     logger.info(f"ATK parameters: {atk_params}")
#     logger.info(f"Early stopping patience: {early_stopping_patience}")
#     logger.info(f"Fusion mode: {fusion}")

#     # Reproducibility
#     np.random.seed(random_seed)
#     torch.manual_seed(random_seed)

#     if fusion:
#         # Multi-modality case
#         data_list = data  # data is expected to be a list of data objects
#         train_mask = data_list[0].train_mask
#         val_mask = data_list[0].val_mask
#         test_mask = data_list[0].test_mask
#         Y_test = data_list[0].y[test_mask].cpu().numpy()
        
#         # Get input dimensions for each modality
#         input_dims = [data_item.num_features for data_item in data_list]
         
#         model = ATKFusion(
#             num_nodes=data_list[0].num_nodes,
#             input_dims=input_dims,
#             embedding_size=embedding_size,
#             atk_params=atk_params,
#             prior_guides=[data_item.prior_guide for data_item in data_list],
#         )
#     else:
#         # Original single modality case
#         train_mask = data.train_mask
#         val_mask = data.val_mask
#         test_mask = data.test_mask
#         Y_test = data.y[test_mask].cpu().numpy()
        
#         model = ATK(
#             num_nodes=data.num_nodes,
#             in_dim=data.num_features,
#             embedding_size=embedding_size,
#             atk_params=atk_params,
#             prior_guide=data.prior_guide,
#         )
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
#     )

#     history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'lr': []}

#     best_val_loss = float('inf')
#     best_loss_epoch = 0
#     best_loss_model_state = None

#     model = model.to(device)
    
#     if fusion:
#         # Move all data objects to device
#         data_list = [d.to(device) for d in data_list]
#     else:
#         # Move single data object to device
#         data = data.to(device)

#     for epoch in range(1, num_epochs + 1):
#         model.train()
#         optimizer.zero_grad()

#         if fusion:
#             # Forward pass for multi-modality
#             combined_logits, attentions, modality_logits = model(data_list)
#             loss = model.get_loss(combined_logits[train_mask], data_list[0].y[train_mask], attentions, 
#                                   modality_logits=[m[train_mask] for m in modality_logits])
#         else:
#             # Original forward pass
#             logits, attentions = model(data)
#             loss = model.get_loss(logits[train_mask], data.y[train_mask], attentions)
#             combined_logits = logits
#             modality_logits = None
            
#         loss.backward()
#         optimizer.step()

#         if epoch < 50:
#             continue

#         model.eval()
#         with torch.no_grad():
#             if fusion:
#                 logits, attentions, modality_logits = model(data_list)
#                 val_loss = model.get_loss(logits[val_mask], data_list[0].y[val_mask], attentions,
#                                           modality_logits=[m[val_mask] for m in modality_logits]).item()
                
#                 # Evaluate accuracy for each individual modality
#                 y_val = data_list[0].y[val_mask].cpu().numpy()
#                 logger.info(f"Epoch {epoch}/{num_epochs} - Validation metrics:")
                
#                 # Combined model accuracy
#                 val_probs = torch.sigmoid(logits[val_mask]).cpu().numpy()
#                 val_preds = (val_probs > 0.5).astype(int)
#                 val_acc = accuracy_score(y_val, val_preds)
#                 logger.info(f"  Combined model - Accuracy: {val_acc:.4f}")
                
#                 # Individual modality accuracies
#                 for i, mod_logits in enumerate(modality_logits):
#                     mod_val_probs = torch.sigmoid(mod_logits[val_mask]).cpu().numpy()
#                     mod_val_preds = (mod_val_probs > 0.5).astype(int)
#                     mod_val_acc = accuracy_score(y_val, mod_val_preds)
#                     try:
#                         logger.info(f"  Modality {i+1} - Val Accuracy: {mod_val_acc:.4f}")
#                     except:
#                         logger.info(f"  Modality {i+1} - Val Accuracy: {mod_val_acc:.4f}")
#             else:
#                 logits, attentions = model(data)
#                 val_loss = model.get_loss(logits[val_mask], data.y[val_mask], attentions).item()

#         current_lr = optimizer.param_groups[0]['lr']
#         history['loss'].append(loss.item())
#         history['val_loss'].append(val_loss)
#         history['lr'].append(current_lr)

#         logger.info(
#             f"Epoch {epoch}/{num_epochs} - "
#             f"LR: {current_lr:.6f}"
#         )

#         scheduler.step(val_loss)

#         loss_improved = val_loss < best_val_loss - early_stopping_delta

#         if loss_improved:
#             best_val_loss = val_loss
#             best_loss_epoch = epoch
#             best_loss_model_state = {
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_loss': val_loss,
#             }
#             logger.info(f"Saved checkpoint with best val_loss: {best_val_loss:.4f} at epoch {best_loss_epoch}")
#             no_improve_count = 0
#         else:
#             no_improve_count += 1
#             logger.info(f"No val_loss improvement for {no_improve_count} epochs (best loss: {best_val_loss:.4f})")

#         if no_improve_count >= early_stopping_patience:
#             logger.info(f"Early stopping triggered after {epoch} epochs due to no val_loss improvement")
#             break

#     logger.info("Training completed!")

#     if best_loss_model_state is not None:
#         model.load_state_dict(best_loss_model_state['model_state_dict'])
#         logger.info(f"Loaded best loss model from epoch {best_loss_epoch} with val_loss {best_val_loss:.4f}")

#     best_models_state = {
#         'best_loss': {
#             'epoch': best_loss_epoch,
#             'val_loss': best_val_loss,
#             'model_state': best_loss_model_state
#         }
#     }

#     result = {
#         'best_models': {
#             'best_loss_model': model
#         },
#         'history': history,
#         'best_models_state': best_models_state
#     }

#     model.eval()
#     with torch.no_grad():
#         if fusion:
#             logits, attentions, modality_logits  = model(data_list)
#         else:
#             logits, attentions = model(data)

#         test_logits = logits[test_mask]
#         test_probs = torch.sigmoid(test_logits).cpu().numpy()
#         fpr, tpr, thresholds = roc_curve(Y_test, test_probs)
#         youden_index = np.argmax(tpr - fpr)
#         optimal_threshold_youden = thresholds[youden_index]
#         f1_scores = [f1_score(Y_test, (test_probs > t).astype(int)) for t in thresholds]
#         optimal_threshold_f1 = thresholds[np.argmax(f1_scores)]
#         print(f"Optimal threshold (Youden's J): {optimal_threshold_youden}")
#         print(f"Optimal threshold (F1 score): {optimal_threshold_f1}")
#         test_preds = (test_probs > optimal_threshold_f1).astype(int)

#         attention_maps = []
        
#         # Handle attention maps differently based on fusion mode
#         if fusion:
#             # For multi-modality: attentions is a list of lists (per modality)
#             for mod_idx, mod_attentions in enumerate(attentions):
#                 mod_attention_maps = []
#                 for layer_attn in mod_attentions:
#                     if isinstance(layer_attn, torch.Tensor):
#                         if len(layer_attn.shape) > 2:
#                             layer_attn = layer_attn.mean(dim=0)
#                         test_attends_to_all = layer_attn[test_mask, :].cpu().numpy()
#                         test_attends_to_test = layer_attn[test_mask, :][:, test_mask].cpu().numpy()
#                         mod_attention_maps.append({
#                             'test_attends_to_all': test_attends_to_all,
#                             'test_attends_to_test': test_attends_to_test,
#                             'modality': mod_idx
#                         })
#                 attention_maps.append({
#                     'modality': mod_idx,
#                     'layers': mod_attention_maps
#                 })
#         else:
#             # Original single-modality attention processing
#             for layer_attn in attentions:
#                 if len(layer_attn.shape) > 2:
#                     layer_attn = layer_attn.mean(dim=0)
#                 test_attends_to_all = layer_attn[test_mask, :].cpu().numpy()
#                 test_attends_to_test = layer_attn[test_mask, :][:, test_mask].cpu().numpy()
#                 attention_maps.append({
#                     'test_attends_to_all': test_attends_to_all,
#                     'test_attends_to_test': test_attends_to_test
#                 })

#     prediction_results = {
#         'probabilities': test_probs,
#         'predictions': test_preds,
#         'attention_maps': attention_maps
#     }

#     if Y_test is not None:
#         accuracy = accuracy_score(Y_test, test_preds)
#         cm = confusion_matrix(Y_test, test_preds)
#         tn, fp, fn, tp = cm.ravel()
#         sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#         specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#         auc = roc_auc_score(Y_test, test_probs)

#         prediction_results['metrics'] = {
#             'confusion_matrix': cm.tolist(),
#             'accuracy': accuracy,
#             'sensitivity': sensitivity,
#             'specificity': specificity,
#             'auc': auc,
#             'classification_report': classification_report(Y_test, test_preds, output_dict=True),
#             'fpr': fpr,
#             'tpr': tpr,
#             'thresholds': thresholds
#         }
#         prediction_results['y_test'] = Y_test
#         prediction_results['y_pred_prob'] = test_probs

#         logger.info(f"Test accuracy: {accuracy:.4f}")
#         logger.info(f"Test sensitivity: {sensitivity:.4f}")
#         logger.info(f"Test specificity: {specificity:.4f}")
#         logger.info(f"Test AUC: {auc:.4f}")
#         logger.info(f"Confusion matrix:\n{cm}")

#     result['prediction_results'] = prediction_results

#     return result


# def find_matching_module(source_model, target_model_list, source_name):
#     """Find matching module between models based on structure."""
#     source_module = getattr(source_model, source_name)
#     source_state = source_module.state_dict()
    
#     for target_module in target_model_list:
#         try:
#             target_module.load_state_dict(source_state)
#             return True
#         except:
#             continue
#     return False

# def train_test_progressive(
#         data,
#         embedding_size=128,
#         atk_params=None,
#         num_epochs=100,
#         lr=1e-3,
#         fine_tune_lr=5e-4,  # Smaller learning rate for fine-tuning
#         random_seed=42,
#         checkpoint_dir="checkpoints",
#         early_stopping_patience=40,
#         early_stopping_delta=0.001, 
#         logger=None,
# ):
#     """
#     Progressive training strategy for multi-modal ATK:
#     1. Train individual modality models first
#     2. Initialize fusion model with pre-trained components
#     3. Fine-tune the fusion model
    
#     Args:
#         data: List of PyG Data objects, one for each modality
#         embedding_size: Dimension of embeddings
#         atk_params: Parameters for ATK models
#         num_epochs: Number of training epochs per phase
#         lr: Learning rate for individual modality training
#         fine_tune_lr: Learning rate for fusion fine-tuning
#         random_seed: Random seed for reproducibility
#         checkpoint_dir: Directory to save checkpoints
#         early_stopping_patience: Patience for early stopping
#         early_stopping_delta: Minimum improvement for early stopping
#         logger: Logger object
#     """
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

#     # Log training configuration
#     logger.info("Progressive Training Configuration:")
#     logger.info(f"Embedding size: {embedding_size}")
#     logger.info(f"Initial training epochs: {num_epochs}")
#     logger.info(f"Fine-tuning epochs: {num_epochs // 2}")  # Fewer epochs for fine-tuning
#     logger.info(f"Initial learning rate: {lr}")
#     logger.info(f"Fine-tuning learning rate: {fine_tune_lr}")
#     logger.info(f"ATK parameters: {atk_params}")
    
#     # Reproducibility
#     np.random.seed(random_seed)
#     torch.manual_seed(random_seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(random_seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

#     # Extract masks and labels
#     data_list = data
#     train_mask = data_list[0].train_mask
#     val_mask = data_list[0].val_mask
#     test_mask = data_list[0].test_mask
#     Y_test = data_list[0].y[test_mask].cpu().numpy()
    
#     # PHASE 1: Train individual modality models
#     logger.info("===== PHASE 1: Training Individual Modality Models =====")
    
#     individual_models = []
#     individual_perfs = []
    
#     for i, modality_data in enumerate(data_list):
#         logger.info(f"Training modality {i+1}/{len(data_list)}")
        
#         # Create individual ATK model for this modality
#         model = ATK(
#             num_nodes=modality_data.num_nodes,
#             in_dim=modality_data.num_features,
#             embedding_size=embedding_size,
#             atk_params=atk_params,
#             prior_guide=modality_data.prior_guide,
#         )
        
#         # Print model's component names to debug
#         logger.info(f"Model components: {[name for name, _ in model.named_children()]}")
        
#         model = model.to(device)
#         modality_data = modality_data.to(device)
        
#         # Train individual model
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
#         )
        
#         best_val_loss = float('inf')
#         best_model_state = None
#         no_improve_count = 0
        
#         for epoch in range(1, num_epochs + 1):
#             # Training
#             model.train()
#             optimizer.zero_grad()
#             logits, attentions = model(modality_data)
#             loss = model.get_loss(logits[train_mask], modality_data.y[train_mask], attentions)
#             loss.backward()
#             optimizer.step()
            
#             # Skip validation for early epochs to allow more exploration
#             if epoch < min(20, num_epochs // 5):
#                 continue
                
#             # Validation
#             model.eval()
#             with torch.no_grad():
#                 logits, attentions = model(modality_data)
#                 val_loss = model.get_loss(logits[val_mask], modality_data.y[val_mask], attentions).item()
                
#                 # Calculate and log validation metrics periodically
#                 if epoch % 10 == 0:
#                     val_probs = torch.sigmoid(logits[val_mask]).cpu().numpy()
#                     val_preds = (val_probs > 0.5).astype(int)
#                     val_acc = accuracy_score(modality_data.y[val_mask].cpu().numpy(), val_preds)
#                     logger.info(f"Modality {i+1} - Epoch {epoch}/{num_epochs} - Val Loss: {val_loss:.4f} - Val ACC: {val_acc:.4f}")
            
#             scheduler.step(val_loss)
            
#             # Early stopping check
#             if val_loss < best_val_loss - early_stopping_delta:
#                 best_val_loss = val_loss
#                 best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}  # Save to CPU memory
#                 no_improve_count = 0
#             else:
#                 no_improve_count += 1
                
#             if no_improve_count >= early_stopping_patience:
#                 logger.info(f"Early stopping modality {i+1} after {epoch} epochs")
#                 break
        
#         # Load best model
#         if best_model_state:
#             model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
            
#         # Evaluate individual model
#         model.eval()
#         with torch.no_grad():
#             logits, _ = model(modality_data)
#             test_probs = torch.sigmoid(logits[test_mask]).cpu().numpy()
            
#             # Use Youden's index for optimal threshold
#             fpr, tpr, thresholds = roc_curve(Y_test, test_probs)
#             youden_index = np.argmax(tpr - fpr)
#             optimal_threshold = thresholds[youden_index] 
#             test_preds = (test_probs > optimal_threshold).astype(int)
            
#             accuracy = accuracy_score(Y_test, test_preds)
#             auc = roc_auc_score(Y_test, test_probs)
            
#             # Calculate sensitivity and specificity
#             cm = confusion_matrix(Y_test, test_preds)
#             tn, fp, fn, tp = cm.ravel()
#             sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#             specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
#         logger.info(f"Modality {i+1} Performance:")    
#         logger.info(f"  Test Accuracy: {accuracy:.4f}")
#         logger.info(f"  Test AUC: {auc:.4f}")
#         logger.info(f"  Sensitivity: {sensitivity:.4f}")
#         logger.info(f"  Specificity: {specificity:.4f}")
        
#         # Record model performance
#         individual_perfs.append({
#             'accuracy': accuracy,
#             'auc': auc,
#             'sensitivity': sensitivity,
#             'specificity': specificity
#         })
        
#         # Save the trained model for fusion initialization
#         individual_models.append(model)
    
#     # Check which modality performs best
#     best_modality_idx = np.argmax([perf['auc'] for perf in individual_perfs])
#     logger.info(f"Best performing individual modality: {best_modality_idx+1} with AUC {individual_perfs[best_modality_idx]['auc']:.4f}")
    
#     # PHASE 2: Initialize fusion model with pre-trained components
#     logger.info("\n===== PHASE 2: Fine-tuning Fusion Model =====")
    
#     # Get input dimensions
#     input_dims = [data_item.num_features for data_item in data_list]
    
#     # Create fusion model
#     fusion_model = ATKFusion(
#         num_nodes=data_list[0].num_nodes,
#         input_dims=input_dims,
#         embedding_size=embedding_size,
#         atk_params=atk_params, 
#         prior_guides=[data_item.prior_guide for data_item in data_list],
#     )
    
#     # Print fusion model's component names to debug
#     logger.info(f"Fusion model components: {[name for name, _ in fusion_model.named_children()]}")
    
#     # Copy pre-trained components to fusion model safely
#     try:
#         for i, indiv_model in enumerate(individual_models):
#             # Check component names
#             source_components = dict(indiv_model.named_children())
            
#             # The component naming may differ between ATK and ATKFusion - handle both cases
#             if hasattr(indiv_model, 'embedder'):
#                 logger.info(f"Copying embedder from modality {i+1}")
#                 fusion_model.embedders[i].load_state_dict(indiv_model.embedder.state_dict())
#             elif hasattr(indiv_model, 'embedding'):
#                 fusion_model.embedders[i].load_state_dict(indiv_model.embedding.state_dict()) 
                
#             if hasattr(indiv_model, 'atk'):
#                 logger.info(f"Copying ATK from modality {i+1}")
#                 fusion_model.atk_modules[i].load_state_dict(indiv_model.atk.state_dict())
#             else:
#                 # Try to find matching component by parameter count/shape
#                 logger.warning(f"Component name mismatch for modality {i+1} - trying to match by structure")
#                 for name, module in indiv_model.named_children():
#                     if 'atk' in name.lower():
#                         fusion_model.atk_modules[i].load_state_dict(module.state_dict())
#                         logger.info(f"Found ATK module with name '{name}'")
                        
#     except Exception as e:
#         logger.warning(f"Error copying pre-trained weights: {e}")
#         logger.warning("Continuing with random initialization...")
    
#     # Move fusion model to device
#     fusion_model = fusion_model.to(device)
#     data_list = [d.to(device) for d in data_list]
    
#     # PHASE 3: Fine-tune fusion model
#     # Freeze individual components initially
#     for embedder in fusion_model.embedders:
#         for param in embedder.parameters():
#             param.requires_grad = False
            
#     for atk_module in fusion_model.atk_modules:
#         for param in atk_module.parameters():
#             param.requires_grad = False
    
#     # First train only fusion components for a few epochs
#     optimizer = torch.optim.Adam(
#         list(fusion_model.fusion.parameters()) + 
#         list(fusion_model.classifier.parameters()), 
#         lr=fine_tune_lr
#     )
    
#     # Fine-tune for a subset of epochs
#     fine_tune_epochs = num_epochs // 2
    
#     logger.info("\n--- Stage 1: Training only fusion components ---")
#     best_val_loss_stage1 = float('inf')
    
#     for epoch in range(1, fine_tune_epochs // 2 + 1):
#         fusion_model.train()
#         optimizer.zero_grad()
        
#         logits, attentions = fusion_model(data_list)
#         loss = fusion_model.get_loss(logits[train_mask], data_list[0].y[train_mask], attentions)
        
#         loss.backward()
#         optimizer.step()
        
#         # Update temperature annealing if applicable
#         if hasattr(fusion_model, 'update_step'):
#             fusion_model.update_step(epoch)
            
#         # Validation every few epochs
#         if epoch % 5 == 0:
#             fusion_model.eval()
#             with torch.no_grad():
#                 logits, attentions = fusion_model(data_list)
#                 val_loss = fusion_model.get_loss(logits[val_mask], data_list[0].y[val_mask], attentions).item()
                
#                 if val_loss < best_val_loss_stage1:
#                     best_val_loss_stage1 = val_loss
                
#                 val_probs = torch.sigmoid(logits[val_mask]).cpu().numpy()
#                 val_preds = (val_probs > 0.5).astype(int)
#                 val_acc = accuracy_score(data_list[0].y[val_mask].cpu().numpy(), val_preds)
                
#                 logger.info(f"Fusion Stage 1 - Epoch {epoch}/{fine_tune_epochs//2} - "
#                             f"Loss: {loss.item():.4f} - Val Loss: {val_loss:.4f} - Val ACC: {val_acc:.4f}")
    
#     logger.info(f"Stage 1 complete - Best validation loss: {best_val_loss_stage1:.4f}")
    
#     # Now unfreeze all components and train end-to-end
#     for embedder in fusion_model.embedders:
#         for param in embedder.parameters():
#             param.requires_grad = True
            
#     for atk_module in fusion_model.atk_modules:
#         for param in atk_module.parameters():
#             param.requires_grad = True
    
#     # Smaller learning rate for pre-trained components
#     optimizer = torch.optim.Adam([
#         {'params': [p for embedder in fusion_model.embedders for p in embedder.parameters()], 
#          'lr': fine_tune_lr / 10},  # Even smaller LR for embedders
#         {'params': [p for atk_module in fusion_model.atk_modules for p in atk_module.parameters()], 
#          'lr': fine_tune_lr / 5},  # Small LR for ATK modules
#         {'params': fusion_model.fusion.parameters(), 'lr': fine_tune_lr},
#         {'params': fusion_model.classifier.parameters(), 'lr': fine_tune_lr}
#     ])
    
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
#     )
    
#     # Full fine-tuning
#     logger.info("\n--- Stage 2: End-to-end fine-tuning ---")
#     best_val_loss = float('inf')
#     best_model_state = None
#     no_improve_count = 0
    
#     # Add a max_steps parameter for MultiScaleKernelFusion if used
#     if hasattr(fusion_model, 'max_steps'):
#         fusion_model.max_steps = fine_tune_epochs
        
#     if hasattr(fusion_model, 'current_step'):
#         fusion_model.current_step = 0
    
#     for epoch in range(1, fine_tune_epochs + 1):
#         fusion_model.train()
#         optimizer.zero_grad()
        
#         logits, attentions = fusion_model(data_list)
#         loss = fusion_model.get_loss(logits[train_mask], data_list[0].y[train_mask], attentions)
        
#         loss.backward()
#         optimizer.step()
        
#         # Update temperature annealing
#         if hasattr(fusion_model, 'update_step'):
#             fusion_model.update_step(epoch)
        
#         # Validation
#         fusion_model.eval()
#         with torch.no_grad():
#             logits, attentions = fusion_model(data_list)
#             val_loss = fusion_model.get_loss(logits[val_mask], data_list[0].y[val_mask], attentions).item()
            
#             # Calculate validation metrics periodically
#             if epoch % 5 == 0:
#                 val_probs = torch.sigmoid(logits[val_mask]).cpu().numpy()
#                 val_preds = (val_probs > 0.5).astype(int)
#                 val_acc = accuracy_score(data_list[0].y[val_mask].cpu().numpy(), val_preds)
                
#                 logger.info(f"Fusion Stage 2 - Epoch {epoch}/{fine_tune_epochs} - "
#                             f"Loss: {loss.item():.4f} - Val Loss: {val_loss:.4f} - Val ACC: {val_acc:.4f}")
        
#         scheduler.step(val_loss)
        
#         # Early stopping check
#         if val_loss < best_val_loss - early_stopping_delta:
#             best_val_loss = val_loss
#             best_model_state = {k: v.cpu() for k, v in fusion_model.state_dict().items()}  # Save to CPU memory
#             no_improve_count = 0
#         else:
#             no_improve_count += 1
            
#         if no_improve_count >= early_stopping_patience // 2:  # Shorter patience for fine-tuning
#             logger.info(f"Early stopping fusion after {epoch} epochs")
#             break
    
#     # Load best model
#     if best_model_state:
#         fusion_model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

#     result = {
#         'best_models': {
#             'best_loss_model': fusion_model
#         }
#     }
    
#     # Final evaluation
#     fusion_model.eval()
#     with torch.no_grad():
#         logits, attentions = fusion_model(data_list)
        
#         test_logits = logits[test_mask]
#         test_probs = torch.sigmoid(test_logits).cpu().numpy()
#         fpr, tpr, thresholds = roc_curve(Y_test, test_probs)
        
#         # Calculate optimal threshold using Youden's index
#         youden_index = np.argmax(tpr - fpr)
#         optimal_threshold = thresholds[youden_index]
#         test_preds = (test_probs > optimal_threshold).astype(int)
        
#         # Calculate metrics
#         accuracy = accuracy_score(Y_test, test_preds)
#         auc = roc_auc_score(Y_test, test_probs)
#         cm = confusion_matrix(Y_test, test_preds)
#         tn, fp, fn, tp = cm.ravel()
#         sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#         specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#         f1 = f1_score(Y_test, test_preds)

    
#     # Compare with best individual model
#     best_indiv_perf = individual_perfs[best_modality_idx]
#     improvement = auc - best_indiv_perf['auc']
    
#     logger.info("\n===== Final Results =====")
#     logger.info(f"Best individual modality AUC: {best_indiv_perf['auc']:.4f}")
#     logger.info(f"Fusion model AUC: {auc:.4f} ({'+' if improvement > 0 else ''}{improvement:.4f})")
#     logger.info(f"Test Accuracy: {accuracy:.4f}")
#     logger.info(f"F1 Score: {f1:.4f}")
#     logger.info(f"Sensitivity: {sensitivity:.4f}")
#     logger.info(f"Specificity: {specificity:.4f}")
#     logger.info(f"Confusion matrix:\n{cm}")
    
#     # Process attention maps for visualization
#     attention_maps = []
#     # For multi-modality: attentions is a list of lists (per modality)
#     for mod_idx, mod_attentions in enumerate(attentions):
#         mod_attention_maps = []
#         for layer_attn in mod_attentions:
#             if isinstance(layer_attn, torch.Tensor):
#                 if len(layer_attn.shape) > 2:
#                     layer_attn = layer_attn.mean(dim=0)
#                 test_attends_to_all = layer_attn[test_mask, :].cpu().numpy()
#                 test_attends_to_test = layer_attn[test_mask, :][:, test_mask].cpu().numpy()
#                 mod_attention_maps.append({
#                     'test_attends_to_all': test_attends_to_all,
#                     'test_attends_to_test': test_attends_to_test,
#                     'modality': mod_idx
#                 })
#         attention_maps.append({
#             'modality': mod_idx,
#             'layers': mod_attention_maps
#         })
    
#     # Create prediction results with ALL required keys
#         prediction_results = {
#             'probabilities': test_probs,
#             'predictions': test_preds,
#             'attention_maps': attention_maps,
#             'y_test': Y_test,
#             'y_pred_prob': test_probs,
#             'metrics': {
#                 'confusion_matrix': cm.tolist(),
#                 'accuracy': accuracy,
#                 'auc': auc,
#                 'sensitivity': sensitivity, 
#                 'specificity': specificity,
#                 'f1': f1,
#                 'classification_report': classification_report(Y_test, test_preds, output_dict=True),
#                 'fpr': fpr,
#                 'tpr': tpr,
#                 'thresholds': thresholds,
#                 'optimal_threshold': optimal_threshold
#             }
#         }

#         result['prediction_results'] = prediction_results
    
#     return result

