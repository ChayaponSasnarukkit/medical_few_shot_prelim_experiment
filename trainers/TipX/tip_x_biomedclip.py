import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import numpy as np

from tqdm import tqdm
from trainers.method import FSCLIPmethod
from trainers.utils import build_cache_model, search_hp_tip_biomedclip, cls_acc

def _compute_distributions(features, text_weights, temp=0.5):
    """Computes softmax class distributions for given features."""
    # The temperature `temp` can be tuned; 0.5 is a common default.
    dists = 85.223 * features @ text_weights
    dists = F.softmax(dists / temp, dim=1)
    return dists

def _scale(x, target_range_tensor):
    """Scales tensor x to the min/max range of target_range_tensor."""
    x_min, x_max = x.min(), x.max()
    target_min, target_max = target_range_tensor.min(), target_range_tensor.max()
    
    if (x_max - x_min) == 0:
        return torch.full_like(x, (target_min + target_max) / 2) # Return mid-point of target
        
    y = (x - x_min) / (x_max - x_min)
    y = y * (target_max - target_min) + target_min
    return y

# +++ REFACTORED KL-Divergence Function +++
def _compute_scaled_kl_affinity(eval_dists, support_dists, affinity_for_scaling):
    """
    Computes KL-divergence, negates it, and scales it to the range of the
    provided affinity matrix, fully implementing the logic from the Tip-X paper.
    """
    kl_sim = torch.zeros((eval_dists.shape[0], support_dists.shape[0]), device=eval_dists.device)
    for i in tqdm(range(eval_dists.shape[0]), desc="Computing KL-Div Sim", leave=False):
        p = eval_dists[i].unsqueeze(0)
        q = support_dists
        kl_values = (p * (p.log() - q.log())).sum(dim=1)
        # Negate the KL divergence values to turn distance into similarity
        kl_sim[i] = -kl_values

    # Scale the KL similarities to match the range of cosine similarities (affinity_for_scaling)
    scaled_kl_affinity = _scale(kl_sim, affinity_for_scaling)
    
    return scaled_kl_affinity

def search_hp_tipx_biomedclip(cfg, cache_keys, cache_values, val_features, val_labels, text_weights):
    """
    Performs a 3D grid search with the new refactored KL-affinity function.
    """
    print("\n**** Searching for best hyperparameters (alpha, beta, gamma) for Tip-X ****")
    
    clip_logits = 85.2323 * val_features @ text_weights
    affinity = val_features @ cache_keys # Cosine similarity for cache_logits

    # +++ Use the new refactored function +++
    val_dists = _compute_distributions(val_features, text_weights)
    support_dists = _compute_distributions(cache_keys.t(), text_weights)
    scaled_kl_affinity = _compute_scaled_kl_affinity(val_dists, support_dists, affinity)
    kl_logits = scaled_kl_affinity @ cache_values.float()
    beta_search_range = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
    alpha_search_range = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
    gamma_search_range = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
    best_acc = 0.0
    best_beta, best_alpha, best_gamma = 0.0, 0.0, 0.0

    # Iterate over all hyperparameter combinations
    for beta in tqdm(beta_search_range, desc="Searching Beta"):
        # The cache_logits depend on beta, so they are calculated in the outer loop
        cache_logits = ((-1) * (beta - beta * affinity.float())).exp() @ cache_values.float()
        for alpha in alpha_search_range:
            for gamma in gamma_search_range:
                
                # Combine all three logit components with current hyperparameters
                tipx_logits = clip_logits + cache_logits * alpha + kl_logits * gamma
                acc = cls_acc(tipx_logits, val_labels)

                if acc > best_acc:
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha
                    best_gamma = gamma
    
    print(f"**** Best Val Acc: {best_acc:.2f} | Best Beta: {best_beta:.2f}, Best Alpha: {best_alpha:.2f}, Best Gamma: {best_gamma:.2f} ****")
    
    return best_beta, best_alpha, best_gamma


class TIPX_BiomedCLIP(FSCLIPmethod):
    '''
    TIP-X methods (Refactored for non-finetuning path)
    '''

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.cfg = args
        # Removed unused hyperparameters: self.lr, self.epoch, self.finetune
        self.shot = args['shots']
        self.init_beta = args['init_beta']
        self.init_alpha = args['init_alpha']
        # self.init_gamma is retrieved from cfg directly in the forward pass

    def forward(self,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                test_features: torch.tensor,
                test_labels: torch.tensor,
                text_weights: torch.tensor,
                model: nn.Module,
                classnames):
        """
        inputs:
            train_loader : torch.utils.data.DataLoader (for building cache)
            val_loader : torch.utils.data.DataLoader (for hyperparameter search)
            test_features : torch.Tensor of shape [test_data_size, feature_dim]
            test_labels : torch.Tensor of shape [test_data_size]
            text_weights : torch.Tensor of shape [num_classes, feature_dim]
            model : The CLIP model
            classnames : A list of class names
        """

        # Build cache model from the few-shot support set
        # cache_keys: Support set image embeddings [num_shot*num_classes, feature_dim]
        # cache_values: One-hot labels for the support set [num_shot*num_classes, num_classes]
        cache_keys, cache_values = build_cache_model(self.cfg, model, train_loader)

        # Initialize hyperparameters
        beta, alpha, gamma = self.cfg['init_beta'], self.cfg['init_alpha'], self.cfg['init_gamma']

        #--- Start Hyperparameter Search on Validation Set ---#
        print("\nExtracting visual features and labels from validation set for HP search.")
        val_features, val_labels = [], []
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(val_loader)):
                images, target = images.cuda(), target.cuda()
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                val_features.append(image_features)
                val_labels.append(target)
        val_features, val_labels = torch.cat(val_features), torch.cat(val_labels)

        # Search for the best hyperparameters using the validation set
        best_beta, best_alpha, best_gamma = search_hp_tipx_biomedclip(self.cfg, cache_keys, cache_values, val_features, val_labels, text_weights)

        #--- Start Evaluation on Test Set with Best Hyperparameters ---#
        print("\nEvaluating on the test set with optimal hyperparameters...")
        start_time = time.time()

        # 1. Zero-shot CLIP logits
        clip_logits = 85.2323 * test_features @ text_weights
        acc_clip = cls_acc(clip_logits, test_labels)
        print(f"**** Zero-shot BiomedCLIP's test accuracy: {acc_clip:.2f}% ****")

        # 2. Tip-Adapter logits (using the cache)
        affinity = test_features @ cache_keys.t()
        cache_logits = ((-1) * (best_beta - best_beta * affinity.float())).exp() @ cache_values.float()

        # 3. TIP-X (KL-Divergence) logits
        print("Computing KL-Divergence based logits for the test set...")
        test_dists = _compute_distributions(test_features, text_weights)
        support_dists = _compute_distributions(cache_keys, text_weights)
        kl_sim = _compute_scaled_kl_affinity(test_dists, support_dists, affinity)
        kl_logits = kl_sim @ cache_values.float()

        # 4. Ensemble all logits for final prediction
        tipx_logits = clip_logits + cache_logits * best_alpha + kl_logits * best_gamma
        final_acc = cls_acc(tipx_logits, test_labels)
        print(f"**** TIP-X final test accuracy: {final_acc:.2f}% ****")
        print(f"Inference time: {time.time() - start_time:.2f} seconds.")

        # In the original code, the second return value was `acc`, which was the
        # test accuracy. Here we return the final accuracy.
        return None, final_acc