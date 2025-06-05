import os
import random
import argparse
import yaml
import time
import pandas as pd
from tqdm import tqdm
import requests

import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer


import datasets.busi
import datasets.lungcolon
import datasets.chmnist
import datasets.covid
import datasets.btmri
import datasets.ctkidney
import datasets.kvasir
import datasets.retina
import datasets.kneexray
import datasets.dermamnist 
import datasets.octmnist

import trainers.Zeroshot.zeroshot
import trainers.CoOp.coop_clip
import trainers.CoOp.coop_biomedclip
import trainers.CoOp.coop_pubmedclip
import trainers.CoOp.coop_pmcclip
import trainers.CoCoOp.cocoop_clip
import trainers.CoCoOp.cocoop_biomedclip
import trainers.CoCoOp.cocoop_pubmedclip
import trainers.CoCoOp.cocoop_pmcclip
import trainers.KgCoOp.kgcoop_clip
import trainers.KgCoOp.kgcoop_biomedclip
import trainers.KgCoOp.kgcoop_pubmedclip
import trainers.KgCoOp.kgcoop_pmcclip
import trainers.ProGrad.prograd_clip
import trainers.ProGrad.prograd_biomedclip
import trainers.ProGrad.prograd_pubmedclip
import trainers.ProGrad.prograd_pmcclip
import trainers.BiomedCoOp.biomedcoop_clip
import trainers.BiomedCoOp.biomedcoop_biomedclip
import trainers.BiomedCoOp.biomedcoop_pubmedclip
import trainers.BiomedCoOp.biomedcoop_pmcclip

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from trainers import __dict__ as all_methods
from utils import *
from open_clip.src.open_clip import create_model_from_pretrained

from clip.pmcclip import ModifiedResNet, image_transform

from transformers import AutoTokenizer, AutoModel

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_config', default='configs/base.yaml',
        help='setting of Few-shot CLIP')
    parser.add_argument(
        '--dataset_config', default='configs/caltech101.yaml',
        help='dataset config')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = load_cfg_from_cfg_file(args.base_config)
    cfg.update(load_cfg_from_cfg_file(args.dataset_config))
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg

def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head



def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.COCOOP.CSC = False  # class-specific context
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.COCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.BIOMEDCOOP = CN()
    cfg.TRAINER.BIOMEDCOOP.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.BIOMEDCOOP.CSC = False  # class-specific context
    cfg.TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.BIOMEDCOOP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.BIOMEDCOOP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.BIOMEDCOOP.SCCM_LAMBDA = 1.0
    cfg.TRAINER.BIOMEDCOOP.KDSP_LAMBDA = 1.0
    cfg.TRAINER.BIOMEDCOOP.TAU = 1.5
    cfg.TRAINER.BIOMEDCOOP.N_PROMPTS = 50

    cfg.TRAINER.KGCOOP = CN()
    cfg.TRAINER.KGCOOP.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KGCOOP.CSC = False  # class-specific context
    cfg.TRAINER.KGCOOP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.KGCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.KGCOOP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.KGCOOP.W = 1.0

    cfg.TRAINER.PROGRAD = CN()
    cfg.TRAINER.PROGRAD.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.PROGRAD.CSC = False  # class-specific context
    cfg.TRAINER.PROGRAD.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.PROGRAD.N_CTX = 4  # number of context vectors
    cfg.TRAINER.PROGRAD.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.PROGRAD.GM = False
    cfg.TRAINER.PROGRAD.NAME = ""
    cfg.TRAINER.PROGRAD.ALPHA = 0.
    cfg.TRAINER.PROGRAD.T = 1.
    cfg.TRAINER.PROGRAD.LAMBDA = 1.

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    # cfg.freeze()

    return cfg

def get_text_features_from_dassl_trainer(dassl_trainer) -> torch.Tensor:
    """
    Extracts "ready-to-use" (normalized) text features from a DASSL trainer
    that has a trained CustomCLIP-like model.

    Args:
        dassl_trainer: Your DASSL trainer object (e.g., instance of BiomedCoOp_BiomedCLIP)
                       after building it and loading model weights.

    Returns:
        torch.Tensor: Normalized text features, shape (number_of_classes, feature_dimension).
    """
    if not hasattr(dassl_trainer, 'model'):
        raise ValueError("The provided DASSL trainer does not have a 'model' attribute.")

    # 1. Access the underlying model from the trainer
    actual_model = dassl_trainer.model

    # 2. Handle DataParallel if the model was trained on multiple GPUs
    if isinstance(actual_model, nn.DataParallel):
        actual_model = actual_model.module
    
    # Basic check to see if the model seems to be our CustomCLIP
    # You might want more robust checks depending on your setup
    if not hasattr(actual_model, 'prompt_learner') or not hasattr(actual_model, 'text_encoder'):
        raise TypeError("The model within the DASSL trainer does not appear to be the expected CustomCLIP model "
                        "with 'prompt_learner' and 'text_encoder' attributes.")

    actual_model.eval()  # Set the model to evaluation mode

    prompt_learner = actual_model.prompt_learner
    text_encoder = actual_model.text_encoder

    # The `tokenized_prompts` from the PromptLearner are the original tokenizations
    # (e.g., for "X X X X classname.") that the BiomedCLIP's `encode_text`
    # method uses internally when `is_prompts_embeddings=True`.
    original_tokenized_prompts_for_encoder = prompt_learner.tokenized_prompts.to(
        next(text_encoder.parameters()).device  # Ensure it's on the same device as the model
    )

    with torch.no_grad():
        # 3. Get the full prompt embeddings from the PromptLearner.
        # Expected shape: (number_of_classes, sequence_length, embedding_dimension)
        prompt_embeddings = prompt_learner()

        # 4. Pass these embeddings through the TextEncoder.
        # Expected shape: (number_of_classes, feature_dimension)
        text_features = text_encoder(
            prompt_embeddings,
            True, # This flag indicates `prompt_embeddings` are indeed embeddings
            original_tokenized_prompts_for_encoder
        )

        # 5. Normalize the text features.
        text_features_normalized = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features_normalized

def main(args):

    # Load config file
    coop_cfg = setup_cfg(args)
    if coop_cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(coop_cfg.SEED))
        set_random_seed(coop_cfg.SEED)
    setup_logger(coop_cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and coop_cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # print_args(args, coop_cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    biomedcoop_trainer = build_trainer(coop_cfg)
    print("Trainer built successfully.")
    biomedcoop_trainer.load_model(args.model_dir, epoch=args.load_epoch)
    # text_features will be used as text_weight for tip
    text_features = get_text_features_from_dassl_trainer(biomedcoop_trainer)

    # ============== END of CoOp ===================== #
    tip_cfg = load_cfg_from_cfg_file(args.base_config)
    tip_cfg.update(load_cfg_from_cfg_file(args.dataset_config_file))
    if args.opts is not None:
        tip_cfg = merge_cfg_from_list(tip_cfg, args.opts)

    cache_dir = os.path.join('./caches', tip_cfg.DATASET.NAME)
    os.makedirs(cache_dir, exist_ok=True)
    tip_cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(tip_cfg, "\n")

    # method is instance of TipAdapter with args=tip_cfg
    method = all_methods[tip_cfg['method']](args=tip_cfg)

    clip_model_pretrained = tip_cfg['clip_model']

    if(clip_model_pretrained == 'CLIP'):
        clip_model, preprocess = clip.load(tip_cfg['backbone'])
        clip_model.eval()
    elif(clip_model_pretrained == 'BiomedCLIP'):

        # Load the model and config files from the Hugging Face Hub
        clip_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        clip_model = clip_model.cuda()
        clip_model.eval()

    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)

    tip_cfg.DATASET.ROOT = tip_cfg['root_path']
    tip_cfg.SEED = 1
    tip_cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    tip_cfg.DATASET.NUM_SHOTS = tip_cfg['shots']

    print("Preparing dataset.")
    dataset = build_dataset(tip_cfg)
    classnames = dataset.classnames
    test_loader = build_data_loader(data_source=dataset.test, batch_size=100, is_train=False, tfm=preprocess, shuffle=False)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    # Pre-load test features
    f_test_time = time.time()
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(
        tip_cfg, "test", clip_model, test_loader)

    total_acc = 0
    predictions = []
    # tasks is number of seed evaluated
    for i in range(tip_cfg['tasks']):
        random.seed(i+1)
        torch.manual_seed(i+1)
        print("Start Training Task:{}".format(str(i+1)))
        few_shot_train_data = dataset.generate_fewshot_dataset_(tip_cfg['shots'], split="train")
        few_shot_val_data = dataset.generate_fewshot_dataset_(tip_cfg['shots'], split="val") 

        if tip_cfg['finetune']:
            train_loader = build_data_loader(
                data_source=few_shot_train_data, batch_size=tip_cfg["batch_size"], tfm=train_tranform, is_train=True, shuffle=True)
        else:
            train_loader = build_data_loader(
                data_source=few_shot_train_data, batch_size=tip_cfg["batch_size"], tfm=train_tranform, is_train=True, shuffle=False)
        val_loader = build_data_loader(
            data_source=few_shot_val_data, batch_size=tip_cfg["batch_size"], tfm=preprocess, is_train=False, shuffle=False)

        loss, acc = method(train_loader=train_loader,
                        val_loader=val_loader,
                        test_features=test_features,
                        test_labels=test_labels,
                        text_weights=text_features,
                        model=clip_model,
                        classnames=classnames)
        print('Final Accuracy on task {}: {}'.format(str(i+1), acc))
        predictions.append(acc)
    tasks_acc, tasks_std = compute_confidence_interval(predictions)
    test_stats = {}
    test_stats['acc'] = tasks_acc
    test_stats['std'] = tasks_std

    print('Total Accuracy and std on {} tasks: {:.4f} , {:.4f}'.format(
        str(tip_cfg['tasks']), tasks_acc, tasks_std))
    if not os.path.exists(tip_cfg['output_dir']):
        os.makedirs(tip_cfg['output_dir'])
    csv_path = os.path.join(tip_cfg['output_dir'], tip_cfg.DATASET.NAME +".csv")


if __name__ == "__main__":
    # Tip tip_cfg
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_config', default='configs/base.yaml',
        help='setting of Few-shot CLIP TipAdapter')
    # parser.add_argument(
    #     '--dataset_config', default='configs/caltech101.yaml',
    #     help='dataset config')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    # args = parser.parse_args()
    # cfg = load_cfg_from_cfg_file(args.base_config)
    # cfg.update(load_cfg_from_cfg_file(args.dataset_config))
    # if args.opts is not None:
    #     cfg = merge_cfg_from_list(cfg, args.opts)

    # BioMedCoOp

    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)