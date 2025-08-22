import os
import json
import argparse
import torch

from EVE import VAE_model
from utils.data_utils import ESM2EmbeddingDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train VAE using ESM2 embeddings")
    parser.add_argument('--embedding_file', type=str, required=True, help='Path to .npy file of residue embeddings')
    parser.add_argument('--ids_file', type=str, required=True, help='Path to .txt or .json file with sequence IDs')
    parser.add_argument('--protein_name', type=str, required=True, help='Protein name identifier')
    parser.add_argument('--VAE_checkpoint_location', type=str, required=True, help='Folder to store model checkpoints')
    parser.add_argument('--weights_file', type=str, required=True, help='Path to save/load sequence weights')
    parser.add_argument('--model_name_suffix', type=str, default='esm', help='Suffix for model checkpoint file')
    parser.add_argument('--model_parameters_location', type=str, required=True, help='Path to JSON model config')
    parser.add_argument('--training_logs_location', type=str, required=True, help='Where to save training logs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set reference sequence explicitly
    reference_sequence = "MLEKCLTAGYCSQLLFFWCIVPFCFAALVNAASNSSSHLQLIYNLTICELNGTDWLNQKFDWAVETFVIFPVLTHIVSYGALTTSHFLDTAGLITVSTAGYYHGRYVLSSIYAVFALAALICFVIRLTKNCMSWRYSCTRYTNFLLDTKGNLYRWRSPVVIERRGKVEVGDHLIDLKRVVLDGSAATPITKISAEQWGRP"

    # Load ESM2 embeddings and weights
    data = ESM2EmbeddingDataset(
        embedding_file=args.embedding_file,
        ids_file=args.ids_file,
        weights_file=args.weights_file,
        save_weights_to=args.weights_file,
        theta=0.01,
        reference_sequence=reference_sequence
    )

    # Load model parameters
    with open(args.model_parameters_location) as f:
        model_params = json.load(f)

    model_name = f"{args.protein_name}_{args.model_name_suffix}"
    model = VAE_model.VAE_model(
        model_name=model_name,
        data=data,
        encoder_parameters=model_params['encoder_parameters'],
        decoder_parameters=model_params['decoder_parameters'],
        random_seed=args.seed
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("Started Training")
    # Train model
    model.train_model(
        data=data,
        training_parameters=model_params["training_parameters"],
        model_dir=args.VAE_checkpoint_location,
        logs_dir=args.training_logs_location,
        save_model=True
    )
