import os
import numpy as np
import pandas as pd
import time
import tqdm
from scipy.special import erfinv
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from . import VAE_encoder, VAE_decoder

class VAE_model(nn.Module):
    """
    Class for the VAE model with estimation of weights distribution parameters via Mean-Field VI.
    """
    def __init__(self,
            model_name,
            data,
            encoder_parameters,
            decoder_parameters,
            random_seed
            ):

        super().__init__()

        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.random_seed = random_seed
        torch.manual_seed(random_seed)

        self.seq_len = data.seq_len
        self.embedding_dim = data.one_hot_encoding.shape[-1]
        self.alphabet_size = getattr(data, 'alphabet_size', self.embedding_dim)
        
        # Compute or load weights
        if hasattr(data, 'weights') and data.weights is not None:
            self.weights = data.weights
            print("✅ Loaded sequence weights from dataset")
        else:
            print("⚠️ No weights found in dataset, computing using cosine similarity of avg embeddings")
            avg_embeddings = np.mean(data.one_hot_encoding, axis=1)  # (N, D)
            sim = cosine_similarity(avg_embeddings)
            theta = 0.01
            weights = np.array([
                1.0 / np.sum(sim[i] > theta) if np.sum(sim[i] > theta) > 0 else 0.0
                for i in range(sim.shape[0])
            ])
            self.weights = weights
            data.weights = weights
            print("✅ Computed and assigned new weights")

            # 👉 Save to file if `data.weights_path` is defined
            if hasattr(data, 'weights_path'):
                os.makedirs(os.path.dirname(data.weights_path), exist_ok=True)
                np.save(data.weights_path, self.weights)
                print(f"💾 Saved computed weights to {data.weights_path}")

        self.Neff = float(np.sum(self.weights))

        self.encoder_parameters = encoder_parameters
        self.decoder_parameters = decoder_parameters

        encoder_parameters['seq_len'] = self.seq_len
        encoder_parameters['alphabet_size'] = self.alphabet_size
        decoder_parameters['seq_len'] = self.seq_len
        decoder_parameters['alphabet_size'] = self.alphabet_size

        self.encoder = VAE_encoder.VAE_MLP_encoder(params=encoder_parameters)
        if decoder_parameters['bayesian_decoder']:
            self.decoder = VAE_decoder.VAE_Bayesian_MLP_decoder(params=decoder_parameters)
        else:
            self.decoder = VAE_decoder.VAE_Standard_MLP_decoder(params=decoder_parameters)
        self.logit_sparsity_p = decoder_parameters['logit_sparsity_p']



    def sample_latent(self, mu, log_var):
        """
        Samples a latent vector via reparametrization trick
        """
        eps = torch.randn_like(mu).to(self.device)
        z = torch.exp(0.5*log_var) * eps + mu
        return z

    def KLD_diag_gaussians(self, mu, logvar, p_mu, p_logvar):
        """
        KL divergence between diagonal gaussian with prior diagonal gaussian.
        """
        KLD = 0.5 * (p_logvar - logvar) + 0.5 * (torch.exp(logvar) + torch.pow(mu-p_mu,2)) / (torch.exp(p_logvar)+1e-20) - 0.5

        return torch.sum(KLD)

    def annealing_factor(self, annealing_warm_up, training_step):
        """
        Annealing schedule of KL to focus on reconstruction error in early stages of training
        """
        if training_step < annealing_warm_up:
            return training_step/annealing_warm_up
        else:
            return 1

    def KLD_global_parameters(self):
        """
        KL divergence between the variational distributions and the priors (for the decoder weights).
        """
        KLD_decoder_params = 0.0
        zero_tensor = torch.tensor(0.0).to(self.device) 
        
        for layer_index in range(len(self.decoder.hidden_layers_sizes)):
            for param_type in ['weight','bias']:
                KLD_decoder_params += self.KLD_diag_gaussians(
                                    self.decoder.state_dict(keep_vars=True)['hidden_layers_mean.'+str(layer_index)+'.'+param_type].flatten(),
                                    self.decoder.state_dict(keep_vars=True)['hidden_layers_log_var.'+str(layer_index)+'.'+param_type].flatten(),
                                    zero_tensor,
                                    zero_tensor
                )
                
        for param_type in ['weight','bias']:
                KLD_decoder_params += self.KLD_diag_gaussians(
                                        self.decoder.state_dict(keep_vars=True)['last_hidden_layer_'+param_type+'_mean'].flatten(),
                                        self.decoder.state_dict(keep_vars=True)['last_hidden_layer_'+param_type+'_log_var'].flatten(),
                                        zero_tensor,
                                        zero_tensor
                )

        if self.decoder.include_sparsity:
            self.logit_scale_sigma = 4.0
            self.logit_scale_mu = 2.0**0.5 * self.logit_scale_sigma * erfinv(2.0 * self.logit_sparsity_p - 1.0)

            sparsity_mu = torch.tensor(self.logit_scale_mu).to(self.device) 
            sparsity_log_var = torch.log(torch.tensor(self.logit_scale_sigma**2)).to(self.device)
            KLD_decoder_params += self.KLD_diag_gaussians(
                                    self.decoder.state_dict(keep_vars=True)['sparsity_weight_mean'].flatten(),
                                    self.decoder.state_dict(keep_vars=True)['sparsity_weight_log_var'].flatten(),
                                    sparsity_mu,
                                    sparsity_log_var
            )
            
        if self.decoder.convolve_output:
            for param_type in ['weight']:
                KLD_decoder_params += self.KLD_diag_gaussians(
                                    self.decoder.state_dict(keep_vars=True)['output_convolution_mean.'+param_type].flatten(),
                                    self.decoder.state_dict(keep_vars=True)['output_convolution_log_var.'+param_type].flatten(),
                                    zero_tensor,
                                    zero_tensor
                )

        if self.decoder.include_temperature_scaler:
            KLD_decoder_params += self.KLD_diag_gaussians(
                                    self.decoder.state_dict(keep_vars=True)['temperature_scaler_mean'].flatten(),
                                    self.decoder.state_dict(keep_vars=True)['temperature_scaler_log_var'].flatten(),
                                    zero_tensor,
                                    zero_tensor
            )        
        return KLD_decoder_params

    def loss_function(self, x_recon_log, x, mu, log_var,
                      kl_latent_scale, kl_global_params_scale,
                      annealing_warm_up, training_step, Neff):
        """
        Returns mean of negative ELBO, reconstruction loss and KL divergence across batch x.
        Adds NaN protection and clamping.
        """
        # Clamp logits to prevent numerical instability
        x_recon_log = torch.clamp(x_recon_log, min=-10, max=10)

        # Clamp log_var to a stable range
        log_var = torch.clamp(log_var, min=-10, max=10)

        BCE = F.binary_cross_entropy_with_logits(x_recon_log, x, reduction='sum') / x.shape[0]
        KLD_latent = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())) / x.shape[0]

        if self.decoder.bayesian_decoder:
            KLD_decoder_params_normalized = self.KLD_global_parameters() / Neff
        else:
            KLD_decoder_params_normalized = 0.0

        warm_up_scale = self.annealing_factor(annealing_warm_up, training_step)
        neg_ELBO = BCE + warm_up_scale * (kl_latent_scale * KLD_latent + kl_global_params_scale * KLD_decoder_params_normalized)

        # Check for NaNs
        if torch.isnan(neg_ELBO) or torch.isnan(BCE) or torch.isnan(KLD_latent):
            print("❌ NaN detected in loss components!")
            print("BCE:", BCE)
            print("KLD_latent:", KLD_latent)
            print("KLD_global_params:", KLD_decoder_params_normalized)
            print("log_var stats:", log_var.min().item(), log_var.max().item())
            print("mu stats:", mu.min().item(), mu.max().item())
            raise ValueError("NaNs detected in VAE loss. Try reducing learning rate or inspecting inputs.")

        return neg_ELBO, BCE, KLD_latent, KLD_decoder_params_normalized


    def all_likelihood_components(self, x):
        """
        Returns tensors of ELBO, reconstruction loss and KL divergence for each point in batch x.
        """
        mu, log_var = self.encoder(x)
        z = self.sample_latent(mu, log_var)
        recon_x_log = self.decoder(z)

        recon_x_log = recon_x_log.view(-1,self.alphabet_size*self.seq_len)
        x = x.view(-1,self.alphabet_size*self.seq_len)
        
        BCE_batch_tensor = torch.sum(F.binary_cross_entropy_with_logits(recon_x_log, x, reduction='none'),dim=1)
        KLD_batch_tensor = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),dim=1))
        
        ELBO_batch_tensor = -(BCE_batch_tensor + KLD_batch_tensor)

        return ELBO_batch_tensor, BCE_batch_tensor, KLD_batch_tensor

    def train_model(self, data, training_parameters):
        """
        Trains the VAE model using the provided data and training parameters.
        """
        if torch.cuda.is_available():
            cudnn.benchmark = True
        self.train()

        # Logging
        if training_parameters['log_training_info']:
            log_path = os.path.join(training_parameters['training_logs_location'], f"{self.model_name}_losses.csv")
            with open(log_path, "w") as logs:
                logs.write("Number of sequences in alignment file:\t" + str(data.num_sequences) + "\n")
                logs.write("Neff:\t" + str(self.Neff) + "\n")
                logs.write("Alignment sequence length:\t" + str(data.seq_len) + "\n")

        # Optimizer
        optimizer = optim.Adam(
            self.parameters(),
            lr=training_parameters['learning_rate'],
            weight_decay=training_parameters['l2_regularization']
        )
        scheduler = None
        if training_parameters['use_lr_scheduler']:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=training_parameters['lr_scheduler_step_size'],
                gamma=training_parameters['lr_scheduler_gamma']
            )

        # Split data (if needed)
        if training_parameters['use_validation_set']:
            x_train, x_val, weights_train, weights_val = train_test_split(
                data.one_hot_encoding, data.weights,
                test_size=training_parameters['validation_set_pct'],
                random_state=self.random_seed
            )
            best_val_loss = float('inf')
            best_model_step_index = 0
        else:
            x_train = data.one_hot_encoding
            weights_train = data.weights
            best_val_loss = None
            best_model_step_index = training_parameters['num_training_steps']

        # Sanitize weights
        weights_train = np.nan_to_num(weights_train, nan=0.0, posinf=0.0, neginf=0.0)
        if np.sum(weights_train) == 0:
            weights_train = np.ones_like(weights_train)
        seq_sample_probs = weights_train / np.sum(weights_train)

        # Save weights (optional)
        if "weights_save_path" in training_parameters:
            np.save(training_parameters["weights_save_path"], weights_train)

        self.Neff_training = np.sum(weights_train)
        N_training = x_train.shape[0]
        batch_order = np.arange(N_training)

        # Training loop
        start = time.time()
        for training_step in tqdm.tqdm(range(1, training_parameters['num_training_steps'] + 1), desc="Training model"):
            batch_size = training_parameters['batch_size']
            if batch_size > N_training:
                raise ValueError(f"Batch size {batch_size} exceeds number of training samples {N_training}.")

            batch_index = np.random.choice(batch_order, batch_size, p=seq_sample_probs).tolist()
            x = torch.tensor(x_train[batch_index], dtype=self.dtype).to(self.device)
            optimizer.zero_grad()

            mu, log_var = self.encoder(x)
            z = self.sample_latent(mu, log_var)
            recon_x_log = self.decoder(z)

            neg_ELBO, BCE, KLD_latent, KLD_decoder_params_norm = self.loss_function(
                recon_x_log, x, mu, log_var,
                training_parameters['kl_latent_scale'],
                training_parameters['kl_global_params_scale'],
                training_parameters['annealing_warm_up'],
                training_step,
                self.Neff_training
            )

            neg_ELBO.backward()
            for name, param in self.named_parameters():
                if torch.isnan(param.grad).any():
                    print(f"NaNs in gradient of {name}")

            optimizer.step()
            if scheduler:
                scheduler.step()

            # Logging progress
            if training_step % training_parameters['log_training_freq'] == 0:
                log_msg = (f"|Train : Update {training_step}. Negative ELBO : {neg_ELBO:.3f}, "
                           f"BCE: {BCE:.3f}, KLD_latent: {KLD_latent:.3f}, "
                           f"KLD_decoder_params_norm: {KLD_decoder_params_norm:.3f}, "
                           f"Time: {time.time() - start:.2f} |")
                print(log_msg)
                if training_parameters['log_training_info']:
                    with open(log_path, "a") as logs:
                        logs.write(log_msg + "\n")

            # Save periodic checkpoint
            if training_step % training_parameters['save_model_params_freq'] == 0:
                self.save(
                    model_checkpoint=os.path.join(training_parameters['model_checkpoint_location'],
                                                  f"{self.model_name}_step_{training_step}"),
                    encoder_parameters=self.encoder_parameters,
                    decoder_parameters=self.decoder_parameters,
                    training_parameters=training_parameters
                )

            # Validation
            if training_parameters['use_validation_set'] and training_step % training_parameters['validation_freq'] == 0:
                self.eval()
                with torch.no_grad():
                    val_batch_index = np.random.choice(
                        np.arange(x_val.shape[0]),
                        batch_size,
                        p=weights_val / np.sum(weights_val)
                    ).tolist()
                    x_val_batch = torch.tensor(x_val[val_batch_index], dtype=self.dtype).to(self.device)
                    val_neg_ELBO, val_BCE, val_KLD_latent, val_KLD_global = self.test_model(
                        x_val_batch,
                        weights_val,
                        batch_size
                    )
                    val_msg = (f"\t\t\t|Val : Update {training_step}. Negative ELBO : {val_neg_ELBO:.3f}, "
                               f"BCE: {val_BCE:.3f}, KLD_latent: {val_KLD_latent:.3f}, "
                               f"KLD_decoder_params_norm: {val_KLD_global:.3f}, Time: {time.time() - start:.2f} |")
                    print(val_msg)
                    if training_parameters['log_training_info']:
                        with open(log_path, "a") as logs:
                            logs.write(val_msg + "\n")

                    if val_neg_ELBO < best_val_loss:
                        best_val_loss = val_neg_ELBO
                        best_model_step_index = training_step
                        self.save(
                            model_checkpoint=os.path.join(training_parameters['model_checkpoint_location'],
                                                          f"{self.model_name}_best"),
                            encoder_parameters=self.encoder_parameters,
                            decoder_parameters=self.decoder_parameters,
                            training_parameters=training_parameters
                        )
                self.train()

    def test_model(self, x_val, weights_val, batch_size):
        self.eval()
        
        with torch.no_grad():
            val_batch_order = np.arange(x_val.shape[0])
            val_seq_sample_probs = weights_val / np.sum(weights_val)

            val_batch_index = np.random.choice(val_batch_order, batch_size, p=val_seq_sample_probs).tolist()
            x = torch.tensor(x_val[val_batch_index], dtype=self.dtype).to(self.device)
            mu, log_var = self.encoder(x)
            z = self.sample_latent(mu, log_var)
            recon_x_log = self.decoder(z)
            
            neg_ELBO, BCE, KLD_latent, KLD_global_parameters = self.loss_function(recon_x_log, x, mu, log_var, kl_latent_scale=1.0, kl_global_params_scale=1.0, annealing_warm_up=0, training_step=1, Neff = self.Neff_training) #set annealing factor to 1
            
        return neg_ELBO.item(), BCE.item(), KLD_latent.item(), KLD_global_parameters.item()
        

    def save(self, model_checkpoint, encoder_parameters, decoder_parameters, training_parameters, batch_size=8):
        torch.save({
            'model_state_dict':self.state_dict(),
            'encoder_parameters':encoder_parameters,
            'decoder_parameters':decoder_parameters,
            'training_parameters':training_parameters,
            }, model_checkpoint)

    def compute_evol_indices(self, msa_data, list_mutations_location, num_samples=20000, batch_size=8):
        """
        Compute evolutionary indices using ESM2 embeddings and reference sequence.

        Args:
            msa_data (ESM2EmbeddingDataset): dataset containing embeddings and weights
            list_mutations_location (str): CSV file with mutations in format 'wtAAposmutAA'
            num_samples (int): number of samples for ELBO approximation
            batch_size (int): batch size for VAE sampling

        Returns:
            list_valid_mutations (List[str])
            evol_indices (List[float])
            elbo_mutant (List[float])
            elbo_wt (float)
        """
        import pandas as pd
        from tqdm import tqdm

        reference_sequence = list(msa_data.reference_sequence)
        seq_len = msa_data.seq_len
        embedding_dim = msa_data.embedding_dim
        aas = "ACDEFGHIKLMNPQRSTVWY"

        # Load mutations
        mutation_df = pd.read_csv(list_mutations_location)
        mutation_list = mutation_df["mutations"].tolist()

        # Validate and parse mutations
        valid_mutations = []
        mutation_positions = []

        for mut in mutation_list:
            if len(mut) < 3:
                continue
            wt, mut_aa = mut[0], mut[-1]
            try:
                pos = int(mut[1:-1]) - 1
            except ValueError:
                continue

            if pos >= seq_len or wt != reference_sequence[pos] or mut_aa not in aas:
                continue

            valid_mutations.append(mut)
            mutation_positions.append((pos, mut_aa))

        if len(valid_mutations) == 0:
            raise ValueError("No valid mutations found. Check your reference sequence and input format.")

        print(f"✅ Found {len(valid_mutations)} valid mutations")

        # Compute ELBO of WT
        wt_input = torch.tensor(msa_data.one_hot_encoding[0:1], dtype=self.dtype).to(self.device)

        elbo_wt_all = []
        self.eval()
        with torch.no_grad():
            for _ in range(num_samples // batch_size):
                mu, log_var = self.encoder(wt_input)
                z = self.sample_latent(mu, log_var)
                recon_x_log = self.decoder(z)
                neg_elbo, _, _, _ = self.loss_function(recon_x_log, wt_input, mu, log_var,
                                                       kl_latent_scale=1.0, kl_global_params_scale=1.0,
                                                       annealing_warm_up=0, training_step=1, Neff=1.0)
                elbo_wt_all.append(-neg_elbo.item())

        elbo_wt = np.mean(elbo_wt_all)

        # Compute ELBO for each mutant
        elbo_mutant = []
        mutation_batches = [
            mutation_positions[i:i+batch_size]
            for i in range(0, len(mutation_positions), batch_size)
        ]

        for batch in tqdm(mutation_batches, desc="Computing ELBO for mutants"):
            batch_size_actual = len(batch)
            mutant_input = np.repeat(msa_data.one_hot_encoding[0:1], batch_size_actual, axis=0)
            mutant_input = torch.tensor(mutant_input, dtype=self.dtype)

            # Apply mutation to relevant positions
            for i, (pos, mut_aa) in enumerate(batch):
                one_hot_vector = np.zeros(msa_data.embedding_dim)
                aa_index = aas.index(mut_aa)
                one_hot_vector[aa_index] = 1.0
                mutant_input[i, pos] = torch.tensor(one_hot_vector, dtype=self.dtype)

            mutant_input = mutant_input.to(self.device)

            elbos_mut = []
            with torch.no_grad():
                for _ in range(num_samples // batch_size):
                    mu, log_var = self.encoder(mutant_input)
                    z = self.sample_latent(mu, log_var)
                    recon_x_log = self.decoder(z)
                    neg_elbo, _, _, _ = self.loss_function(recon_x_log, mutant_input, mu, log_var,
                                                           kl_latent_scale=1.0, kl_global_params_scale=1.0,
                                                           annealing_warm_up=0, training_step=1, Neff=1.0)
                    elbos_mut.append(-neg_elbo.cpu().numpy())  # shape (B,)
            elbo_mutant.append(elbos_mut)

        # Compute evolutionary index = ELBO(mutant) - ELBO(wt)
        evol_indices = [mut_elbo - elbo_wt for mut_elbo in elbo_mutant]
        print(f"# mutations: {len(valid_mutations)}")
        print(f"# elbo_mutant: {len(elbo_mutant)}")
        print(f"# elbo_wt: {elbo_wt}")


        return valid_mutations, evol_indices, elbo_mutant, elbo_wt
