# --- START OF FILE models/vqvae_model.py (med EMA, KMeans, Perplexity och Korrekt Indentering) ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

# =====================================================
# 🧱 ENCODER – (Från din "gamla" fungerande version)
# =====================================================
class Encoder(nn.Module):
    """Konvolutionell encoder som mappar bilder till latenta representationer.

    Fyra konvolutionslager med stride-2 downsampling (3x) ger
    en spatial reducering med faktor 8 (64x64 -> 8x8).
    """

    def __init__(self, embedding_dim: int = 64) -> None:
        """Initiera Encoder.

        Args:
            embedding_dim: Dimensionen på utdata-kanalerna (latent dim).
        """
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, embedding_dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Kör forward pass genom encodern.

        Args:
            x: Indatabild med shape [B, 1, H, W].

        Returns:
            Latent representation z_e med shape [B, embedding_dim, H/8, W/8].
        """
        z_e = self.conv_layers(x)
        return z_e

# =====================================================
# 🔁 VECTOR QUANTIZER – (Med EMA + Avancerad Återhämtning)
# =====================================================
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=128, embedding_dim=64, commitment_cost=0.25,
                 decay=0.99, epsilon=1e-5,
                 recovery_threshold=1e-6, recovery_probability=0.1,
                 recovery_window=100, recovery_noise_scale=0.01):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.recovery_threshold = recovery_threshold
        self.recovery_probability = recovery_probability
        self.recovery_window = recovery_window
        self.recovery_noise_scale = recovery_noise_scale

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_embedding_sum', self.embedding.weight.data.clone())
        # Buffers nedan används f.n. inte i den aktuella _recover_dead_codes, men kan vara användbara senare
        # self.register_buffer('usage_history', torch.zeros(num_embeddings, recovery_window))
        # self.register_buffer('usage_pos', torch.zeros(1, dtype=torch.long))
        # self.register_buffer('update_count', torch.zeros(1, dtype=torch.long))

        # Använd en buffer för perplexity så den flyttas till rätt device automatiskt
        self.register_buffer('perplexity', torch.tensor(0.0))

        print(f"INFO: VectorQuantizer initialiserad med EMA (decay={self.decay}, epsilon={self.epsilon})")
        print(f"INFO: Kodåterhämtning aktiv (threshold={self.recovery_threshold}, probability={self.recovery_probability})")

    def _tile(self, x):
        d, ew = x.shape
        if d < self.num_embeddings:
            n_repeats = (self.num_embeddings + d - 1) // d
            try:
                 std_val = float(ew.item() if isinstance(ew, torch.Tensor) else ew)
                 std = 0.01 / np.sqrt(std_val + 1e-9)
            except Exception:
                 std = 0.01 / torch.sqrt(torch.tensor(float(ew), dtype=torch.float32, device=x.device) + 1e-9)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def kmeans_init(self, data_loader, encoder, device, max_batches_for_kmeans=25):
        print("Starting K-Means initialization (for EMA)...")
        encoder.eval()
        latent_vectors = []
        actual_batches_processed = 0
        if not (data_loader and len(data_loader) > 0):
            print("Warning: data_loader is None or empty. Skipping K-Means.") ; encoder.train(); return
        num_batches_to_process = min(max_batches_for_kmeans, len(data_loader))
        if num_batches_to_process == 0: num_batches_to_process = 1
        print(f"Collecting encoder outputs from up to {num_batches_to_process} batches.")
        with torch.no_grad():
            for i, batch_data in enumerate(data_loader):
                if i >= num_batches_to_process: break
                actual_batches_processed += 1
                if isinstance(batch_data, (list, tuple)): inputs = batch_data[0].to(device)
                else: inputs = batch_data.to(device)
                z_e = encoder(inputs)
                z_e_permuted = z_e.permute(0, 2, 3, 1).contiguous()
                z_e_flat = z_e_permuted.view(-1, self.embedding_dim)
                latent_vectors.append(z_e_flat.cpu())
        if not latent_vectors:
            print(f"Warning: No latent vectors collected. Skipping K-Means.") ; encoder.train(); return
        latent_vectors_np = torch.cat(latent_vectors, dim=0).numpy()
        print(f"Running K-Means on {latent_vectors_np.shape[0]} vectors...")
        if latent_vectors_np.shape[0] < self.num_embeddings:
            print(f"Warning: Tiling samples for K-Means.")
            latent_vectors_torch = torch.from_numpy(latent_vectors_np).float()
            latent_vectors_tiled = self._tile(latent_vectors_torch)
            latent_vectors_np = latent_vectors_tiled.numpy()
        max_kmeans_samples = 50000
        if latent_vectors_np.shape[0] > max_kmeans_samples:
            print(f"Subsampling {max_kmeans_samples} for K-Means.")
            indices = np.random.choice(latent_vectors_np.shape[0], max_kmeans_samples, replace=False)
            latent_vectors_np = latent_vectors_np[indices]
        if latent_vectors_np.shape[0] == 0:
            print("Error: No samples for K-Means. Skipping.") ; encoder.train(); return

        # Normalisera INTE före KMeans
        kmeans_model = KMeans(n_clusters=self.num_embeddings, random_state=42, n_init=10)
        try:
            kmeans_model.fit(latent_vectors_np)
            cluster_centers = torch.tensor(kmeans_model.cluster_centers_, dtype=torch.float32, device=device)
            self.embedding.weight.data.copy_(cluster_centers)
            self._ema_embedding_sum.data.copy_(cluster_centers)
            self._ema_cluster_size.data.fill_(1.0) # Start med 1.0 för stabilitet
            print("K-Means initialization complete.")
        except Exception as e:
            print(f"Error during K-Means: {e}. Using random init.")
            self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
            self._ema_embedding_sum.data.copy_(self.embedding.weight.data)
            self._ema_cluster_size.data.fill_(1.0)
        encoder.train()

    def _recover_dead_codes(self, flat_input):
        """Identifiera och återställ inaktiva kodvektorer."""
        with torch.no_grad():
            current_usage = self._ema_cluster_size
            inactive_mask = (current_usage < self.recovery_threshold)
            n_inactive = inactive_mask.sum().item()

            if n_inactive == 0:
                return

            print(f"Attempting to recover {n_inactive} inactive codes...")
            inactive_indices = torch.nonzero(inactive_mask).squeeze(1)
            active_mask = ~inactive_mask
            n_active = active_mask.sum().item()

            if n_active == 0:
                 print("Warning: All codes seem inactive! Reinitializing randomly.")
                 random_indices = torch.randint(0, flat_input.size(0), (n_inactive,))
                 self.embedding.weight.data[inactive_indices] = flat_input[random_indices]
                 self._ema_cluster_size.data[inactive_indices] = 1.0
                 self._ema_embedding_sum.data[inactive_indices] = self.embedding.weight.data[inactive_indices].clone()
                 return

            # Blanda strategier
            use_batch_strategy = (torch.rand(n_inactive, device=flat_input.device) < 0.5)
            use_popular_strategy = ~use_batch_strategy

            # Batch Exemplar
            if use_batch_strategy.any():
                indices_to_reset = inactive_indices[use_batch_strategy]
                n_reset = len(indices_to_reset)
                random_input_indices = torch.randint(0, flat_input.size(0), (n_reset,), device=flat_input.device)
                new_vectors = flat_input[random_input_indices]
                noise = torch.randn_like(new_vectors) * self.recovery_noise_scale
                self.embedding.weight.data[indices_to_reset] = new_vectors + noise

            # Popular w/ Noise
            if use_popular_strategy.any():
                indices_to_reset = inactive_indices[use_popular_strategy]
                n_reset = len(indices_to_reset)
                active_indices = torch.where(active_mask)[0]
                active_cluster_sizes = self._ema_cluster_size[active_mask]
                top_k = min(n_reset, n_active)
                if n_active > 0 :
                   _, top_active_local_indices = torch.topk(active_cluster_sizes, k=min(top_k, n_active))
                   top_global_indices = active_indices[top_active_local_indices]
                   sampled_popular_indices = top_global_indices[torch.randint(0, len(top_global_indices), (n_reset,), device=flat_input.device)]
                   new_vectors = self.embedding.weight.data[sampled_popular_indices].clone()
                   noise = torch.randn_like(new_vectors) * self.recovery_noise_scale
                   self.embedding.weight.data[indices_to_reset] = new_vectors + noise
                #else: print("Warning: No active codes for 'Popular w/ Noise' recovery.") # Redundant

            # Återställ EMA för alla återställda koder
            mean_active_cluster_size = self._ema_cluster_size[active_mask].mean().item() if n_active > 0 else 1.0
            reset_cluster_size = max(mean_active_cluster_size * 0.1, self.epsilon)
            self._ema_cluster_size.data[inactive_indices] = reset_cluster_size
            self._ema_embedding_sum.data[inactive_indices] = self.embedding.weight.data[inactive_indices] * reset_cluster_size

    def forward(self, z_e):
        b, c, h, w = z_e.shape
        if c != self.embedding_dim:
             raise ValueError(f"Input channels {c} != embedding_dim {self.embedding_dim}")
        z_e_permuted = z_e.permute(0, 2, 3, 1).contiguous()
        flat_z_e = z_e_permuted.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_z_e**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_z_e, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1)
        encodings_one_hot = F.one_hot(encoding_indices, self.num_embeddings).float()
        z_q_flat = torch.matmul(encodings_one_hot, self.embedding.weight)

        if self.training:
            with torch.no_grad():
                # EMA Uppdatering
                self._ema_cluster_size = self._ema_cluster_size * self.decay + \
                                        (1 - self.decay) * torch.sum(encodings_one_hot, dim=0)
                self._ema_embedding_sum = self._ema_embedding_sum * self.decay + \
                                         (1 - self.decay) * torch.matmul(encodings_one_hot.t(), flat_z_e)
                n_total_usage = self._ema_cluster_size.sum()
                usage_normalized_for_update = (self._ema_cluster_size + self.epsilon) / \
                                       (n_total_usage + self.num_embeddings * self.epsilon) * n_total_usage
                world_space_centroids = self._ema_embedding_sum / usage_normalized_for_update.unsqueeze(1)
                self.embedding.weight.data.copy_(world_space_centroids)

                # Kör återhämtning med viss sannolikhet
                if torch.rand(1).item() < self.recovery_probability:
                    self._recover_dead_codes(flat_z_e)

        # Perplexity Beräkning
        with torch.no_grad():
             avg_probs = torch.mean(encodings_one_hot, dim=0)
             # Uppdatera perplexity-bufferten
             self.perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Commitment loss (utan beta)
        commitment_loss = F.mse_loss(flat_z_e, z_q_flat.detach())
        # Codebook loss (endast för loggning)
        codebook_loss_log = F.mse_loss(z_q_flat.detach(), flat_z_e)

        # STE
        z_q_ste_flat = flat_z_e + (z_q_flat - flat_z_e).detach()
        z_q_ste_reshaped = z_q_ste_flat.view(b, h, w, c)
        z_q_ste = z_q_ste_reshaped.permute(0, 3, 1, 2).contiguous()
        encoding_indices_reshaped = encoding_indices.view(b, h, w)

        return z_q_ste, encoding_indices_reshaped, codebook_loss_log, commitment_loss

# =====================================================
# 🧱 DECODER – (Från din "gamla" fungerande version)
# =====================================================
class Decoder(nn.Module):
    """Transponerad konvolutionell decoder som återskapar bilder från latenta representationer.

    Spegelvänd arkitektur mot Encoder med stride-2 upsampling (3x)
    för att gå från 8x8 tillbaka till 64x64.
    """

    def __init__(self, embedding_dim: int = 64) -> None:
        """Initiera Decoder.

        Args:
            embedding_dim: Dimensionen på indata-kanalerna (latent dim).
        """
        super().__init__()
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """Kör forward pass genom decodern.

        Args:
            z_q: Kvantiserad latent representation [B, embedding_dim, H, W].

        Returns:
            Rekonstruerad bild med shape [B, 1, H*8, W*8], värden i [0, 1].
        """
        x_recon = self.deconv_layers(z_q)
        return x_recon

# =====================================================
# 🧠 KOMPLETT VQ-VAE MODELL (med EMA och Återhämtning)
# =====================================================
class VQVAE(nn.Module):
    """Komplett VQ-VAE-modell med EMA-baserad vektorkvantisering.

    Kombinerar Encoder, VectorQuantizer (EMA + dead code recovery)
    och Decoder till en end-to-end-modell för bildrekonstruktion.

    Forward returnerar en 5-tuple:
        (z_e, z_q_ste, recon, codebook_loss_log, commitment_loss_term)
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        num_embeddings: int = 128,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
        ema_recovery_threshold: float = 1e-6,
        ema_recovery_probability: float = 0.1,
        ema_recovery_window: int = 100,
        ema_recovery_noise_scale: float = 0.01,
    ) -> None:
        """Initiera VQ-VAE med encoder, quantizer och decoder.

        Args:
            embedding_dim: Storlek på varje vektor i kodboken.
            num_embeddings: Antal kodvektorer i kodboken.
            commitment_cost: Vikten för commitment loss.
            ema_decay: EMA decay-faktor.
            ema_epsilon: Epsilon för numerisk stabilitet i EMA.
            ema_recovery_threshold: Tröskel för att identifiera döda koder.
            ema_recovery_probability: Sannolikhet att köra kodåterhämtning per forward.
            ema_recovery_window: Fönsterstorlek för användningshistorik.
            ema_recovery_noise_scale: Brusens storlek vid kodåterhämtning.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoder = Encoder(embedding_dim=embedding_dim)
        self.quantizer = VectorQuantizer(num_embeddings=num_embeddings,
                                         embedding_dim=embedding_dim,
                                         commitment_cost=commitment_cost,
                                         decay=ema_decay,
                                         epsilon=ema_epsilon,
                                         # Skicka vidare parametrarna
                                         recovery_threshold=ema_recovery_threshold,
                                         recovery_probability=ema_recovery_probability,
                                         recovery_window=ema_recovery_window,
                                         recovery_noise_scale=ema_recovery_noise_scale)
        self.decoder = Decoder(embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Kör forward pass genom hela VQ-VAE-pipelinen.

        Args:
            x: Indatabild med shape [B, 1, H, W].

        Returns:
            Tuple av (z_e, z_q_ste, recon, codebook_loss_log, commitment_loss_term):
                - z_e: Encoder-output [B, D, H', W']
                - z_q_ste: Kvantiserad med straight-through estimator [B, D, H', W']
                - recon: Rekonstruerad bild [B, 1, H, W]
                - codebook_loss_log: Codebook loss (för loggning)
                - commitment_loss_term: Commitment loss (utan beta)
        """
        z_e = self.encoder(x)
        quantized_ste, _, codebook_loss_log, commitment_loss_term = self.quantizer(z_e)
        recon = self.decoder(quantized_ste)
        # Returnera värden för träningsloopen
        return z_e, quantized_ste, recon, codebook_loss_log, commitment_loss_term

    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Encodera en bild till spatiala kodbok-index.

        Args:
            x: Indatabild med shape [B, 1, H, W].

        Returns:
            Spatiala index med shape [B, H', W'] där varje värde
            är ett index i kodboken.
        """
        z_e = self.encoder(x)
        _, indices, _, _ = self.quantizer(z_e)
        return indices

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Avkoda spatiala kodbok-index till bilder.

        Args:
            indices: Spatiala index med shape [B, H, W].

        Returns:
            Rekonstruerad bild med shape [B, 1, H*8, W*8].
        """
        assert len(indices.shape) == 3, "Input indices must be spatial [B, H, W]"
        b, h, w = indices.shape
        flat_indices = indices.view(-1)
        z_q_flat = self.quantizer.embedding(flat_indices)
        z_q_reshaped = z_q_flat.view(b, h, w, self.embedding_dim)
        z_q = z_q_reshaped.permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)

# --- END OF FILE models/vqvae_model.py ---
