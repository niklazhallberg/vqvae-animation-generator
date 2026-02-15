# --- START OF FILE tests/test_basic.py ---

"""Grundläggande tester för VQ-VAE-modellen."""

import torch
import pytest

from models.vqvae_model import VQVAE, Encoder, Decoder, VectorQuantizer


# --- Testkonstanter ---
BATCH_SIZE = 2
IMG_SIZE = 64
EMBEDDING_DIM = 64
NUM_EMBEDDINGS = 128
LATENT_SIZE = IMG_SIZE // 8  # 8 = encoder downsampling factor


@pytest.fixture
def model() -> VQVAE:
    """Skapa en VQVAE-modell för testning."""
    m = VQVAE(embedding_dim=EMBEDDING_DIM, num_embeddings=NUM_EMBEDDINGS)
    m.eval()
    return m


@pytest.fixture
def sample_input() -> torch.Tensor:
    """Skapa en slumpmässig indatabatch."""
    return torch.randn(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)


class TestVQVAEShapes:
    """Testa att alla tensor-shapes genom modellen stämmer."""

    def test_forward_output_shapes(
        self, model: VQVAE, sample_input: torch.Tensor
    ) -> None:
        """Forward pass ska returnera 5 tensorer med korrekta shapes."""
        with torch.no_grad():
            z_e, z_q_ste, recon, codebook_loss, commitment_loss = model(sample_input)

        assert z_e.shape == (BATCH_SIZE, EMBEDDING_DIM, LATENT_SIZE, LATENT_SIZE)
        assert z_q_ste.shape == z_e.shape
        assert recon.shape == sample_input.shape
        assert codebook_loss.shape == ()
        assert commitment_loss.shape == ()

    def test_encoder_output_shape(self, sample_input: torch.Tensor) -> None:
        """Encoder ska ge [B, embedding_dim, H/8, W/8]."""
        encoder = Encoder(embedding_dim=EMBEDDING_DIM)
        with torch.no_grad():
            z_e = encoder(sample_input)

        assert z_e.shape == (BATCH_SIZE, EMBEDDING_DIM, LATENT_SIZE, LATENT_SIZE)

    def test_decoder_output_shape(self) -> None:
        """Decoder ska ge [B, 1, H, W] från latent input."""
        decoder = Decoder(embedding_dim=EMBEDDING_DIM)
        latent_input = torch.randn(BATCH_SIZE, EMBEDDING_DIM, LATENT_SIZE, LATENT_SIZE)
        with torch.no_grad():
            recon = decoder(latent_input)

        assert recon.shape == (BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)


class TestVQVAERoundtrip:
    """Testa encode/decode-index roundtrip."""

    def test_encode_decode_indices_roundtrip(
        self, model: VQVAE, sample_input: torch.Tensor
    ) -> None:
        """encode_indices -> decode_indices ska ge korrekt output-shape."""
        with torch.no_grad():
            indices = model.encode_indices(sample_input)

        assert indices.shape == (BATCH_SIZE, LATENT_SIZE, LATENT_SIZE)
        assert indices.dtype == torch.long
        assert indices.min() >= 0
        assert indices.max() < NUM_EMBEDDINGS

        with torch.no_grad():
            recon = model.decode_indices(indices)

        assert recon.shape == sample_input.shape

    def test_reconstruction_range(
        self, model: VQVAE, sample_input: torch.Tensor
    ) -> None:
        """Rekonstruktioner ska ligga i [0, 1] (Sigmoid output)."""
        with torch.no_grad():
            _, _, recon, _, _ = model(sample_input)

        assert recon.min() >= 0.0
        assert recon.max() <= 1.0


# --- END OF FILE tests/test_basic.py ---
