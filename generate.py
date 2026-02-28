# --- START OF FILE generate.py ---

# ======================================================
# 🌀 GENERATE – Skapa AI-genererade bilder (Looping kräver separat Prior-modell)
# ======================================================

import sys
from pathlib import Path

import torch
import numpy as np
import argparse # För att kunna ange modellfil från kommandoraden
from torchvision.utils import save_image

from models.vqvae_model import VQVAE # Importera Spatial VQ-VAE
from config import Config

# ======================================================
# ⚙️ Argument Parser (Valfritt men användbart)
# ======================================================
def parse_args() -> argparse.Namespace:
    """Parsa kommandoradsargument för generering."""
    parser = argparse.ArgumentParser(description="Generera bilder med tränad Spatial VQ-VAE")
    parser.add_argument('--model_path', type=str, default="outputs/checkpoints/vqvae_best.pth",
                        help='Sökväg till den tränade modellfilen (.pth)')
    parser.add_argument('--output_path', type=str, default="outputs/generated/spatial_vqvae_sample.png",
                        help='Sökväg där den genererade bilden ska sparas')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Antal bilder att generera (från slumpmässiga koder)')
    parser.add_argument('--latent_h', type=int, default=8, help='Höjd på den latenta gridden')
    parser.add_argument('--latent_w', type=int, default=8, help='Bredd på den latenta gridden')
    return parser.parse_args()


# ======================================================
# 🚀 HUVUDFUNKTION
# ======================================================
def main() -> None:
    """Ladda modell och generera bilder."""
    args = parse_args()

    # ======================================================
    # 🧠 LADDA MODELL
    # ======================================================
    if not Path(args.model_path).exists():
        print(f"❌ Modellfilen hittades inte: {args.model_path}")
        sys.exit(1)

    device = torch.device(
        'mps' if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'cpu'
    )

    print(f"Laddar modell från: {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            print(f"Checkpoint från epoch: {checkpoint.get('epoch', 'unknown')}")
        else:
            model_state_dict = checkpoint

    except Exception as e:
        print(f"❌ Kunde inte ladda modellen från {args.model_path}. Fel: {e}")
        sys.exit(1)

    # Initiera modellen (samma parametrar som latent_walk.py och start_training.py)
    model = VQVAE(
        embedding_dim=Config.EMBEDDING_DIM,
        num_embeddings=Config.NUM_EMBEDDINGS,
        commitment_cost=Config.BETA_START,
        ema_decay=Config.EMA_DECAY,
        ema_epsilon=Config.EMA_EPSILON,
        ema_recovery_threshold=Config.RECOVERY_THRESHOLD,
        ema_recovery_probability=Config.RECOVERY_PROB,
        ema_recovery_noise_scale=Config.RECOVERY_NOISE_SCALE
    ).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    print(f"✅ Modellen laddad till {device}.")

    # ======================================================
    # 🖼️ GENERERA BILD(ER) FRÅN SLUMPMÄSSIGA KODER
    # ======================================================
    print(f"Genererar {args.num_samples} exempelbild(er)...")

    # Skapa en batch av slumpmässiga spatiala indices
    # Shape: [num_samples, latent_h, latent_w]
    random_indices = torch.randint(0, Config.NUM_EMBEDDINGS,
                                   (args.num_samples, args.latent_h, args.latent_w),
                                   device=device)

    # Avkoda dessa indices till bilder
    with torch.no_grad():
        generated_images = model.decode_indices(random_indices)

    # Spara den/de genererade bilderna
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(generated_images, args.output_path, nrow=int(np.sqrt(args.num_samples))) # Spara som grid om flera
    print(f"🖼️ Exempelbild(er) sparad till: {args.output_path}")
    print("\nOBS: Detta är *enstaka bilder* genererade från slumpmässiga latenta koder.")
    print("För att generera en *sammanhängande animation* (och ev. loopande)")
    print("behövs ett andra steg: träna en Prior-modell (t.ex. Transformer)")
    print("på sekvenser av koder från VQ-VAE:n.")


if __name__ == '__main__':
    main()

# --- END OF FILE generate.py ---
