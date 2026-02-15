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
from utils import plot_training_curve # Behåll plot-funktion

# ======================================================
# ⚙️ Argument Parser (Valfritt men användbart)
# ======================================================
parser = argparse.ArgumentParser(description="Generera bilder med tränad Spatial VQ-VAE")
parser.add_argument('--model_path', type=str, default="outputs/checkpoints/vqvae_best.pth",
                    help='Sökväg till den tränade modellfilen (.pth)')
parser.add_argument('--output_path', type=str, default="outputs/generated/spatial_vqvae_sample.png",
                    help='Sökväg där den genererade bilden ska sparas')
parser.add_argument('--num_samples', type=int, default=1,
                    help='Antal bilder att generera (från slumpmässiga koder)')
parser.add_argument('--latent_h', type=int, default=8, help='Höjd på den latenta gridden')
parser.add_argument('--latent_w', type=int, default=8, help='Bredd på den latenta gridden')
parser.add_argument('--log_path', type=str, default="outputs/logs/training_log.csv",
                    help='Sökväg till träningsloggfilen för att plotta kurvan')
args = parser.parse_args()


# ======================================================
# 🧠 LADDA MODELL (Försöker läsa parametrar från checkpoint)
# ======================================================
# Defaultvärden om de inte hittas i checkpoint
default_embedding_dim = 64
default_num_embeddings = 128

embedding_dim = default_embedding_dim
num_embeddings = default_num_embeddings

if not Path(args.model_path).exists():
    print(f"❌ Modellfilen hittades inte: {args.model_path}")
    sys.exit(1)

print(f"Laddar modell från: {args.model_path}")
try:
    checkpoint = torch.load(args.model_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        # Läs parametrar om de finns
        if 'embedding_dim' in checkpoint and checkpoint['embedding_dim'] != 'N/A':
            embedding_dim = checkpoint['embedding_dim']
        if 'num_embeddings' in checkpoint and checkpoint['num_embeddings'] != 'N/A':
            num_embeddings = checkpoint['num_embeddings']
        print(f"Använder parametrar: embedding_dim={embedding_dim}, num_embeddings={num_embeddings}")
        model_state_dict = checkpoint['model_state_dict']
    else:
        print("⚠️ Checkpoint innehåller endast state_dict. Använder default model params.")
        model_state_dict = checkpoint

except Exception as e:
    print(f"❌ Kunde inte ladda modellen från {args.model_path}. Fel: {e}")
    sys.exit(1)

# Initiera modellen
model = VQVAE(
    embedding_dim=embedding_dim,
    num_embeddings=num_embeddings
)
# Ladda vikterna
model.load_state_dict(model_state_dict)
model.eval()
device = torch.device(
    'mps' if torch.backends.mps.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
)
model.to(device)
print(f"✅ Modellen laddad till {device}.")

# ======================================================
# 🖼️ GENERERA BILD(ER) FRÅN SLUMPMÄSSIGA KODER
# ======================================================
print(f"Genererar {args.num_samples} exempelbild(er)...")

# Skapa en batch av slumpmässiga spatiala indices
# Shape: [num_samples, latent_h, latent_w]
random_indices = torch.randint(0, num_embeddings,
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

# ======================================================
# 📈 PLOTTAR LOSS-KURVA – Efter träning
# ======================================================
# Använd plot-funktionen från utils
plot_training_curve(log_path=args.log_path, save_path='outputs/logs/training_curve.png')

# --- END OF FILE generate.py ---
