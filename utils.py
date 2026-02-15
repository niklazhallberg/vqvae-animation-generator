# --- START OF FILE utils.py (KOMPLETT MED get_latest_checkpoint) ---

import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd


# =====================================================
# 💾 HITTA SENASTE CHECKPOINT
# =====================================================
def get_latest_checkpoint(
    checkpoint_dir: Path | str,
    pattern: str = "vqvae_latest_epoch_*.pth"
) -> Optional[Path]:
    """Hitta senaste checkpoint-filen i en mapp.

    Letar först efter 'vqvae_best.pth', annars senaste epoch-checkpointen.

    Args:
        checkpoint_dir: Sökväg till checkpoint-mappen.
        pattern: Globmönster för att hitta epoch-checkpoints.

    Returns:
        Path till senaste checkpoint, eller None om ingen hittas.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None

    best_ckpt_path = checkpoint_dir / "vqvae_best.pth"
    if best_ckpt_path.exists():
        return best_ckpt_path

    list_of_files = list(checkpoint_dir.glob(pattern))
    if not list_of_files:
        return None

    try:
        latest_file = max(list_of_files, key=lambda p: p.stat().st_mtime)
        return latest_file
    except Exception as e:
        print(f"Error finding latest checkpoint: {e}")
        return None


# =====================================================
# 💾 SPARA ORIGINAL + REKONSTRUERADE FRAMES
# =====================================================
def save_frame_examples(
    original_batch: torch.Tensor,
    recon_batch: torch.Tensor,
    epoch: int,
    save_dir: Path | str = Path('outputs/images'),
    prefix: str = 'recon'
) -> None:
    """Spara jämförelsebilder av original- och rekonstruerade frames.

    Skapar en matplotlib-figur med originalbilder på övre raden
    och rekonstruktioner på den nedre.

    Args:
        original_batch: Originaltensorer [N, C, H, W].
        recon_batch: Rekonstruerade tensorer [N, C, H, W].
        epoch: Aktuell epok (för filnamn och titel).
        save_dir: Mapp att spara bilden i.
        prefix: Filnamnsprefix.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    original_batch = original_batch.cpu().detach()
    recon_batch = recon_batch.cpu().detach()

    if original_batch.nelement() == 0 or recon_batch.nelement() == 0:
        return

    n_examples = min(len(original_batch), 8)
    if n_examples == 0:
        return

    if n_examples == 1:
        fig, axes = plt.subplots(2, 1, figsize=(2, 4))
        axes = np.array([axes[0], axes[1]]).reshape(2, 1)
    else:
        fig, axes = plt.subplots(2, n_examples, figsize=(n_examples * 2, 4))

    for i in range(n_examples):
        axes[0, i].imshow(original_batch[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon_batch[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')

    if n_examples > 0:
        axes[0, 0].set_title("Original", fontsize=10)
        axes[1, 0].set_title("Reconstruction", fontsize=10)

    fig.suptitle(f"Epoch {epoch}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = save_dir / f'{prefix}_epoch_{epoch:04d}.png'
    plt.savefig(path)
    plt.close(fig)


# =====================================================
# 📊 SPARA KODBOKSANVÄNDNING
# =====================================================
def save_codebook_usage_plot(
    encoding_indices_flat: torch.Tensor,
    codebook_size: int,
    epoch: int,
    save_dir: Path | str = Path('outputs/logs')
) -> None:
    """Spara stapeldiagram över kodboksanvändning.

    Visar hur ofta varje kodvektor i kodboken används,
    normaliserad om möjligt.

    Args:
        encoding_indices_flat: Platt tensor med kodindex.
        codebook_size: Antal kodvektorer i kodboken.
        epoch: Aktuell epok (för filnamn och titel).
        save_dir: Mapp att spara diagrammet i.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(encoding_indices_flat, torch.Tensor):
        return

    flat = encoding_indices_flat.cpu().view(-1)
    if flat.numel() == 0:
        return

    usage = torch.bincount(flat, minlength=codebook_size).float()
    usage_sum = usage.sum()

    if usage_sum > 0:
        plot_data = (usage / usage_sum).numpy()
        y_label = 'Användningsfrekvens (Normaliserad)'
    else:
        plot_data = usage.numpy()
        y_label = 'Antal Användningar'

    plt.figure(figsize=(10, 4))
    plt.bar(range(codebook_size), plot_data)
    plt.title(f'Kodbokanvändning (Epoch {epoch})')
    plt.xlabel('Kodvektor Index')
    plt.ylabel(y_label)
    plt.ylim(bottom=0)
    path = save_dir / f'codebook_usage_epoch_{epoch:04d}.png'
    plt.savefig(path)
    plt.close()


# =====================================================
# 📉 PLOTTAR LOSS-KURVA (med stabilitetskontroll)
# =====================================================
def plot_training_curve(
    log_path: Path | str = Path('outputs/logs/training_log.csv'),
    save_path: Path | str = Path('outputs/logs/training_curve.png')
) -> None:
    """Läser CSV-loggfil och ritar upp träningskurvor.

    Plottar tre subplots: total loss, loss-komponenter, och
    learning rate / perplexity. Kontrollerar även stabilitet
    i val_loss -- avbryter med sys.exit(1) om std > 0.02.

    Args:
        log_path: Sökväg till träningslogg-CSV.
        save_path: Sökväg att spara den genererade figuren.
    """
    log_path = Path(log_path)
    save_path = Path(save_path)

    if not log_path.exists():
        print(f"❌ Loggfil {log_path} hittades ej")
        return

    try:
        df = pd.read_csv(log_path)
        if df.empty or 'epoch' not in df.columns:
            print(f"❌ Loggfil {log_path} tom/saknar epoch")
            return

        # Stabilitetskontroll (från visualizations.py)
        if len(df) >= 5 and 'val_loss' in df.columns:
            recent_val_losses = df['val_loss'][-5:].values
            std_val = np.std(recent_val_losses)
            if std_val > 0.02:
                print(f"⚠️ Varning: Instabil träning! Val_loss varierar mycket (std = {std_val:.4f})")
                print("❌ Träningen avbryts p.g.a. instabil validerings-förlust.")
                sys.exit(1)

        print(f"Plotting training curve from: {log_path}")

        plt.figure(figsize=(12, 9))

        # Subplot 1: Total Loss
        plt.subplot(3, 1, 1)
        if 'train_loss' in df.columns:
            plt.plot(df['epoch'], df['train_loss'], label='Train Loss (Total)', c='tab:blue', lw=2)
        if 'val_loss' in df.columns:
            plt.plot(df['epoch'], df['val_loss'], label='Val Loss (Total)', c='tab:orange', ls='--', lw=2)
        plt.ylabel('Total Loss')
        plt.title('Träningskurva - Total Förlust')
        plt.legend(fontsize='small')
        plt.grid(True, ls=':', alpha=0.6)
        plt.yscale('log')

        # Subplot 2: Loss Components
        plt.subplot(3, 1, 2)
        if 'train_recon_loss' in df.columns:
            plt.plot(df['epoch'], df['train_recon_loss'], label='Train Recon', alpha=0.8, color='lightblue')
        if 'val_recon_loss' in df.columns:
            plt.plot(df['epoch'], df['val_recon_loss'], label='Val Recon', linestyle='--', alpha=0.8, color='moccasin')
        if 'train_commitment_loss' in df.columns:
            plt.plot(df['epoch'], df['train_commitment_loss'], label='Train Commit', alpha=0.6, color='lightgrey')
        if 'val_commitment_loss' in df.columns:
            plt.plot(df['epoch'], df['val_commitment_loss'], label='Val Commit', linestyle='--', alpha=0.6, color='darkgrey')
        plt.ylabel('Loss Components')
        plt.title('Förlust-komponenter')
        plt.legend(fontsize='small')
        plt.grid(True, ls=':', alpha=0.6)
        plt.yscale('log')

        # Subplot 3: Learning Rate & Perplexity
        plt.subplot(3, 1, 3)
        ax_lr = plt.gca()
        lines: list = []
        labels: list[str] = []

        if 'lr' in df.columns:
            line_lr, = ax_lr.plot(df['epoch'], df['lr'], label='Learning Rate', color='green')
            ax_lr.set_ylabel('Learning Rate', color='green')
            ax_lr.tick_params(axis='y', labelcolor='green')
            ax_lr.set_ylim(bottom=0)
            lines.append(line_lr)
            labels.append('Learning Rate')

        if 'perplexity' in df.columns:
            ax_ppl = ax_lr.twinx()
            line_ppl, = ax_ppl.plot(df['epoch'], df['perplexity'], label='Perplexity', color='red')
            ax_ppl.set_ylabel('Perplexity', color='red')
            ax_ppl.tick_params(axis='y', labelcolor='red')
            if not df['perplexity'].empty and df['perplexity'].max() > 1:
                ax_ppl.set_ylim(bottom=0.9)
            else:
                ax_ppl.set_ylim(bottom=0)
            lines.append(line_ppl)
            labels.append('Perplexity')

        plt.xlabel('Epoch')
        plt.title('Learning Rate & Perplexity')
        plt.grid(True, ls=':', alpha=0.6)
        if lines:
            ax_lr.legend(lines, labels, loc='best', fontsize='small')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"📈 Träningskurva sparad: {save_path}")

    except Exception as e:
        print(f"❌ Plot error: {e}")


# --- END OF FILE utils.py ---
