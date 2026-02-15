# --- START OF FILE train_vqvae.py (KORRIGERAD FÖR model.quantizer) ---

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

# Importera från dina andra moduler
# Viktigt: Detta antar att VQVAE-klassen finns i models/vqvae_model.py
# och att VectorQuantizer-klassen inuti den har ett attribut .perplexity
from models.vqvae_model import VQVAE
from utils import (
    save_frame_examples,
    save_codebook_usage_plot,
    get_latest_checkpoint # Antag att denna finns i utils.py
)


# ======================================================
# 🚀 TRÄNINGSFUNKTION FÖR VQ-VAE (med EMA)
# ======================================================
def train_vqvae(
    model: VQVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    beta_start: float = 0.05,
    beta_end: float = 0.25,
    beta_epochs: int = 30,
    patience: int = 10,
    min_delta: float = 1e-4,
    codebook_loss_weight: float = 0.0,
    checkpoint_dir: Path | str = Path('outputs/checkpoints'),
    log_dir: Path | str = Path('outputs/logs'),
    output_dir: Path | str = Path('outputs/images'),
    vis_epoch_interval: int = 5,
    log_interval: int = 50,
    force_kmeans_init: bool = False,
    scheduler_factor: float = 0.5,
    scheduler_min_lr: float = 1e-6,
    num_vis_examples: int = 8,
) -> None:
    """Träna VQ-VAE-modellen med EMA-kvantisering.

    Huvudsaklig träningsloop med stöd för:
    - Beta-warmup (commitment loss scheduling)
    - K-Means-initiering av kodboken
    - Checkpoint-sparande och -laddning
    - Early stopping baserat på val_loss
    - Visualiseringar av rekonstruktioner och kodboksanvändning
    - LR-scheduling via ReduceLROnPlateau

    Args:
        model: VQ-VAE-modell att träna.
        train_loader: DataLoader för träningsdata.
        val_loader: DataLoader för valideringsdata.
        epochs: Antal epoker att träna.
        learning_rate: Startlärfrekvens.
        beta_start: Startvärde för beta (commitment loss weight).
        beta_end: Slutvärde för beta efter warmup.
        beta_epochs: Antal epoker för beta-warmup.
        patience: Antal epoker utan förbättring innan early stopping.
        min_delta: Minimala förbättringen för att räknas som framsteg.
        codebook_loss_weight: Vikt för codebook loss (0.0 vid EMA).
        checkpoint_dir: Mapp för checkpoints.
        log_dir: Mapp för loggfiler.
        output_dir: Mapp för exempelbilder.
        vis_epoch_interval: Hur ofta exempelbilder sparas.
        log_interval: Hur ofta batch-statistik loggas.
        force_kmeans_init: Tvinga K-Means-initiering oavsett checkpoint.
        scheduler_factor: Faktor för LR-reducering vid platå.
        scheduler_min_lr: Minsta tillåtna lärfrekvens.
        num_vis_examples: Antal exempelbilder att visa vid visualisering.
    """
    # --- Initiering (Optimizer, Scheduler, Variabler) ---

    checkpoint_dir = Path(checkpoint_dir)
    log_dir = Path(log_dir)
    output_dir = Path(output_dir)

    device = next(model.parameters()).device
    print(f"Tränar på enhet: {device}")
    model.to(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Skapa parametergrupper: en för kodboken (som INTE optimeras), en för resten.
    encoder_decoder_params = [
        param for name, param in model.named_parameters()
        if 'quantizer.embedding' not in name # <<< Använder model.quantizer här
    ]

    # Optimera endast encoder/decoder
    optimizer = torch.optim.Adam(encoder_decoder_params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min',
        factor=scheduler_factor,
        patience=max(1, patience // 2 - 1),
        min_lr=scheduler_min_lr,
    )

    best_val_loss = float('inf')
    wait_counter = 0
    not_improved_epochs: list[int] = []

    log_file = log_dir / 'training_log.csv'
    start_epoch = 0

    # --- Checkpoint Loading ---
    loaded_checkpoint_path = None
    if not force_kmeans_init:
        loaded_checkpoint_path = get_latest_checkpoint(checkpoint_dir) if checkpoint_dir else None

    if loaded_checkpoint_path:
        print(f"Laddar checkpoint: {loaded_checkpoint_path}")
        try:
            checkpoint_data = torch.load(loaded_checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            try:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            except ValueError as e:
                print(f"VARNING: Kunde inte ladda optimizer state: {e}. Startar om optimizer.")
            if 'scheduler_state_dict' in checkpoint_data:
                try:
                    scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                    if hasattr(scheduler, 'best') and 'best_val_loss' in checkpoint_data:
                        scheduler.best = checkpoint_data['best_val_loss']
                except Exception as e:
                    print(f"VARNING: Kunde inte ladda scheduler state: {e}. Startar om scheduler.")
            start_epoch = checkpoint_data.get('epoch', 0) + 1
            best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))
            print(f"Återupptar träning från epok {start_epoch}. Bästa val_loss hittills: {best_val_loss:.6f}")
        except Exception as e:
            print(f"Fel vid laddning av checkpoint: {e}. Startar från scratch.")
            start_epoch = 0
            best_val_loss = float('inf')
            loaded_checkpoint_path = None

    # --- K-Means Initiering ---

    if not loaded_checkpoint_path or force_kmeans_init:
        # <<< Använd model.quantizer istället för model.get_quantizer() >>>
        if hasattr(model.quantizer, 'kmeans_init'):
            if force_kmeans_init:
                print("Tvingar K-Means initiering...")
            else:
                print("Ingen checkpoint hittad. Försöker med K-Means initiering...")

            if hasattr(train_loader, 'dataset') and len(train_loader.dataset) > 0:
                kmeans_batch_size = train_loader.batch_size * 2 if train_loader.batch_size else 64
                kmeans_device = next(model.encoder.parameters()).device
                kmeans_init_loader = DataLoader(
                    train_loader.dataset,
                    batch_size=kmeans_batch_size,
                    shuffle=False,
                    num_workers=train_loader.num_workers
                )
                # <<< Använd model.quantizer och model.encoder >>>
                model.quantizer.kmeans_init(kmeans_init_loader, model.encoder, kmeans_device)
                del kmeans_init_loader
            else:
                print("VARNING: Kunde inte skapa KMeans init loader från train_loader.")
        elif hasattr(model, 'quantizer'):
            print("INFO: Modellen quantizer saknar kmeans_init metod.")
        else:
            print("VARNING: Modellen saknar 'quantizer'-attribut.")


    # --- Huvudträningsloop ---
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()

        # --- Beta Scheduling ---
        current_beta = beta_start
        if beta_epochs > 0:
            if epoch < beta_epochs:
                current_beta = beta_start + (beta_end - beta_start) * (epoch / beta_epochs)
            else:
                current_beta = beta_end
        # Beta appliceras nedan

        # --- Träningssteg ---
        model.train()
        running_train_loss = 0.0
        running_train_recon_loss = 0.0
        running_train_commit_loss = 0.0
        running_train_perplexity = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs-1} [Train]", leave=False)
        for batch_idx, batch_data in enumerate(train_loop):
            if isinstance(batch_data, (list, tuple)):
                x = batch_data[0].to(device)
            else:
                x = batch_data.to(device)
            optimizer.zero_grad(set_to_none=True)

            try:
                # model.forward returnerar: z_e, z_q_ste, recon, codebook_loss_log, commitment_loss_term
                _, _, x_recon, _, commitment_loss_term = model(x)

                recon_loss_bce = F.binary_cross_entropy(x_recon, x, reduction='mean')
                # total_loss = recon_loss_bce + codebook_loss_weight * codebook_loss_log + current_beta * commitment_loss_term
                # Eftersom codebook_loss_weight är 0.0:
                total_loss = recon_loss_bce + current_beta * commitment_loss_term

                total_loss.backward()
                optimizer.step()

                # Loggning
                running_train_loss += total_loss.item()
                running_train_recon_loss += recon_loss_bce.item()
                running_train_commit_loss += (current_beta * commitment_loss_term.item())

                # <<< Hämta perplexity från model.quantizer.perplexity >>>
                current_perplexity = model.quantizer.perplexity.item() if hasattr(model.quantizer, 'perplexity') else 0.0
                running_train_perplexity += current_perplexity

                if batch_idx % log_interval == 0:
                    train_loop.set_postfix(loss=total_loss.item(), recon=recon_loss_bce.item(), perp=current_perplexity)

            except Exception as e:
                print(f"\nFel under träningsbatch {batch_idx}: {e}")
                continue

        avg_train_loss = running_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        avg_train_recon = running_train_recon_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        avg_train_commit = running_train_commit_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        avg_train_perplexity = running_train_perplexity / len(train_loader) if len(train_loader) > 0 else 0.0


        # --- Valideringssteg ---
        model.eval()
        running_val_loss = 0.0
        running_val_recon_loss = 0.0
        running_val_commit_loss = 0.0
        running_val_perplexity = 0.0
        all_val_indices_flat: list[torch.Tensor] = []

        val_loop = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs-1} [Val]", leave=False)
        with torch.no_grad():
            for batch_data in val_loop:
                if isinstance(batch_data, (list, tuple)):
                    x = batch_data[0].to(device)
                else:
                    x = batch_data.to(device)
                try:
                    _, _, x_recon, _, commitment_loss_term = model(x)
                    recon_loss_bce = F.binary_cross_entropy(x_recon, x, reduction='mean')
                    total_loss = recon_loss_bce + current_beta * commitment_loss_term

                    running_val_loss += total_loss.item()
                    running_val_recon_loss += recon_loss_bce.item()
                    running_val_commit_loss += (current_beta * commitment_loss_term.item())

                    # <<< Hämta perplexity från model.quantizer.perplexity >>>
                    current_perplexity = model.quantizer.perplexity.item() if hasattr(model.quantizer, 'perplexity') else 0.0
                    running_val_perplexity += current_perplexity
                    val_loop.set_postfix(loss=total_loss.item(), perp=current_perplexity)

                    if epoch % vis_epoch_interval == 0:
                        if hasattr(model, 'encode_indices'):
                            encoding_indices_batch = model.encode_indices(x)
                            all_val_indices_flat.append(encoding_indices_batch.view(-1).cpu())

                except Exception as e:
                    print(f"\nFel under valideringsbatch: {e}")
                    continue

        avg_val_loss = running_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        avg_val_recon = running_val_recon_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        avg_val_commit = running_val_commit_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        avg_val_perplexity = running_val_perplexity / len(val_loader) if len(val_loader) > 0 else 0.0

        # --- Epoch-utskrift ---
        current_lr = optimizer.param_groups[0]['lr']
        epoch_duration = time.time() - epoch_start_time
        print(f"📉 Epoch {epoch} färdig ({epoch_duration:.1f}s). "
              f"Train Loss: {avg_train_loss:.4f} (R: {avg_train_recon:.4f}) | "
              f"Val Loss: {avg_val_loss:.4f} (R: {avg_val_recon:.4f}) | "
              f"Perplexity(V): {avg_val_perplexity:.2f} | lr: {current_lr:.6f} | Beta: {current_beta:.4f}")

        # --- Logga till CSV ---
        try:
            log_header = "epoch,train_loss,val_loss,lr,perplexity,train_recon_loss,val_recon_loss,train_commitment_loss,val_commitment_loss\n"
            write_header = not log_file.exists() or log_file.stat().st_size == 0
            with open(log_file, "a") as f:
                if write_header:
                    f.write(log_header)
                f.write(f"{epoch},{avg_train_loss:.6f},{avg_val_loss:.6f},{current_lr:.6f},{avg_val_perplexity:.4f},"
                        f"{avg_train_recon:.6f},{avg_val_recon:.6f},"
                        f"{avg_train_commit:.6f},{avg_val_commit:.6f}\n")
        except IOError as e:
            print(f"Fel vid skrivning till loggfil {log_file}: {e}")

        # --- Spara Exempelbilder (vid intervall) ---
        if epoch % vis_epoch_interval == 0:
            with torch.no_grad():
                try:
                    vis_batch_data = next(iter(val_loader))
                    if isinstance(vis_batch_data, (list, tuple)):
                        vis_x = vis_batch_data[0].to(device)
                    else:
                        vis_x = vis_batch_data.to(device)
                    n_examples = min(num_vis_examples, vis_x.size(0))
                    vis_x_subset = vis_x[:n_examples]
                    _, _, recon_example, _, _ = model(vis_x_subset)
                    save_frame_examples(
                        vis_x_subset, recon_example, epoch,
                        save_dir=output_dir,
                        prefix="ema_recon",
                    )
                except StopIteration:
                    print("⚠️ Valideringsladdare tom, kan inte spara exempelbilder.")
                except Exception as e:
                    print(f"⚠️ Kunde inte spara exempelbilder för epok {epoch}: {e}")

        # --- Spara Kodboksplot (vid intervall) ---
        if epoch % vis_epoch_interval == 0 and all_val_indices_flat:
            # <<< Använd model.quantizer >>>
            if hasattr(model.quantizer, 'num_embeddings'):
                encoding_indices_all_val_flat = torch.cat(all_val_indices_flat, dim=0)
                actual_num_embeddings = model.quantizer.num_embeddings
                save_codebook_usage_plot(
                    encoding_indices_all_val_flat, actual_num_embeddings,
                    epoch, save_dir=log_dir,
                )
            else:
                print("Varning: Kan inte hämta num_embeddings från quantizer.")

        # --- Spara Checkpoints & Early stopping ---
        current_val_loss = avg_val_loss
        is_best = current_val_loss < best_val_loss - min_delta
        if is_best:
            best_val_loss = current_val_loss
            wait_counter = 0
            not_improved_epochs = []
            best_ckpt_path = checkpoint_dir / 'vqvae_best.pth'
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'loss': current_val_loss, 'best_val_loss': best_val_loss, 'perplexity': avg_val_perplexity
            }, best_ckpt_path)
            print(f"✅ Epoch {epoch}: Ny bästa modell sparad! (val_loss: {current_val_loss:.6f}) till {best_ckpt_path}")
        else:
            wait_counter += 1
            not_improved_epochs.append(epoch)
            if epoch > start_epoch:
                print(f"⏳ Ingen förbättring ({wait_counter}/{patience}) – val_loss: {current_val_loss:.6f} (bästa: {best_val_loss:.6f}). Förbättrades ej epoker: {not_improved_epochs}")
            if wait_counter >= patience:
                print(f"⛔️ Early stopping slog till vid epoch {epoch}! Ingen förbättring på {patience} epoker.")
                break

        if wait_counter < patience:
            latest_ckpt_path = checkpoint_dir / f'vqvae_latest_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'loss': current_val_loss, 'best_val_loss': best_val_loss, 'perplexity': avg_val_perplexity
            }, latest_ckpt_path)

        # --- LR Scheduling ---
        scheduler.step(current_val_loss)

    print("Träning avslutad.")

# --- END OF FILE train_vqvae.py ---
