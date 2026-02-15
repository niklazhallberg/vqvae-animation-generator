"""
Latent Walk - Morphing mellan frames via VQ-VAE latent space
Genererar smooth transitions mellan två animationsframes
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
import imageio
from torchvision import transforms
from models.vqvae_model import VQVAE
from config import Config

# ===========================
# 🎨 KONFIGURATION
# ===========================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate latent walk between two frames')
    parser.add_argument('--model_path', type=str, 
                       default='outputs/checkpoints/vqvae_best.pth',
                       help='Path to trained VQ-VAE checkpoint')
    parser.add_argument('--frame_a', type=str, required=True,
                       help='Path to first frame (e.g., data/frames_animation1/frame_0000.png)')
    parser.add_argument('--frame_b', type=str, required=True,
                       help='Path to second frame (e.g., data/frames_animation5/frame_0090.png)')
    parser.add_argument('--steps', type=int, default=60,
                       help='Number of interpolation steps (default: 60)')
    parser.add_argument('--output', type=str, default='outputs/latent_walk.mp4',
                       help='Output video path')
    parser.add_argument('--fps', type=int, default=30,
                       help='Output video FPS (default: 30)')
    parser.add_argument('--save_frames', action='store_true',
                       help='Save individual frames as images')
    return parser.parse_args()


# ===========================
# 🔧 HJÄLPFUNKTIONER
# ===========================

def load_model(checkpoint_path: str, device: torch.device) -> VQVAE:
    """
    Ladda tränad VQ-VAE modell
    
    Args:
        checkpoint_path: Path till checkpoint (.pth fil)
        device: torch device (mps/cuda/cpu)
    
    Returns:
        Laddad VQVAE modell i eval mode
    """
    print(f"Laddar modell från: {checkpoint_path}")
    
    # Ladda checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Extrahera model state
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Checkpoint från epoch: {checkpoint.get('epoch', 'unknown')}")
    else:
        state_dict = checkpoint
    
    # Skapa modell (exakt samma parametrar som start_training.py)
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
    
    # Ladda weights
    model.load_state_dict(state_dict)
    model.eval()
    
    print("✅ Modell laddad!")
    return model


def load_frame(image_path: str, device: torch.device) -> torch.Tensor:
    """
    Ladda och preprocessa en frame
    
    Args:
        image_path: Path till bildfil
        device: torch device
    
    Returns:
        Tensor [1, 1, 64, 64] redo för modellen
    """
    # Ladda bild
    img = Image.open(image_path).convert('L')  # Grayscale
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor()
    ])
    
    # Applicera transform och lägg till batch dimension
    tensor = transform(img).unsqueeze(0).to(device)  # [1, 1, 64, 64]
    
    return tensor


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Konvertera tensor till numpy array för video
    
    Args:
        tensor: [1, 1, H, W] tensor
    
    Returns:
        [H, W, 3] numpy array (0-255, uint8, RGB)
    """
    # Ta bort batch och channel dimensions
    img = tensor.squeeze().cpu().numpy()
    
    # Clamp till [0, 1] och konvertera till [0, 255]
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    
    # Konvertera grayscale till RGB (duplicera över 3 kanaler)
    img_rgb = np.stack([img, img, img], axis=-1)  # [H, W, 3]
    
    return img_rgb


# ===========================
# 🎬 LATENT WALK FUNKTION
# ===========================

def latent_walk(
    model: VQVAE,
    frame_a: torch.Tensor,
    frame_b: torch.Tensor,
    steps: int = 60,
) -> list[torch.Tensor]:
    """
    Generera smooth morph mellan två frames via latent space interpolation
    
    Args:
        model: Tränad VQVAE modell
        frame_a: Start frame [1, 1, 64, 64]
        frame_b: End frame [1, 1, 64, 64]
        steps: Antal interpolationssteg (minst 2)

    Returns:
        Lista med interpolerade frames (tensors)
    """
    if steps < 2:
        raise ValueError(f"steps måste vara >= 2, fick {steps}")

    print(f"\n🎨 Genererar latent walk med {steps} steg...")

    frames = []

    with torch.no_grad():
        # Encode båda frames till continuous latent space (FÖRE quantization)
        # Detta ger smidigare interpolation än att interpolera diskreta codes
        
        # frame → encoder → z_e (continuous)
        z_e_a = model.encoder(frame_a)  # [1, 64, 8, 8]
        z_e_b = model.encoder(frame_b)  # [1, 64, 8, 8]
        
        print(f"Latent space shape: {z_e_a.shape}")
        
        # Interpolera i continuous space
        for i in range(steps):
            # Linear interpolation: alpha går från 0 → 1
            alpha = i / (steps - 1)
            
            # Weighted average av latent vectors
            z_e_mixed = (1 - alpha) * z_e_a + alpha * z_e_b
            
            # Quantize mixed vector (hitta närmaste codes i codebook)
            z_q_mixed, *_ = model.quantizer(z_e_mixed)
            
            # Decode tillbaka till image space
            frame_mixed = model.decoder(z_q_mixed)
            
            frames.append(frame_mixed)
            
            # Progress indicator
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Progress: {i+1}/{steps} frames ({alpha*100:.1f}%)")
    
    print(f"✅ {len(frames)} frames genererade!")
    return frames


# ===========================
# 💾 SAVE FUNCTIONS
# ===========================

def save_video(frames: list[torch.Tensor], output_path: str, fps: int = 30):
    """
    Spara frames som video
    
    Args:
        frames: Lista med frame tensors
        output_path: Output video path (.mp4)
        fps: Frames per second
    """
    print(f"\n💾 Sparar video till: {output_path}")
    
    # Konvertera tensors till numpy arrays
    frame_arrays = [tensor_to_image(f) for f in frames]
    
    # Skapa output directory om den inte finns
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Spara som video med imageio (minimal parameters)
    imageio.mimsave(
        output_path, 
        frame_arrays, 
        fps=fps,
        codec='libx264'
    )
    
    print(f"✅ Video sparad! ({len(frames)} frames @ {fps} FPS)")
    print(f"   Duration: {len(frames)/fps:.2f} seconds")


def save_frames_as_images(frames: list[torch.Tensor], output_dir: str = 'outputs/latent_walk_frames'):
    """
    Spara varje frame som individuell bild
    
    Args:
        frames: Lista med frame tensors
        output_dir: Output directory för frames
    """
    print(f"\n💾 Sparar frames till: {output_dir}/")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, frame in enumerate(frames):
        img_array = tensor_to_image(frame)
        img = Image.fromarray(img_array, mode='RGB')  # RGB eftersom vi konverterade till 3 kanaler
        img.save(output_path / f"frame_{i:04d}.png")
    
    print(f"✅ {len(frames)} frames sparade!")


# ===========================
# 🎯 MAIN FUNCTION
# ===========================

def main():
    """Main execution"""
    args = parse_args()
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print("=" * 60)
    print("🎨 LATENT WALK - VQ-VAE Morphing Generator")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Frame A: {args.frame_a}")
    print(f"Frame B: {args.frame_b}")
    print(f"Steps: {args.steps}")
    print(f"Output: {args.output}")
    
    # Ladda modell
    model = load_model(args.model_path, device)
    
    # Ladda frames
    print(f"\n📸 Laddar frames...")
    frame_a = load_frame(args.frame_a, device)
    frame_b = load_frame(args.frame_b, device)
    print(f"✅ Frames laddade! Shape: {frame_a.shape}")
    
    # Generera latent walk
    frames = latent_walk(model, frame_a, frame_b, args.steps)
    
    # Spara som video
    save_video(frames, args.output, args.fps)
    
    # Spara frames om requested
    if args.save_frames:
        save_frames_as_images(frames)
    
    print("\n" + "=" * 60)
    print("✅ KLART!")
    print("=" * 60)
    print(f"\n🎬 Öppna video: open {args.output}")


if __name__ == '__main__':
    main()