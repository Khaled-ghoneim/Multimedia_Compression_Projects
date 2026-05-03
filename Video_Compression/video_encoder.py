import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.fftpack import dct, idct


# ───────────────────────────────────────────
# DCT HELPERS
# ───────────────────────────────────────────
def dct2(x):
    return dct(dct(x.T, norm='ortho').T, norm='ortho')


def idct2(x):
    return idct(idct(x.T, norm='ortho').T, norm='ortho')


# ───────────────────────────────────────────
# JPEG QUANTIZATION MATRIX
# ───────────────────────────────────────────
Q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])

# Pre-computed diagonal zigzag indices for an 8x8 block
_ZIGZAG_IDX = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7),
]


# ───────────────────────────────────────────
# ENCODING HELPERS
# ───────────────────────────────────────────
def zigzag(block):
    return [block[r, c] for r, c in _ZIGZAG_IDX]


def rle(data):
    out = []
    count = 1

    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            out.append((data[i - 1], count))
            count = 1

    out.append((data[-1], count))

    return out


# ───────────────────────────────────────────
# I FRAME ENCODING
# ───────────────────────────────────────────
def encode_iframe(frame):
    h, w = frame.shape

    # float32 so negative/large DCT coefficients are stored correctly
    comp = np.zeros((h, w), dtype=np.float32)
    stream = []

    for i in range(0, h // 8 * 8, 8):
        for j in range(0, w // 8 * 8, 8):
            block = frame[i:i+8, j:j+8].astype(np.float32) - 128

            q = np.round(dct2(block) / Q)

            comp[i:i+8, j:j+8] = q

            stream += rle(zigzag(q))

    return comp, stream


# ───────────────────────────────────────────
# I FRAME DECODING
# ───────────────────────────────────────────
def decode_iframe(comp):
    h, w = comp.shape

    out = np.zeros((h, w), dtype=np.float32)

    for i in range(0, h // 8 * 8, 8):
        for j in range(0, w // 8 * 8, 8):
            block = comp[i:i+8, j:j+8] * Q

            out[i:i+8, j:j+8] = idct2(block) + 128

    return np.clip(out, 0, 255).astype(np.uint8)


# ───────────────────────────────────────────
# MOTION ESTIMATION
# ───────────────────────────────────────────
def motion_est(curr, ref, bs=16, area=8):
    h, w = curr.shape

    residual = np.zeros_like(curr, dtype=np.int16)

    vectors = []

    for i in range(0, h - bs, bs):
        for j in range(0, w - bs, bs):
            best = (0, 0, 1e9)

            for dy in range(-area, area + 1):
                for dx in range(-area, area + 1):
                    y = i + dy
                    x = j + dx

                    if 0 <= y < h - bs and 0 <= x < w - bs:
                        diff = np.sum(
                            np.abs(
                                curr[i:i+bs, j:j+bs]
                                - ref[y:y+bs, x:x+bs]
                            )
                        )

                        if diff < best[2]:
                            best = (dy, dx, diff)

            dy, dx, _ = best

            vectors.append((dy, dx))

            residual[i:i+bs, j:j+bs] = (
                curr[i:i+bs, j:j+bs]
                - ref[i+dy:i+dy+bs, j+dx:j+dx+bs]
            )

    return vectors, residual


# ───────────────────────────────────────────
# BITSTREAM HELPERS
# ───────────────────────────────────────────
FRAME_TYPE_I = 0
FRAME_TYPE_P = 1


def pack_iframe(frame_idx, stream):
    """
    Header: [frame_idx: 4B][type: 1B][num_pairs: 4B]
    Payload: [value: 2B signed][count: 2B] * num_pairs
    """
    buf = bytearray()
    buf += frame_idx.to_bytes(4, 'big')
    buf += FRAME_TYPE_I.to_bytes(1, 'big')
    buf += len(stream).to_bytes(4, 'big')
    for val, cnt in stream:
        v = int(val)
        buf += v.to_bytes(2, 'big', signed=True)
        buf += cnt.to_bytes(2, 'big')
    return buf


def pack_pframe(frame_idx, vectors, residual):
    """
    Header: [frame_idx: 4B][type: 1B][num_vectors: 4B]
    Vectors: [dy: 1B signed][dx: 1B signed] * num_vectors
    Residual RLE pairs: [value: 2B signed][count: 2B]
    """
    res_rle = rle(residual.flatten().tolist())
    buf = bytearray()
    buf += frame_idx.to_bytes(4, 'big')
    buf += FRAME_TYPE_P.to_bytes(1, 'big')
    buf += len(vectors).to_bytes(4, 'big')
    for dy, dx in vectors:
        buf += dy.to_bytes(1, 'big', signed=True)
        buf += dx.to_bytes(1, 'big', signed=True)
    buf += len(res_rle).to_bytes(4, 'big')
    for val, cnt in res_rle:
        v = int(val)
        buf += v.to_bytes(2, 'big', signed=True)
        buf += cnt.to_bytes(2, 'big')
    return buf


# ───────────────────────────────────────────
# PSNR
# ───────────────────────────────────────────
def psnr(a, b):
    # cast to float32 to avoid uint8 underflow (e.g. 10 - 20 = 246)
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)

    if mse == 0:
        return np.inf

    return 10 * np.log10(255 ** 2 / mse)


# ───────────────────────────────────────────
# VIDEO PIPELINE
# ───────────────────────────────────────────
def run_pipeline(params, log, progress, done):
    cap = cv2.VideoCapture(params["video_path"])

    reference = None
    ref_recon = None

    frame_types = []
    psnr_values = []
    frame_pairs = []

    original_size   = 0
    compressed_size = 0

    max_frames  = params["max_frames"]
    i_interval  = params["i_interval"]
    block_size  = params["block_size"]
    search_area = params["search_area"]
    output_dir  = params.get("output_dir", "outputs")

    os.makedirs(output_dir, exist_ok=True)
    bitstream_path = os.path.join(output_dir, "video.bin")

    with open(bitstream_path, "wb") as bs_file:

        for i in range(max_frames):
            ret, frame = cap.read()

            if not ret:
                break

            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y   = yuv[:, :, 0]

            original_size += y.nbytes

            if i % i_interval == 0:
                log(f"Encoding I-frame {i}")

                comp, stream = encode_iframe(y)
                recon        = decode_iframe(comp)

                packet = pack_iframe(i, stream)
                bs_file.write(packet)
                compressed_size += len(packet)

                reference = y
                ref_recon = recon
                frame_types.append("I")

            else:
                log(f"Encoding P-frame {i}")

                vectors, residual = motion_est(
                    y,
                    reference,
                    bs=block_size,
                    area=search_area,
                )

                recon = np.clip(
                    ref_recon.astype(np.int16) + residual,
                    0, 255
                ).astype(np.uint8)

                packet = pack_pframe(i, vectors, residual)
                bs_file.write(packet)
                compressed_size += len(packet)

                reference = y
                ref_recon = recon
                frame_types.append("P")

            psnr_values.append(psnr(y, recon))

            if len(frame_pairs) < 6:
                frame_pairs.append((y.copy(), recon.copy()))

            progress(int(100 * (i + 1) / max_frames))

    cap.release()
    log("Saved: video.bin")

    avg_psnr = np.mean(psnr_values)
    ratio    = round(original_size / compressed_size, 2) if compressed_size != 0 else 1

    # ───────────────────────────────────────
    # SAVE PSNR CHART
    # ───────────────────────────────────────
    log("Saving PSNR chart...")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    colors = ["#E53935" if t == "I" else "#1E88E5" for t in frame_types]
    ax1.bar(range(len(psnr_values)), psnr_values, color=colors, width=0.8)
    ax1.plot(range(len(psnr_values)), psnr_values, color="#333333", linewidth=1)
    ax1.axhline(avg_psnr, color="#FF6F00", linestyle="--", linewidth=1.5, label=f"Avg: {avg_psnr:.2f} dB")
    ax1.set_title("PSNR Per Frame  (red = I-frame, blue = P-frame)")
    ax1.set_xlabel("Frame index")
    ax1.set_ylabel("PSNR (dB)")
    ax1.legend()
    fig1.tight_layout()
    psnr_path = os.path.join(output_dir, "psnr_chart.png")
    fig1.savefig(psnr_path, dpi=150)
    plt.close(fig1)
    log("Saved: psnr_chart.png")

    # ───────────────────────────────────────
    # SAVE FRAME TYPE CHART
    # ───────────────────────────────────────
    log("Saving frame type chart...")
    fig2, ax2 = plt.subplots(figsize=(10, 2.5))
    frame_values = [1 if f == "I" else 0 for f in frame_types]
    ax2.step(range(len(frame_values)), frame_values, where="mid", color="#333333")
    ax2.fill_between(range(len(frame_values)), frame_values, step="mid", alpha=0.3, color="#E53935")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["P-frame", "I-frame"])
    ax2.set_xlabel("Frame index")
    ax2.set_title("Frame Type Sequence")
    fig2.tight_layout()
    frame_type_path = os.path.join(output_dir, "frame_types.png")
    fig2.savefig(frame_type_path, dpi=150)
    plt.close(fig2)
    log("Saved: frame_types.png")

    # ───────────────────────────────────────
    # SAVE FRAME COMPARISON
    # ───────────────────────────────────────
    log("Saving frame comparison...")
    n = len(frame_pairs)
    if n > 0:
        fig3, axes = plt.subplots(2, n, figsize=(3 * n, 6))
        if n == 1:
            axes = np.expand_dims(axes, axis=1)
        for col, (orig, recon) in enumerate(frame_pairs):
            axes[0, col].imshow(orig,  cmap="gray", vmin=0, vmax=255)
            axes[0, col].set_title(f"Original #{col}", fontsize=8)
            axes[0, col].axis("off")
            axes[1, col].imshow(recon, cmap="gray", vmin=0, vmax=255)
            axes[1, col].set_title(f"Reconstructed #{col}", fontsize=8)
            axes[1, col].axis("off")
        fig3.suptitle("Original vs Reconstructed Frames (Y channel)")
        fig3.tight_layout()
        compare_path = os.path.join(output_dir, "frame_comparison.png")
        fig3.savefig(compare_path, dpi=150)
        plt.close(fig3)
        log("Saved: frame_comparison.png")

    log("All outputs saved to: outputs/")

    # ───────────────────────────────────────
    # RETURN DATA TO GUI
    # ───────────────────────────────────────
    data = {
        "frame_types":    frame_types,
        "psnr_values":    psnr_values,
        "frame_pairs":    frame_pairs,
        "bitstream_path": bitstream_path,
    }

    summary = {
        "avg_psnr":     avg_psnr,
        "ratio":        ratio,
        "total_frames": len(frame_types),
    }

    log(f"Bitstream written → {bitstream_path}")
    log("Video compression completed.")

    done(data, summary)