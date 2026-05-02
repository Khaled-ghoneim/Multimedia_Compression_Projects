import cv2
import numpy as np
from scipy.fftpack import dct, idct
import os
import pickle
import heapq
from collections import defaultdict

# ==========================================
# 1. Helper Functions (Transforms & Entropy)
# ==========================================
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def zigzag_scan(block):
    """Step 3: Apply Zig-zag scan to an 8x8 block"""
    indices = [
        0, 1, 8, 16, 9, 2, 3, 10,
        17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63
    ]
    flat = block.flatten()
    return [flat[i] for i in indices]

def run_length_encode(data):
    """Step 3: Run-length encoding for 1D array"""
    encoded = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i-1]:
            count += 1
        else:
            encoded.append((data[i-1], count))
            count = 1
    encoded.append((data[-1], count))
    return encoded

def build_huffman_tree(data):
    """Step 5: Entropy Coding using Huffman"""
    freq = defaultdict(int)
    for symbol in data:
        # Convert mutable lists/arrays to tuples so they can be dict keys
        if isinstance(symbol, np.ndarray) or isinstance(symbol, list):
            symbol = tuple(symbol)
        freq[symbol] += 1
        
    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapq.heapify(heap)
    
    if len(heap) == 1:
        return {heap[0][1][0]: "0"}
        
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]: pair[1] = '0' + pair[1]
        for pair in hi[1:]: pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        
    return dict(heapq.heappop(heap)[1:])

# ==========================================
# 2. Core Compression Functions
# ==========================================
def process_iframe(y_channel):
    """Step 3: Intra-frame Compression"""
    h, w = y_channel.shape
    compressed_y = np.zeros((h, w))
    encoded_blocks = []
    
    q_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                         [12, 12, 14, 19, 26, 58, 60, 55],
                         [14, 13, 16, 24, 40, 57, 69, 56],
                         [14, 17, 22, 29, 51, 87, 80, 62],
                         [18, 22, 37, 56, 68, 109, 103, 77],
                         [24, 35, 55, 64, 81, 104, 113, 92],
                         [49, 64, 78, 87, 103, 121, 120, 101],
                         [72, 92, 95, 98, 112, 100, 103, 99]])
    
    h_pad = (h // 8) * 8
    w_pad = (w // 8) * 8
    
    for i in range(0, h_pad, 8):
        for j in range(0, w_pad, 8):
            block = y_channel[i:i+8, j:j+8].astype(float)
            dct_block = dct2(block - 128)
            quantized = np.round(dct_block / q_matrix)
            compressed_y[i:i+8, j:j+8] = quantized
            
            # Apply Zig-zag and RLE
            zz_scan = zigzag_scan(quantized)
            rle_data = run_length_encode(zz_scan)
            encoded_blocks.extend(rle_data)
            
    return compressed_y, encoded_blocks

def block_matching(current_frame, ref_frame, block_size=16, search_area=8):
    """Step 4: Inter-frame Compression"""
    h, w = current_frame.shape
    h_pad = (h // block_size) * block_size
    w_pad = (w // block_size) * block_size
    
    motion_vectors = []
    residuals = np.zeros((h, w), dtype=np.int16)
    
    for i in range(0, h_pad, block_size):
        for j in range(0, w_pad, block_size):
            curr_block = current_frame[i:i+block_size, j:j+block_size].astype(int)
            min_mad = float('inf')
            best_dy, best_dx = 0, 0
            
            for dy in range(-search_area, search_area + 1):
                for dx in range(-search_area, search_area + 1):
                    ref_y, ref_x = i + dy, j + dx
                    
                    if 0 <= ref_y <= h - block_size and 0 <= ref_x <= w - block_size:
                        ref_block = ref_frame[ref_y:ref_y+block_size, ref_x:ref_x+block_size].astype(int)
                        mad = np.sum(np.abs(curr_block - ref_block))
                        
                        if mad < min_mad:
                            min_mad = mad
                            best_dy, best_dx = dy, dx
                            
            motion_vectors.append((best_dy, best_dx))
            
            match_y, match_x = i + best_dy, j + best_dx
            matched_block = ref_frame[match_y:match_y+block_size, match_x:match_x+block_size].astype(np.int16)
            residuals[i:i+block_size, j:j+block_size] = curr_block.astype(np.int16) - matched_block
            
    return motion_vectors, residuals

def calculate_psnr(original, decoded):
    mse = np.mean((original.astype(np.float64) - decoded.astype(np.float64)) ** 2)
    if mse == 0: return float('inf')
    return 10 * np.log10((255.0 ** 2) / mse)

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    os.makedirs('outputs', exist_ok=True)
    video_path = 'D:\\الكلية\\cs\\3\\semester 2\\dsp\\Multimedia_Compression_Projects\\Video_Compression\\input_video.mp4' 
    
    if not os.path.exists(video_path):
        print(f"ERROR: Could not find '{video_path}'.")
        return

    print("1. Reading Video...")
    cap = cv2.VideoCapture(video_path)
    
    # Calculate original uncompressed size (approximate for the frames we process)
    original_size_bytes = 0 
    
    bitstream_data = {
        "header": {"width": int(cap.get(3)), "height": int(cap.get(4)), "fps": cap.get(5)},
        "frames": []
    }
    
    frame_idx = 0
    ref_frame_y = None
    
    # Limit to 5 frames to keep processing time reasonable for testing
    MAX_FRAMES = 5 
    print(f"-> Processing {MAX_FRAMES} frames to demonstrate full pipeline...\n")

    while cap.isOpened() and frame_idx < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret: break
            
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        curr_frame_y = yuv_frame[:,:,0]
        original_size_bytes += curr_frame_y.nbytes
        
        # Step 2: Frame Type Decision
        if frame_idx % 10 == 0:  # I-Frame
            print(f"Processing Frame {frame_idx} (I-Frame)...")
            compressed_y_matrix, rle_data = process_iframe(curr_frame_y)
            
            # Step 5: Entropy Coding (Huffman) on RLE Data
            huffman_dict = build_huffman_tree(rle_data)
            
            # Step 6: Package into bitstream
            bitstream_data["frames"].append({
                "type": "I",
                "huffman_tree": huffman_dict,
                "encoded_data": rle_data # In a real system, this would be binary string
            })
            
            ref_frame_y = curr_frame_y
            
        else:  # P-Frame
            print(f"Processing Frame {frame_idx} (P-Frame)...")
            mvs, residuals = block_matching(curr_frame_y, ref_frame_y)
            
            # Step 5: Entropy Coding (Huffman) on Motion Vectors
            huffman_dict_mvs = build_huffman_tree(mvs)
            
            # Step 6: Package
            bitstream_data["frames"].append({
                "type": "P",
                "huffman_tree_mvs": huffman_dict_mvs,
                "motion_vectors": mvs,
                "residuals": residuals # In a real system, residuals would also be DCT'd and Quantized
            })
            
            # PSNR Evaluation for P-Frame
            reconstructed_y = np.clip(ref_frame_y.astype(np.int16) + residuals, 0, 255).astype(np.uint8)
            psnr = calculate_psnr(curr_frame_y, reconstructed_y)
            print(f"   -> PSNR: {psnr:.2f} dB")
            
            ref_frame_y = curr_frame_y # Update reference frame
            
        frame_idx += 1
        
    cap.release()

    print("\n4. Bitstream Formation (Packaging)...")
    bitstream_path = 'outputs/compressed_video.bin'
    with open(bitstream_path, 'wb') as f:
        pickle.dump(bitstream_data, f)
    
    print(f"-> Bitstream saved to {bitstream_path}")

    # Step 7: Testing & Evaluation (Compression Ratio)
    compressed_size_bytes = os.path.getsize(bitstream_path)
    compression_ratio = original_size_bytes / compressed_size_bytes
    
    print("\n==========================================")
    print("PROJECT EVALUATION SUMMARY:")
    print("==========================================")
    print(f"Original Video Size (Y-channel): {original_size_bytes / 1024:.2f} KB")
    print(f"Compressed Bitstream Size:       {compressed_size_bytes / 1024:.2f} KB")
    print(f"Compression Ratio:               {compression_ratio:.2f}x")
    print("==========================================\n")

if __name__ == "__main__":
    main()