#!/usr/bin/env python3

def calculate_cnn_output_size(input_shape):
    """
    Calculate the CNN output size for the V_Light_Barrere architecture
    input_shape: (channels, height, width)
    """
    print(f"Input shape: {input_shape}")
    
    # Block 1: conv1 (3,3), no padding, stride 1
    h1 = input_shape[1] - 3 + 1
    w1 = input_shape[2] - 3 + 1
    print(f"After conv1: (8, {h1}, {w1})")
    
    # Block 1: maxpool1 (2,2)
    h1 = h1 // 2
    w1 = w1 // 2
    print(f"After maxpool1: (8, {h1}, {w1})")
    
    # Block 2: conv2 (3,3), no padding, stride 1
    h2 = h1 - 3 + 1
    w2 = w1 - 3 + 1
    print(f"After conv2: (16, {h2}, {w2})")
    
    # Block 2: maxpool2 (2,2)
    h2 = h2 // 2
    w2 = w2 // 2
    print(f"After maxpool2: (16, {h2}, {w2})")
    
    # Block 3: conv3 (3,3), no padding, stride 1
    h3 = h2 - 3 + 1
    w3 = w2 - 3 + 1
    print(f"After conv3: (32, {h3}, {w3})")
    
    # Block 3: maxpool3 (2,2)
    h3 = h3 // 2
    w3 = w3 // 2
    print(f"After maxpool3: (32, {h3}, {w3})")
    
    # Block 4: conv4 (3,3), no padding, stride 1
    h4 = h3 - 3 + 1
    w4 = w3 - 3 + 1
    print(f"After conv4: (64, {h4}, {w4})")
    
    # Block 5: conv5 (4,2), no padding, stride 1
    h5 = h4 - 4 + 1
    w5 = w4 - 2 + 1
    print(f"After conv5: (128, {h5}, {w5})")
    
    # Final output shape
    final_shape = (128, h5, w5)
    print(f"Final CNN output shape: {final_shape}")
    
    # Calculate C*H for the collapse layer
    c_h = 128 * h5
    print(f"C*H for collapse layer: {c_h}")
    
    return c_h, w5

if __name__ == "__main__":
    print("=== Original size [64, 1024] ===")
    original_c_h, original_w = calculate_cnn_output_size((3, 64, 1024))
    print(f"\nOriginal C*H: {original_c_h}, W: {original_w}")
    
    print("\n=== New size [68, 1800] ===")
    new_c_h, new_w = calculate_cnn_output_size((3, 68, 1800))
    print(f"\nNew C*H: {new_c_h}, W: {new_w}")
    
    print(f"\nDifference in C*H: {new_c_h - original_c_h}")
    print(f"Difference in W: {new_w - original_w}")
    
    # Check if the hardcoded 1152 makes sense
    print(f"\nHardcoded 1152 vs calculated {original_c_h}")
    if original_c_h == 1152:
        print("✓ Hardcoded value matches calculation")
    else:
        print("✗ Hardcoded value doesn't match calculation")
        print(f"  Expected: {original_c_h}, Hardcoded: 1152")
