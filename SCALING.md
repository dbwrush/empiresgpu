# Scaling Empire Simulations Beyond GPU Texture Limits

## Current Limitation

GPU textures are typically limited to 8192x8192 pixels on most modern hardware. This constrains our empire simulation to approximately 67 million cells.

## Current Implementation

The system automatically detects GPU limits and constrains simulation size:
- Requests larger than GPU limit are automatically reduced to maximum supported size
- Warning messages inform users about the constraint
- Simulation continues to work at maximum supported resolution

## Approaches for Larger Simulations

### 1. Storage Buffer Approach (Recommended)

Replace texture-based storage with compute shader storage buffers:

**Advantages:**
- Much larger size limits (typically gigabytes instead of texture memory limits)
- Better memory access patterns for large simulations
- Can support simulations of 65536x65536 or larger

**Implementation:**
- Replace `texture_2d<f32>` with `buffer<u32>` in compute shaders
- Use 1D buffer indexing: `index = y * width + x`
- Keep a smaller texture for display/rendering purposes
- Copy relevant sections from storage buffer to display texture

### 2. Tiled Approach

Split the simulation into multiple 8192x8192 tiles:

**Advantages:**
- Uses existing texture-based approach
- Handles edge cases between tiles
- Can theoretically scale indefinitely

**Challenges:**
- Complex edge handling between tiles
- Memory management for multiple textures
- Synchronization between tiles

### 3. Hierarchical Approach

Use multiple levels of detail:

**Advantages:**
- Efficient for sparse simulations
- Can focus computation on active areas
- Scales well with non-uniform activity

**Implementation:**
- Low-resolution overview grid
- High-resolution detail grids for active areas
- Dynamic allocation of detail grids

## Storage Buffer Implementation Guide

### Compute Shader Changes
```wgsl
// Instead of:
@group(0) @binding(0) var input_texture: texture_2d<f32>;

// Use:
@group(0) @binding(0) var<storage, read> input_buffer: array<u32>;

// Access pattern:
fn get_cell(x: u32, y: u32, width: u32) -> u32 {
    return input_buffer[y * width + x];
}
```

### Rust Changes
```rust
// Create storage buffer instead of texture
let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("Empire Storage Buffer"),
    size: (game_size * game_size * 4) as u64, // 4 bytes per cell
    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
});
```

## Memory Requirements

| Size | Cells | Memory (4 bytes/cell) |
|------|-------|---------------------|
| 8192² | 67M | 268 MB |
| 16384² | 268M | 1.07 GB |
| 32768² | 1.07B | 4.29 GB |
| 65536² | 4.29B | 17.18 GB |

## Recommended Next Steps

1. Implement storage buffer approach for compute shaders
2. Keep texture-based rendering for display
3. Add camera-based view culling to only render visible areas
4. Optimize memory usage with compressed cell data
