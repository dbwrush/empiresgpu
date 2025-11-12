// Boat rendering shader - DEBUG MODE: alternating fullscreen overlay

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct BoatInstance {
    @location(0) position: vec2<f32>,
    @location(1) color: vec3<f32>,  // R=red, G=green, B=blue (we'll encode alpha in red channel for debug)
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_pos: vec2<f32>,  // World position for checkerboard pattern
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: BoatInstance,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Apply hex offset for even rows (same as terrain rendering)
    // Even rows are offset by +0.5 in the positive x direction
    let y_coord = instance.position.y;
    let is_even_row = (u32(y_coord) % 2u) == 0u;
    let hex_offset = select(0.0, 0.5, is_even_row);
    
    // Render a 1x1 pixel square at the boat's position with hex offset
    var vertex_pos: vec2<f32>;
    
    switch vertex_index {
        case 0u: { vertex_pos = instance.position + vec2<f32>(hex_offset, 0.0); }
        case 1u: { vertex_pos = instance.position + vec2<f32>(1.0 + hex_offset, 0.0); }
        case 2u: { vertex_pos = instance.position + vec2<f32>(hex_offset, 1.0); }
        case 3u: { vertex_pos = instance.position + vec2<f32>(1.0 + hex_offset, 0.0); }
        case 4u: { vertex_pos = instance.position + vec2<f32>(1.0 + hex_offset, 1.0); }
        case 5u: { vertex_pos = instance.position + vec2<f32>(hex_offset, 1.0); }
        default: { vertex_pos = instance.position + vec2<f32>(hex_offset, 0.0); }
    }
    
    out.clip_position = camera.view_proj * vec4<f32>(vertex_pos, 0.0, 1.0);
    out.color = instance.color;
    out.world_pos = vertex_pos;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Use the instance color data with proper alpha
    // If color is black (0,0,0), make it fully transparent
    let is_black = in.color.r == 0.0 && in.color.g == 0.0 && in.color.b == 0.0;
    let alpha = select(1.0, 0.0, is_black);
    
    return vec4<f32>(in.color, alpha);
}
