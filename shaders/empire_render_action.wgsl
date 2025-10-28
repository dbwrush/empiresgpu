// Camera uniform buffer
struct CameraUniform {
    view_proj: mat4x4<f32>,
}

@group(1) @binding(0)
var<uniform> camera: CameraUniform;

// Vertex shader
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Transform vertex position by camera matrix
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 0.0, 1.0);
    out.tex_coords = in.tex_coords;
    return out;
}

// Fragment shader
@group(0) @binding(0)
var t_texture: texture_2d<f32>;
@group(0) @binding(1)
var s_texture: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Apply hex offset for visual representation  
    let texture_dims = textureDimensions(t_texture);
    let pixel_coords = in.tex_coords * vec2<f32>(f32(texture_dims.x), f32(texture_dims.y));
    let row = i32(pixel_coords.y);
    
    // For hex grid visualization, offset even rows by 0.5 pixels (odd-r layout)
    var adjusted_tex_coords = in.tex_coords;
    if ((row % 2) == 0) {
        // Offset even rows by half a pixel width
        adjusted_tex_coords.x -= 0.5 / f32(texture_dims.x);
    }
    
    let cell_data = textureSample(t_texture, s_texture, adjusted_tex_coords);
    let empire_id = cell_data.r;
    let action = cell_data.a;       // Action value (0.0 to 1.0)
    
    // Visualize actions - different colors for different action types
    if (empire_id == 0.0) {
        // Unclaimed territory - completely transparent
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    } else {
        // Decode action using NEW 16-bit encoding
        let action_u16 = u32(action * 65535.0);
        let amount_12bit = (action_u16 >> 3u) & 4095u; // Extract amount (bits 3-14)
        let is_reinforce = (action_u16 >> 15u) & 1u; // Extract action type (bit 15)
        
        if (amount_12bit == 0u) {
            // No action - dark gray
            return vec4<f32>(0.2, 0.2, 0.2, 0.6);
        } else if (is_reinforce == 0u) {
            // Attack action - red
            return vec4<f32>(1.0, 0.2, 0.0, 0.8);
        } else {
            // Reinforce action - blue
            return vec4<f32>(0.0, 0.5, 1.0, 0.8);
        }
    }
}