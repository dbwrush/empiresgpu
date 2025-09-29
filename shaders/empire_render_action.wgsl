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
        // Decode action using bit flags (same as compute shader)
        let action_int = u32(action * 255.0);
        let action_type = (action_int >> 7u) & 3u; // Extract action type (bits 7-8)
        
        if (action_int == 0u) {
            // No action - dark gray
            return vec4<f32>(0.2, 0.2, 0.2, 0.6);
        } else if (action_type == 1u) {
            // Attack action - red
            return vec4<f32>(1.0, 0.2, 0.0, 0.8);
        } else if (action_type == 2u) {
            // Reinforce action - blue
            return vec4<f32>(0.0, 0.5, 1.0, 0.8);
        } else {
            // Unknown action - purple
            return vec4<f32>(0.8, 0.2, 0.8, 0.8);
        }
    }
}
