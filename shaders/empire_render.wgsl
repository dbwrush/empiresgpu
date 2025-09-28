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
        adjusted_tex_coords.x += 0.5 / f32(texture_dims.x);
    }
    
    let cell_data = textureSample(t_texture, s_texture, adjusted_tex_coords);
    let empire_id = cell_data.r;
    let strength = cell_data.g;     // Strength value (0.0 to 1.0)
    let need = cell_data.b;         // Need value (0.0 to 1.0)
    
    // Visualize different empires with strength-based transparency
    if (empire_id == 0.0) {
        // Unclaimed territory - completely transparent to show terrain underneath
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    } else {
        // Use strength to determine transparency: stronger = more opaque
        // Minimum alpha of 0.5, maximum of 0.9, scaling with strength
        let strength_alpha = 0.5 + (strength * 0.4);
        
        // Use need to slightly tint the color (higher need = more intense)
        let need_intensity = 0.7 + (need * 0.3);
        
        if (empire_id < 0.004) { // Empire 1 (1/255 ≈ 0.004)
            // Empire 1 - bright red, intensity based on need
            return vec4<f32>(1.0 * need_intensity, 0.2, 0.2, strength_alpha);
        } else if (empire_id < 0.008) { // Empire 2 (2/255 ≈ 0.008)
            // Empire 2 - bright blue
            return vec4<f32>(0.2, 0.2, 1.0 * need_intensity, strength_alpha);
        } else if (empire_id < 0.012) { // Empire 3 (3/255 ≈ 0.012)
            // Empire 3 - bright green
            return vec4<f32>(0.2, 1.0 * need_intensity, 0.2, strength_alpha);
        } else {
            // Other empires - yellow
            return vec4<f32>(1.0 * need_intensity, 1.0 * need_intensity, 0.2, strength_alpha);
        }
    }
}
