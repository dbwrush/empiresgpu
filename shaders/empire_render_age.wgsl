// Age visualization shader - red (new) -> orange -> yellow -> green (old)

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

// HSV to RGB conversion helper
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let h_prime = h * 6.0;
    let x = c * (1.0 - abs(h_prime % 2.0 - 1.0));
    let m = v - c;
    
    var rgb: vec3<f32>;
    if h_prime < 1.0 {
        rgb = vec3<f32>(c, x, 0.0);
    } else if h_prime < 2.0 {
        rgb = vec3<f32>(x, c, 0.0);
    } else if h_prime < 3.0 {
        rgb = vec3<f32>(0.0, c, x);
    } else if h_prime < 4.0 {
        rgb = vec3<f32>(0.0, x, c);
    } else if h_prime < 5.0 {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }
    
    return rgb + vec3<f32>(m, m, m);
}

@group(0) @binding(0) var aux_texture: texture_2d<f32>; // Auxiliary texture (age data in red channel)
@group(0) @binding(1) var texture_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Apply hex offset for visual representation
    let texture_dims = textureDimensions(aux_texture);
    let pixel_coords = in.tex_coords * vec2<f32>(f32(texture_dims.x), f32(texture_dims.y));
    let row = i32(pixel_coords.y);
    
    // For hex grid visualization, offset even rows by 0.5 pixels (even-r layout)
    var adjusted_tex_coords = in.tex_coords;
    if ((row % 2) == 0) {
        // Offset even rows by half a pixel width
        adjusted_tex_coords.x -= 0.5 / f32(texture_dims.x);
    }
    
    // Sample the auxiliary texture for age data
    let aux_sample = textureSample(aux_texture, texture_sampler, adjusted_tex_coords);
    let age_normalized = aux_sample.r; // Age is stored in red channel (0.0-1.0)
    
    // Only render if there's age data (meaning there's an empire)
    // Age > 0.0 means empire exists (very small threshold for floating point precision)
    if age_normalized < 0.0001 { // Changed from <= to < to allow 0.001 values through
        return vec4<f32>(0.0, 0.0, 0.0, 0.0); // Transparent for no empire
    }
    
    // Map age (0-1) to hue rotation from red (0°) to green (120°)
    // Simply use the age value directly - it already represents the full range
    let hue = age_normalized * 0.333; // 0.333 = 120°/360° in normalized hue space
    
    // Keep saturation high and value (brightness) high for vibrant colors
    let saturation = 1.0;
    let value = 1.0;
    
    let color = hsv_to_rgb(hue, saturation, value);
    
    return vec4<f32>(color, 0.8); // Semi-transparent overlay
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Transform vertex position by camera matrix
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 0.0, 1.0);
    out.tex_coords = in.tex_coords;
    return out;
}