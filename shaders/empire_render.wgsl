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

// Hash function for consistent pseudo-random values based on empire ID
fn hash_empire_id(id: u32) -> u32 {
    var x = id;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = (x >> 16u) ^ x;
    return x;
}

// Convert HSV to RGB color space
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let x = c * (1.0 - abs(((h / 60.0) % 2.0) - 1.0));
    let m = v - c;
    
    var rgb = vec3<f32>(0.0, 0.0, 0.0);
    
    if (h >= 0.0 && h < 60.0) {
        rgb = vec3<f32>(c, x, 0.0);
    } else if (h >= 60.0 && h < 120.0) {
        rgb = vec3<f32>(x, c, 0.0);
    } else if (h >= 120.0 && h < 180.0) {
        rgb = vec3<f32>(0.0, c, x);
    } else if (h >= 180.0 && h < 240.0) {
        rgb = vec3<f32>(0.0, x, c);
    } else if (h >= 240.0 && h < 300.0) {
        rgb = vec3<f32>(x, 0.0, c);
    } else if (h >= 300.0 && h < 360.0) {
        rgb = vec3<f32>(c, 0.0, x);
    }
    
    return rgb + vec3<f32>(m, m, m);
}

// Generate consistent empire color based on ID
fn get_empire_color(empire_id: f32) -> vec3<f32> {
    if (empire_id == 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0); // Black for unclaimed (transparent anyway)
    }
    
    // Convert to integer for hashing
    let id_int = u32(empire_id * 255.0);
    let hash_val = hash_empire_id(id_int);
    
    // Generate HSV values from hash
    let hue = f32(hash_val % 360u); // Full hue range 0-360
    let saturation = 0.7 + (f32((hash_val >> 8u) % 30u) / 100.0); // 0.7-1.0 (high saturation, avoid grays)
    let value = 0.6 + (f32((hash_val >> 16u) % 20u) / 100.0); // 0.6-0.8 (mid-range brightness)
    
    return hsv_to_rgb(hue, saturation, value);
}

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
        
        // Get empire color based on ID
        let base_color = get_empire_color(empire_id);
        
        // Use need to slightly intensify the color (higher need = more intense)
        let need_intensity = 0.8 + (need * 0.2);
        let final_color = base_color * need_intensity;
        
        return vec4<f32>(final_color, strength_alpha);
    }
}
