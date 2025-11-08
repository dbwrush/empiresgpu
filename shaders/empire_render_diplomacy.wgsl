// Diplomacy Perspective Render Shader
// Shows empires color-coded by their diplomatic relation to a selected perspective empire
// Self: Blue, Allied: Gold, Neutral: Green, Enemy: Red
// Cells are darkened based on neighbor diversity for border visibility

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
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 0.0, 1.0);
    out.tex_coords = in.tex_coords;
    return out;
}

// Fragment shader
@group(0) @binding(0)
var t_texture: texture_2d<f32>;
@group(0) @binding(1)
var s_texture: sampler;
@group(0) @binding(2)
var<storage, read> diplomacy_relations: array<f32>; // 256×256 relations matrix
@group(0) @binding(3)
var<uniform> perspective_empire: u32; // Which empire we're viewing from

fn u16_to_float(val: u32) -> f32 {
    return f32(val) / 65535.0;
}

fn float_to_u16(val: f32) -> u32 {
    return u32(val * 65535.0 + 0.5);
}

// Get diplomatic relation between two empires
fn get_diplomacy(empire_a: u32, empire_b: u32) -> f32 {
    if (empire_a == 0u || empire_b == 0u || empire_a > 2047u || empire_b > 2047u) {
        return 0.0;
    }
    if (empire_a == empire_b) {
        return 1.0;
    }
    
    let a_idx = empire_a - 1u;
    let b_idx = empire_b - 1u;
    let row = min(a_idx, b_idx);
    let col = max(a_idx, b_idx);
    let idx = row * 2048u + col;
    return diplomacy_relations[idx];
}

// Get color based on diplomatic stance
fn get_diplomatic_color(relation: f32, is_self: bool) -> vec3<f32> {
    if (is_self) {
        return vec3<f32>(0.3, 0.5, 1.0); // Blue for self
    }
    
    if (relation > 0.3) {
        // Allied: Gold
        return vec3<f32>(1.0, 0.85, 0.0);
    } else if (relation >= -0.3) {
        // Neutral: Green with brightness based on relation
        // Map relation from [-0.3, 0.3] to brightness [0.3, 1.0]
        // -0.3 (hostile edge) -> darker, +0.3 (friendly edge) -> brighter
        let normalized = (relation + 0.3) / 0.6; // Maps [-0.3, 0.3] to [0.0, 1.0]
        let brightness = 0.3 + normalized * 0.7; // Maps to [0.3, 1.0] brightness range
        return vec3<f32>(0.2 * brightness, 0.9 * brightness, 0.2 * brightness);
    } else {
        // Hostile: Red
        return vec3<f32>(1.0, 0.2, 0.2);
    }
}

// Sample neighbor at offset to check empire diversity
fn sample_neighbor(tex_coords: vec2<f32>, offset: vec2<f32>, tex_size: vec2<f32>) -> f32 {
    let neighbor_coords = tex_coords + offset / tex_size;
    let neighbor_sample = textureSample(t_texture, s_texture, neighbor_coords);
    return neighbor_sample.r; // Empire ID in red channel
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Apply hex offset for proper grid alignment
    let texture_dims = textureDimensions(t_texture);
    let pixel_coords = in.tex_coords * vec2<f32>(f32(texture_dims.x), f32(texture_dims.y));
    let row = i32(pixel_coords.y);
    
    // For hex grid visualization, offset even rows by 0.5 pixels (even-r layout)
    var adjusted_tex_coords = in.tex_coords;
    if ((row % 2) == 0) {
        // Offset even rows by half a pixel width
        adjusted_tex_coords.x -= 0.5 / f32(texture_dims.x);
    }
    
    let sample = textureSample(t_texture, s_texture, adjusted_tex_coords);
    let empire_id_float = sample.r;
    let empire_id = float_to_u16(empire_id_float);
    
    // If unclaimed territory, make it transparent to show terrain underneath
    if (empire_id == 0u) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    // If no perspective empire selected, show as dark gray
    if (perspective_empire == 0u) {
        return vec4<f32>(0.2, 0.2, 0.2, 1.0);
    }
    
    // Get diplomatic relation
    let relation = get_diplomacy(perspective_empire, empire_id);
    let is_self = (empire_id == perspective_empire);
    
    // Base color from diplomatic stance
    var color = get_diplomatic_color(relation, is_self);
    
    // Calculate border darkening based on neighbor diversity
    let tex_size = vec2<f32>(textureDimensions(t_texture));
    var different_neighbors = 0u;
    
    // Check 8 neighbors (unrolled for WGSL compatibility)
    var neighbor_id = float_to_u16(sample_neighbor(adjusted_tex_coords, vec2<f32>(-1.0, -1.0), tex_size));
    if (neighbor_id != empire_id) { different_neighbors++; }
    
    neighbor_id = float_to_u16(sample_neighbor(adjusted_tex_coords, vec2<f32>(0.0, -1.0), tex_size));
    if (neighbor_id != empire_id) { different_neighbors++; }
    
    neighbor_id = float_to_u16(sample_neighbor(adjusted_tex_coords, vec2<f32>(1.0, -1.0), tex_size));
    if (neighbor_id != empire_id) { different_neighbors++; }
    
    neighbor_id = float_to_u16(sample_neighbor(adjusted_tex_coords, vec2<f32>(-1.0, 0.0), tex_size));
    if (neighbor_id != empire_id) { different_neighbors++; }
    
    neighbor_id = float_to_u16(sample_neighbor(adjusted_tex_coords, vec2<f32>(1.0, 0.0), tex_size));
    if (neighbor_id != empire_id) { different_neighbors++; }
    
    neighbor_id = float_to_u16(sample_neighbor(adjusted_tex_coords, vec2<f32>(-1.0, 1.0), tex_size));
    if (neighbor_id != empire_id) { different_neighbors++; }
    
    neighbor_id = float_to_u16(sample_neighbor(adjusted_tex_coords, vec2<f32>(0.0, 1.0), tex_size));
    if (neighbor_id != empire_id) { different_neighbors++; }
    
    neighbor_id = float_to_u16(sample_neighbor(adjusted_tex_coords, vec2<f32>(1.0, 1.0), tex_size));
    if (neighbor_id != empire_id) { different_neighbors++; }
    
    // Darken based on proportion of different neighbors (borders are darker)
    let diversity = f32(different_neighbors) / 8.0;
    let darkening = 1.0 - (diversity * 0.4); // Up to 40% darker at full borders
    color *= darkening;
    
    return vec4<f32>(color, 1.0);
}
