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
    let need = cell_data.b;         // Need value (0.0 to 1.0)
    
    // Visualize need levels as a heatmap
    if (empire_id == 0.0) {
        // Unclaimed territory - completely transparent
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    } else {
        // Create a green-to-yellow-to-red heatmap for need
        var need_color = vec3<f32>(0.0, 0.0, 0.0);
        if (need < 0.5) {
            need_color = vec3<f32>(need * 2.0, 1.0, 0.0);  // Green to yellow
        } else {
            need_color = vec3<f32>(1.0, (1.0 - need) * 2.0, 0.0);  // Yellow to red
        }
        return vec4<f32>(need_color, 0.8);
    }
}
