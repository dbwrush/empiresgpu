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
var t_aux_texture: texture_2d<f32>;
@group(0) @binding(1)
var s_aux_texture: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Apply hex offset for visual representation  
    let texture_dims = textureDimensions(t_aux_texture);
    let pixel_coords = in.tex_coords * vec2<f32>(f32(texture_dims.x), f32(texture_dims.y));
    let row = i32(pixel_coords.y);
    
    // For hex grid visualization, offset even rows by 0.5 pixels (odd-r layout)
    var adjusted_tex_coords = in.tex_coords;
    if ((row % 2) == 0) {
        // Offset even rows by half a pixel width
        adjusted_tex_coords.x -= 0.5 / f32(texture_dims.x);
    }
    
    let aux_data = textureSample(t_aux_texture, s_aux_texture, adjusted_tex_coords);
    let boat_need = aux_data.g;  // boat_need stored in green channel (0.0 to 2.0 normalized to 0-1)
    
    // Only render if there's boat_need data
    if boat_need < 0.0001 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0); // Transparent for no boat_need
    }
    
    // Create a blue-to-cyan-to-yellow-to-red heatmap for boat_need
    // boat_need ranges from 0.0 to 2.0, but we normalize it to 0-1 for storage
    // So we need to scale it back: actual_value = boat_need * 2.0
    let boat_need_actual = boat_need * 2.0;
    let boat_need_normalized = clamp(boat_need_actual / 2.0, 0.0, 1.0);
    
    var color = vec3<f32>(0.0, 0.0, 0.0);
    if (boat_need_normalized < 0.25) {
        // Blue to cyan (0.0 to 0.5 boat_need)
        let t = boat_need_normalized * 4.0;
        color = vec3<f32>(0.0, t, 1.0);
    } else if (boat_need_normalized < 0.5) {
        // Cyan to green (0.5 to 1.0 boat_need)
        let t = (boat_need_normalized - 0.25) * 4.0;
        color = vec3<f32>(0.0, 1.0, 1.0 - t);
    } else if (boat_need_normalized < 0.75) {
        // Green to yellow (1.0 to 1.5 boat_need)
        let t = (boat_need_normalized - 0.5) * 4.0;
        color = vec3<f32>(t, 1.0, 0.0);
    } else {
        // Yellow to red (1.5 to 2.0 boat_need)
        let t = (boat_need_normalized - 0.75) * 4.0;
        color = vec3<f32>(1.0, 1.0 - t, 0.0);
    }
    
    return vec4<f32>(color, 0.8);
}
