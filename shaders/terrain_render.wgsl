// Terrain rendering shader - converts raw terrain data to realistic colors

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
var terrain_texture: texture_2d<f32>;
@group(0) @binding(1)
var terrain_sampler: sampler;



@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Apply hex offset for proper terrain alignment
    let texture_dims = textureDimensions(terrain_texture);
    let pixel_coords = in.tex_coords * vec2<f32>(f32(texture_dims.x), f32(texture_dims.y));
    let row = i32(pixel_coords.y);
    
    // For hex grid visualization, offset even rows by 0.5 pixels (even-r layout)
    var adjusted_tex_coords = in.tex_coords;
    if ((row % 2) == 0) {
        // Offset even rows by half a pixel width
        adjusted_tex_coords.x += 0.5 / f32(texture_dims.x);
    }
    
    let terrain_data = textureSample(terrain_texture, terrain_sampler, adjusted_tex_coords);
    let altitude = terrain_data.r;      // 0.0 to 1.0 (elevation)
    
    // Fixed ocean cutoff (matching empiresbevy's approach)
    let water_level = 0.53;
    
    if (altitude < water_level) {
        // Ocean - blue gradient based on depth (empiresbevy style)
        let water_brightness = altitude / 1.5; // Deeper = darker
        let ocean_blue = vec3<f32>(0.2, 0.4, 0.8);
        return vec4<f32>(ocean_blue * water_brightness, 1.0);
    } else {
        // Land - green to brown gradient (empiresbevy style)
        let land_brightness = altitude / 1.6; // Higher = slightly darker
        
        // Create a hue shift from green (low) to brown/yellow (high)
        let hue_factor = (altitude - water_level) / (1.0 - water_level); // 0-1 for land
        
        // Green for low elevation, transition to brown for high elevation
        let low_color = vec3<f32>(0.2, 0.6, 0.2);  // Green
        let high_color = vec3<f32>(0.5, 0.4, 0.2); // Brown
        
        let base_color = mix(low_color, high_color, hue_factor);
        return vec4<f32>(base_color * land_brightness, 1.0);
    }
}
