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
    let terrain_data = textureSample(terrain_texture, terrain_sampler, in.tex_coords);
    let altitude = terrain_data.r;      // 0.0 to 1.0
    let humidity = terrain_data.g;      // 0.0 to 1.0  
    let temperature = terrain_data.b;   // 0.0 to 1.0
    
    // Water level threshold
    let water_level = 0.3;
    
    if (altitude < water_level) {
        // Water - deeper water is darker blue
        let water_depth = (water_level - altitude) / water_level; // 0.0 = shallow, 1.0 = deep
        let shallow_blue = vec3<f32>(0.4, 0.7, 1.0);  // Light blue
        let deep_blue = vec3<f32>(0.1, 0.2, 0.6);     // Dark blue
        let water_color = mix(shallow_blue, deep_blue, water_depth);
        return vec4<f32>(water_color, 1.0);
    }
    
    // Land terrain - normalize altitude for land (0.0 = water level, 1.0 = highest peak)
    let land_altitude = (altitude - water_level) / (1.0 - water_level);
    
    // High altitude (> 0.8) - mountains
    if (land_altitude > 0.8) {
        if (temperature < 0.3) {
            // Snow-capped peaks
            let snow_white = vec3<f32>(0.95, 0.95, 1.0);
            let peak_gray = vec3<f32>(0.7, 0.7, 0.75);
            let snow_factor = (1.0 - temperature / 0.3);
            return vec4<f32>(mix(peak_gray, snow_white, snow_factor), 1.0);
        } else {
            // Rocky peaks
            let rock_gray = vec3<f32>(0.6, 0.55, 0.5);
            let dark_rock = vec3<f32>(0.4, 0.35, 0.3);
            let rock_color = mix(rock_gray, dark_rock, land_altitude - 0.8);
            return vec4<f32>(rock_color, 1.0);
        }
    }
    
    // Very low humidity (< 0.2) - desert regardless of other factors
    if (humidity < 0.2) {
        if (temperature > 0.7) {
            // Hot desert - sandy colors
            let sand_color = vec3<f32>(0.8, 0.7, 0.4);
            let red_sand = vec3<f32>(0.7, 0.5, 0.3);
            return vec4<f32>(mix(sand_color, red_sand, temperature - 0.7), 1.0);
        } else {
            // Cold desert - more gray/brown
            let cold_desert = vec3<f32>(0.6, 0.55, 0.45);
            return vec4<f32>(cold_desert, 1.0);
        }
    }
    
    // Mid-high altitude (0.5-0.8) with good conditions - forests
    if (land_altitude > 0.5 && land_altitude <= 0.8) {
        if (temperature > 0.6 && humidity > 0.6) {
            // Tropical/temperate forest - lush green
            let lush_green = vec3<f32>(0.2, 0.6, 0.2);
            let jungle_green = vec3<f32>(0.1, 0.5, 0.1);
            let forest_factor = humidity * temperature;
            return vec4<f32>(mix(lush_green, jungle_green, forest_factor - 0.36), 1.0);
        } else if (temperature < 0.4 && humidity > 0.3) {
            // Evergreen forest - darker green
            let evergreen = vec3<f32>(0.15, 0.4, 0.25);
            return vec4<f32>(evergreen, 1.0);
        }
    }
    
    // Lower altitude areas
    if (temperature > 0.7 && humidity > 0.7) {
        // Tropical lowlands - very lush
        let tropical = vec3<f32>(0.1, 0.7, 0.3);
        return vec4<f32>(tropical, 1.0);
    } else if (temperature > 0.5 && humidity > 0.4) {
        // Temperate grassland - lighter green
        let grassland = vec3<f32>(0.4, 0.6, 0.3);
        return vec4<f32>(grassland, 1.0);
    } else if (temperature < 0.3) {
        // Tundra - sparse, cold
        let tundra = vec3<f32>(0.5, 0.6, 0.4);
        return vec4<f32>(tundra, 1.0);
    } else {
        // Default - mixed terrain
        let mixed = vec3<f32>(0.5, 0.5, 0.4);
        return vec4<f32>(mixed, 1.0);
    }
}
