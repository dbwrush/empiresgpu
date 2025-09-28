@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<uniform> frame_data: u32;

@group(0) @binding(3)
var terrain_texture: texture_2d<f32>;

// High-quality pseudorandom number generator using PCG algorithm
// Returns a pseudorandom u32 based on position and frame
fn pcg_hash(x: u32, y: u32, frame: u32) -> u32 {
    // Combine inputs into a single seed
    var state = x ^ (y << 16u) ^ (frame << 8u);
    
    // PCG algorithm - permuted congruential generator
    state = state * 1664525u + 1013904223u;
    state ^= state >> 16u;
    state = state * 1664525u + 1013904223u;
    state ^= state >> 16u;
    state = state * 1664525u + 1013904223u;
    
    return state;
}

// Generic RNG function that returns a value in range [0, max_value)
fn rng_range(x: u32, y: u32, frame: u32, seed_offset: u32, max_value: u32) -> u32 {
    let hash = pcg_hash(x, y, frame + seed_offset);
    return hash % max_value;
}

// Get random hex direction (0-5) for attacking
fn get_hex_direction(x: u32, y: u32, frame: u32) -> u32 {
    return rng_range(x, y, frame, 0u, 6u);
}

// Get cell data at position with wrapping
fn get_cell(pos: vec2<i32>) -> vec4<f32> {
    let dims = textureDimensions(input_texture);
    let wrapped_pos = vec2<i32>(
        (pos.x + i32(dims.x)) % i32(dims.x),
        (pos.y + i32(dims.y)) % i32(dims.y)
    );
    return textureLoad(input_texture, wrapped_pos, 0);
}

// Get terrain data at position with wrapping
// Returns vec4<f32> with (altitude, unused, unused, alpha) normalized to [0,1]
fn get_terrain(pos: vec2<i32>) -> vec4<f32> {
    let dims = textureDimensions(terrain_texture);
    let wrapped_pos = vec2<i32>(
        (pos.x + i32(dims.x)) % i32(dims.x),
        (pos.y + i32(dims.y)) % i32(dims.y)
    );
    let terrain_data = textureLoad(terrain_texture, wrapped_pos, 0);
    return terrain_data; // altitude, terrain_type, unused, ocean_cutoff
}

// Convert float [0,1] to u8 [0,255]
fn float_to_u8(val: f32) -> u32 {
    return u32(clamp(val * 255.0, 0.0, 255.0));
}

// Convert u8 [0,255] to float [0,1]
fn u8_to_float(val: u32) -> f32 {
    return f32(val) / 255.0;
}

// Get hex neighbor offset based on direction (0-5) and row parity
// Even-r offset coordinate system (even rows are offset)
fn get_hex_neighbor_offset(direction: u32, row: i32) -> vec2<i32> {
    let is_even_row = (row % 2) == 0;
    
    switch (direction) {
        case 0u: { 
            // East
            return vec2<i32>(1, 0);
        }
        case 1u: { 
            // Northeast  
            if (is_even_row) {
                return vec2<i32>(1, -1);  // Even row: right and up
            } else {
                return vec2<i32>(0, -1);  // Odd row: up only
            }
        }
        case 2u: { 
            // Northwest
            if (is_even_row) {
                return vec2<i32>(0, -1);  // Even row: up only
            } else {
                return vec2<i32>(-1, -1); // Odd row: left and up
            }
        }
        case 3u: { 
            // West
            return vec2<i32>(-1, 0);
        }
        case 4u: { 
            // Southwest
            if (is_even_row) {
                return vec2<i32>(0, 1);   // Even row: down only
            } else {
                return vec2<i32>(-1, 1);  // Odd row: left and down
            }
        }
        case 5u: { 
            // Southeast
            if (is_even_row) {
                return vec2<i32>(1, 1);   // Even row: right and down
            } else {
                return vec2<i32>(0, 1);   // Odd row: down only
            }
        }
        default: { return vec2<i32>(0, 0); }
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pos = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let dims = textureDimensions(input_texture);
    
    if (pos.x >= i32(dims.x) || pos.y >= i32(dims.y)) {
        return;
    }
    
    let current_cell = get_cell(pos);
    let empire_id = float_to_u8(current_cell.r);
    let strength = float_to_u8(current_cell.g);
    let need = float_to_u8(current_cell.b);
    let action = float_to_u8(current_cell.a);
    
    // Read terrain data (for AI decisions and water detection)
    let terrain = get_terrain(pos);
    let altitude = terrain.r;      // 0.0 to 1.0 (elevation)
    
    // Fixed ocean cutoff (matching empiresbevy's approach)
    let ocean_cutoff = 0.53;
    
    // Check if this is water (below ocean cutoff)
    let is_water = altitude < ocean_cutoff;
    
    // Calculate terrain factor for strength generation (empiresbevy style)
    // Lower elevation (closer to ocean) = higher generation rate
    let terrain_strength = 0.7; // TERRAIN_STRENGTH constant from empiresbevy
    let normalized_land_height = (altitude - ocean_cutoff) / (1.0 - ocean_cutoff); // 0-1 for land
    let terrain_factor = pow(1.0 - normalized_land_height, 1.0 + 4.0 * terrain_strength) * terrain_strength + (1.0 - terrain_strength);
    
    var new_empire_id = empire_id;
    var new_strength = strength;
    var new_need = need;
    var new_action = action;
    
    // If this is water, clear any empire presence
    if (is_water) {
        new_empire_id = 0u;
        new_strength = 0u;
        new_need = 0u;
        new_action = 0u;
    } else {
        // Handle unclaimed land cells - set initial strength based on terrain
        if (empire_id == 0u && strength == 0u) {
            // Convert terrain altitude to initial strength (0-255 range)
            // DEBUG: Much lower initial strength to allow expansion testing
            new_strength = u32(altitude * 50.0); // DEBUG: Much weaker initial strength
        }
        
        // PHASE 1: Resolve incoming attacks and update strength/ownership
        // Calculate total incoming attack strength from all directions
        var total_attack_strength = 0.0;
        var attacking_empire = 0u;
        var found_attacker = false;
        
        // Check all 6 hex directions for incoming attacks
        for (var dir = 0u; dir < 6u; dir++) {
            let neighbor_offset = get_hex_neighbor_offset(dir, pos.y);
            let neighbor_pos = pos + neighbor_offset;
            let neighbor_cell = get_cell(neighbor_pos);
            let neighbor_empire = float_to_u8(neighbor_cell.r);
            let neighbor_strength = float_to_u8(neighbor_cell.g);
            let neighbor_action = float_to_u8(neighbor_cell.a);
            
            // If neighbor has an empire and is attacking (action != 0)
            if (neighbor_empire != 0u && neighbor_action != 0u) {
                let neighbor_direction = neighbor_action & 7u; // Extract direction (0-5)
                
                // Calculate where the neighbor is attacking
                let neighbor_target_offset = get_hex_neighbor_offset(neighbor_direction, neighbor_pos.y);
                let neighbor_target = neighbor_pos + neighbor_target_offset;
                
                // If the neighbor is attacking this cell
                if (neighbor_target.x == pos.x && neighbor_target.y == pos.y) {
                    // Only count attacks from different empires
                    if (neighbor_empire != empire_id) {
                        let attack_strength = f32(neighbor_strength) / 255.0; // Convert to 0-1 range
                        total_attack_strength += attack_strength;
                        
                        // Remember one of the attacking empires (last one wins if multiple)
                        if (!found_attacker) {
                            attacking_empire = neighbor_empire;
                            found_attacker = true;
                        }
                    }
                }
            }
        }
        
        // Apply combat resolution (3:1 defender advantage)
        if (total_attack_strength > 0.0) {
            let current_strength_f = f32(new_strength) / 255.0; // Convert to 0-1 range
            let damage = total_attack_strength / 3.0; // 3:1 defender advantage
            
            if (damage >= current_strength_f) {
                // Cell is conquered - change ownership
                new_empire_id = attacking_empire;
                new_strength = u32((damage - current_strength_f) * 255.0); // Remaining attack strength becomes new strength
                new_need = 64u; // Reset need for newly conquered cell
            } else {
                // Cell survives but takes damage
                new_strength = u32((current_strength_f - damage) * 255.0);
            }
        }
        
        // PHASE 2: Generate strength for occupied cells and plan attacks
        if (new_empire_id != 0u) {
            // Generate strength based on terrain factor (only for occupied cells)
            let strength_generation = terrain_factor * 255.0 / 10.0; // DEBUG: Much faster generation for testing
            new_strength = min(255u, new_strength + u32(strength_generation));
            
            // Plan attack if we have enough strength
            let current_strength_f = f32(new_strength) / 255.0;
            
            // Look for weakest enemy neighbor to attack
            var min_enemy_strength = 2.0; // Higher than max possible (1.0)
            var chosen_direction = 6u; // Invalid direction initially
            
            for (var dir = 0u; dir < 6u; dir++) {
                let target_offset = get_hex_neighbor_offset(dir, pos.y);
                let target_pos = pos + target_offset;
                let target_cell = get_cell(target_pos);
                let target_empire = float_to_u8(target_cell.r);
                let target_strength = f32(float_to_u8(target_cell.g)) / 255.0;
                
                // Check if target is water
                let target_terrain = get_terrain(target_pos);
                let target_is_water = target_terrain.r < ocean_cutoff;
                
                // Only consider targets that are different empire, on land, and weaker
                if (target_empire != new_empire_id && !target_is_water) {
                    if (target_strength < min_enemy_strength) {
                        // Check if we have enough strength to make meaningful attack
                        let required_strength = target_strength * 3.0; // Need to overcome 3:1 advantage
                        
                        // DEBUG: Much lower threshold for testing
                        if (current_strength_f > required_strength * 0.1) { // Much more aggressive for debugging
                            min_enemy_strength = target_strength;
                            chosen_direction = dir;
                        }
                    }
                }
            }
            
            // Set attack action if we found a valid target
            if (chosen_direction < 6u) {
                new_action = chosen_direction | (31u << 3u); // direction + strength allocation bits
            } else {
                new_action = 0u; // No valid targets
            }
        }
    }
    
    // Write the new cell state
    let output_color = vec4<f32>(
        u8_to_float(new_empire_id),
        u8_to_float(new_strength),
        u8_to_float(new_need),
        u8_to_float(new_action)
    );
    
    textureStore(output_texture, pos, output_color);
}
