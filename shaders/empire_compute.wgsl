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
        
        // Apply combat resolution - DEBUG: More favorable to attackers
        if (total_attack_strength > 0.0) {
            let current_strength_f = f32(new_strength) / 255.0; // Convert to 0-1 range
            let damage = total_attack_strength / 2.0; // DEBUG: 2:1 defender advantage instead of 3:1
            
            if (damage >= current_strength_f) {
                // Cell is conquered - change ownership
                new_empire_id = attacking_empire;
                // DEBUG: Give conquered cell more starting strength
                let remaining_strength = damage - current_strength_f;
                new_strength = u32(clamp((remaining_strength + 0.3) * 255.0, 80.0, 255.0)); // Minimum 80 strength
                new_need = 32u; // Lower initial need for newly conquered cell
            } else {
                // Cell survives but takes damage
                new_strength = u32((current_strength_f - damage) * 255.0);
            }
        }
        
        // PHASE 2: Generate strength, calculate need, and make decisions for occupied cells
        if (new_empire_id != 0u) {
            // Calculate need based on enemy pressure and friendly support
            var calculated_need = 0.0;
            var max_enemy_strength = 0.0;
            var min_enemy_strength = 2.0;
            var min_enemy_direction = 6u;
            var max_friendly_need = 0.0;
            var max_need_direction = 6u;
            var friendly_neighbors = 0u;
            var enemy_neighbors = 0u;
            
            // Analyze all neighbors for need calculation
            for (var dir = 0u; dir < 6u; dir++) {
                let neighbor_offset = get_hex_neighbor_offset(dir, pos.y);
                let neighbor_pos = pos + neighbor_offset;
                let neighbor_cell = get_cell(neighbor_pos);
                let neighbor_empire = float_to_u8(neighbor_cell.r);
                let neighbor_strength = f32(float_to_u8(neighbor_cell.g)) / 255.0;
                let neighbor_need = f32(float_to_u8(neighbor_cell.b)) / 255.0;
                
                // Check if neighbor is water
                let neighbor_terrain = get_terrain(neighbor_pos);
                let neighbor_is_water = neighbor_terrain.r < ocean_cutoff;
                
                if (!neighbor_is_water) {
                    if (neighbor_empire == new_empire_id) {
                        // Friendly neighbor
                        friendly_neighbors++;
                        if (neighbor_need > max_friendly_need) {
                            max_friendly_need = neighbor_need;
                            max_need_direction = dir;
                        } else if (neighbor_need == max_friendly_need) {
                            // If needs are equal, randomly choose between them to avoid directional bias
                            if (rng_range(global_id.x, global_id.y, frame_data, dir * 7u, 2u) == 0u) {
                                max_need_direction = dir;
                            }
                        }
                    } else {
                        // Enemy or unclaimed neighbor
                        enemy_neighbors++;
                        calculated_need += neighbor_strength; // Add pressure from enemies
                        
                        if (neighbor_empire == 0u) {
                            // Unclaimed territory reduces need (easier to expand)
                            calculated_need -= 0.9 * neighbor_strength;
                        }
                        
                        // Track enemy strength for attack decisions
                        if (neighbor_strength > max_enemy_strength) {
                            max_enemy_strength = neighbor_strength;
                        }
                        if (neighbor_strength < min_enemy_strength && neighbor_empire != new_empire_id) {
                            min_enemy_strength = neighbor_strength;
                            min_enemy_direction = dir;
                        } else if (neighbor_strength == min_enemy_strength && neighbor_empire != new_empire_id) {
                            // If strengths are equal, randomly choose between them to avoid directional bias
                            if (rng_range(global_id.x, global_id.y, frame_data, dir * 10u, 2u) == 0u) {
                                min_enemy_direction = dir;
                            }
                        }
                    }
                }
            }
            
            // Normalize need by number of enemy neighbors
            if (enemy_neighbors > 0u) {
                calculated_need /= f32(enemy_neighbors);
            }
            
            // Apply terrain-based need factor (empiresbevy style)
            let need_factor = ((-altitude) / (1.0 - ocean_cutoff)) + (1.0 / (1.0 - ocean_cutoff));
            calculated_need *= need_factor;
            
            // Update need value
            new_need = u32(clamp(calculated_need * 255.0, 0.0, 255.0));
            
            // Apply terrain penalties to strength (empiresbevy style) - DEBUG: Much gentler
            let current_strength_f = f32(new_strength) / 255.0;
            let terrain_penalty = 0.8 + (f32(friendly_neighbors) * 0.05); // DEBUG: Much gentler penalty (80%-110%)
            let penalized_strength = current_strength_f * terrain_penalty;
            
            // Generate strength based on terrain factor (only for occupied cells)
            let strength_generation = terrain_factor / 25.0; // DEBUG: Faster generation rate
            let final_strength = penalized_strength + strength_generation;
            new_strength = u32(clamp(final_strength * 255.0, 0.0, 255.0));
            
            // Decision making: Favor expansion over reinforcement for debugging
            let defensive_reserve = max_enemy_strength / 6.0; // DEBUG: Much smaller reserves
            let extra_strength = current_strength_f - defensive_reserve;
            
            if (extra_strength > 0.05) { // DEBUG: Much lower threshold
                // DEBUG: Very aggressive attack threshold to encourage expansion
                let attack_threshold = min_enemy_strength * 0.5; // DEBUG: Much easier to attack
                
                if (extra_strength > attack_threshold && min_enemy_direction < 6u) {
                    // Attack weakest enemy - prioritize expansion
                    new_action = min_enemy_direction | (31u << 3u);
                } else if (max_friendly_need > calculated_need * 2.0 && max_need_direction < 6u) {
                    // Only reinforce if friend has MUCH higher need - favor expansion
                    new_action = max_need_direction | (15u << 3u); // Half strength for reinforcement
                } else if (min_enemy_direction < 6u) {
                    // If we can't meet attack threshold, try anyway (very aggressive debug mode)
                    new_action = min_enemy_direction | (15u << 3u); // Weaker attack
                } else {
                    new_action = 0u; // No action
                }
            } else {
                new_action = 0u; // Not enough extra strength
            }
            
            // DEBUG: Disable empire extinction for now to debug other issues
            // Empire extinction: isolated cells have chance to die
            // if (friendly_neighbors == 0u && rng_range(global_id.x, global_id.y, frame_data, 100u, 20u) == 0u) {
            //     new_empire_id = 0u;
            //     new_strength = u32(altitude * 50.0); // Revert to terrain strength
            //     new_need = 0u;
            //     new_action = 0u;
            // }
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
