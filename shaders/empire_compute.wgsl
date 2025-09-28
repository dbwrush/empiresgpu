@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<uniform> frame_data: u32;

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
    
    var new_empire_id = empire_id;
    var new_strength = strength;
    var new_need = need;
    var new_action = action;
    
    // If this cell belongs to an empire (empire_id != 0), it will try to spread
    if (empire_id != 0u) {
        // Try to find a valid target (not same empire)
        var chosen_direction = 6u; // Invalid direction initially
        var attempts = 0u;
        
        // Try up to 6 times to find a valid target
        while (attempts < 6u && chosen_direction >= 6u) {
            let test_direction = rng_range(global_id.x, global_id.y, frame_data, attempts, 6u);
            let target_offset = get_hex_neighbor_offset(test_direction, pos.y);
            let target_pos = pos + target_offset;
            let target_cell = get_cell(target_pos);
            let target_empire = float_to_u8(target_cell.r);
            
            // Only attack if target is different empire (including unclaimed = 0)
            if (target_empire != empire_id) {
                chosen_direction = test_direction;
            }
            attempts++;
        }
        
        // If we found a valid target, set the action
        if (chosen_direction < 6u) {
            // Set action: first 3 bits = direction, remaining 5 bits = all 1s (31 = 0b11111)
            new_action = chosen_direction | (31u << 3u); // direction + strength allocation bits
        } else {
            // No valid targets, don't attack
            new_action = 0u;
        }
    } else {
        // Check if any neighboring empire cells are trying to claim this unclaimed cell
        var claiming_empire = 0u;
        
        // Check all 6 hex directions for incoming attacks
        for (var dir = 0u; dir < 6u; dir++) {
            let neighbor_offset = get_hex_neighbor_offset(dir, pos.y);
            let neighbor_pos = pos + neighbor_offset;
            let neighbor_cell = get_cell(neighbor_pos);
            let neighbor_empire = float_to_u8(neighbor_cell.r);
            let neighbor_action = float_to_u8(neighbor_cell.a);
            
            // If neighbor has an empire and is acting (action != 0)
            if (neighbor_empire != 0u && neighbor_action != 0u) {
                let neighbor_direction = neighbor_action & 7u; // Extract first 3 bits (now 0-5)
                
                // Calculate where the neighbor is attacking using hex grid
                let neighbor_target_offset = get_hex_neighbor_offset(neighbor_direction, neighbor_pos.y);
                let neighbor_target = neighbor_pos + neighbor_target_offset;
                
                // If the neighbor is attacking this cell, claim it
                if (neighbor_target.x == pos.x && neighbor_target.y == pos.y) {
                    claiming_empire = neighbor_empire;
                    break;
                }
            }
        }
        
        // If an empire is claiming this cell, convert it
        if (claiming_empire != 0u) {
            new_empire_id = claiming_empire;
            new_strength = 128u; // Default strength for new cells
            new_need = 64u;     // Default need for new cells
            
            // Newly claimed cells should also try to attack in the same frame
            // Try to find a valid target (not same empire)
            var chosen_direction = 6u; // Invalid direction initially
            var attempts = 0u;
            
            // Try up to 6 times to find a valid target
            while (attempts < 6u && chosen_direction >= 6u) {
                let test_direction = rng_range(global_id.x, global_id.y, frame_data, attempts + 10u, 6u);
                let target_offset = get_hex_neighbor_offset(test_direction, pos.y);
                let target_pos = pos + target_offset;
                let target_cell = get_cell(target_pos);
                let target_empire = float_to_u8(target_cell.r);
                
                // Only attack if target is different empire (including unclaimed = 0)
                if (target_empire != claiming_empire) {
                    chosen_direction = test_direction;
                }
                attempts++;
            }
            
            // If we found a valid target, set the action
            if (chosen_direction < 6u) {
                // Set action: first 3 bits = direction, remaining 5 bits = all 1s (31 = 0b11111)
                new_action = chosen_direction | (31u << 3u); // direction + strength allocation bits
            } else {
                // No valid targets, don't attack
                new_action = 0u;
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
