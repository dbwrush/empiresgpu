@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<uniform> frame_data: u32;

@group(0) @binding(3)
var terrain_texture: texture_2d<f32>;

@group(0) @binding(4)
var empire_params_texture: texture_2d<f32>;

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
// Returns vec4<f32> with (altitude, unused, unused, terrain_penalty) normalized to [0,1]
fn get_terrain(pos: vec2<i32>) -> vec4<f32> {
    let dims = textureDimensions(terrain_texture);
    let wrapped_pos = vec2<i32>(
        (pos.x + i32(dims.x)) % i32(dims.x),
        (pos.y + i32(dims.y)) % i32(dims.y)
    );
    let terrain_data = textureLoad(terrain_texture, wrapped_pos, 0);
    return terrain_data; // altitude, unused, unused, terrain_penalty
}

// Get empire parameters for a specific empire
// Returns: R=diplomacy, G=aggression, B=reserved, A=reserved
fn get_empire_params(empire_id: u32) -> vec4<f32> {
    if (empire_id == 0u || empire_id > 255u) {
        // Invalid empire ID, return neutral values
        return vec4<f32>(0.5, 0.5, 0.0, 1.0);
    }
    
    // Empire parameters texture is organized as a 256x256 grid
    // Row = empire asking, Column = target empire (for diplomacy)
    // For aggression, we use the empire's own row (doesn't matter which column)
    let empire_pos = vec2<i32>(0, i32(empire_id - 1u)); // Use first column for aggression
    return textureLoad(empire_params_texture, empire_pos, 0);
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
    let terrain_penalty = terrain.a; // 0.0 to 1.0 (movement/logistics penalty)
    
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
        
        // PHASE 1: Handle incoming attacks and reinforcements
        var total_attack_strength = 0.0;
        var total_reinforcement_strength = 0.0;
        var attacking_empire = 0u;
        var found_attacker = false;
        
        // Check all 6 hex directions for incoming actions
        for (var dir = 0u; dir < 6u; dir++) {
            let neighbor_offset = get_hex_neighbor_offset(dir, pos.y);
            let neighbor_pos = pos + neighbor_offset;
            let neighbor_cell = get_cell(neighbor_pos);
            let neighbor_empire = float_to_u8(neighbor_cell.r);
            let neighbor_strength = f32(float_to_u8(neighbor_cell.g)) / 255.0;
            let neighbor_action = float_to_u8(neighbor_cell.a);
            
            // If neighbor has a valid action targeting this cell
            if (neighbor_empire != 0u && neighbor_action != 0u) {
                let neighbor_direction = neighbor_action & 7u; // Extract direction (0-5)
                let action_type = (neighbor_action >> 7u) & 3u; // Extract action type (bits 7-8)
                
                // Skip invalid directions (6 = no action, 7 = invalid)
                if (neighbor_direction >= 6u) {
                    continue;
                }
                
                // Calculate where the neighbor is acting
                let neighbor_target_offset = get_hex_neighbor_offset(neighbor_direction, neighbor_pos.y);
                let neighbor_target = neighbor_pos + neighbor_target_offset;
                
                // If the neighbor is acting on this cell
                if (neighbor_target.x == pos.x && neighbor_target.y == pos.y) {
                    if (action_type == 1u) {
                        // Attack action - use encoded attack strength
                        if (neighbor_empire != empire_id) {
                            let attack_strength_bits = (neighbor_action >> 3u) & 15u; // Extract attack strength (bits 3-6)
                            let attack_strength = f32(attack_strength_bits) / 15.0; // Convert back to 0-1 range
                            total_attack_strength += attack_strength;
                            
                            // Remember attacking empire (last one wins if multiple)
                            if (!found_attacker) {
                                attacking_empire = neighbor_empire;
                                found_attacker = true;
                            }
                        }
                    } else if (action_type == 2u && neighbor_empire == empire_id) {
                        // Reinforcement action from same empire
                        let transfer_amount_bits = (neighbor_action >> 3u) & 15u; // Extract amount (bits 3-6)
                        let transfer_amount = f32(transfer_amount_bits) / 15.0; // Convert back to 0-1 range
                        total_reinforcement_strength += transfer_amount;
                    }
                }
            }
        }
        
        // Apply reinforcements and combat resolution
        var current_strength_f = f32(new_strength) / 255.0;
        current_strength_f += total_reinforcement_strength;
        
        // Combat resolution with stronger defensive advantage
        if (total_attack_strength > 0.0) {
            let damage = total_attack_strength / 3.0; // 3:1 defender advantage - harder to conquer
            
            if (damage >= current_strength_f) {
                // Cell is conquered - but only if attack was significantly stronger
                new_empire_id = attacking_empire;
                new_strength = u32(clamp((damage - current_strength_f + 0.1) * 255.0, 30.0, 255.0));
                new_need = 64u; // Reset need for conquered cell
            } else {
                // Cell survives - reduce damage taken
                current_strength_f -= damage;
                new_strength = u32(clamp(current_strength_f * 255.0, 0.0, 255.0));
            }
        } else {
            // No combat, just apply reinforcements
            new_strength = u32(clamp(current_strength_f * 255.0, 0.0, 255.0));
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
            
            // CRITICAL: Terrain-aware need propagation to discourage mountain border gore
            // High terrain penalty (mountains) severely reduces need propagation efficiency
            // This encourages empires to prefer coastal/lowland reinforcement routes
            let base_propagation_efficiency = 0.95; // High base efficiency in good terrain
            
            // Exponential decay based on terrain penalty - mountains block need signals effectively
            // Ocean/plains (penalty=0.05-0.15): 90-85% propagation efficiency  
            // Hills (penalty=0.45): 65% propagation efficiency
            // Mountains (penalty=0.95): 15% propagation efficiency (severe blocking)
            let propagation_efficiency = base_propagation_efficiency * (1.0 - terrain_penalty * 0.85);
            
            let propagated_need = max_friendly_need * propagation_efficiency;
            calculated_need = calculated_need + propagated_need;
            
            // Update need value
            new_need = u32(clamp(calculated_need * 255.0, 0.0, 255.0));
            
            // Apply severe terrain penalties to discourage mountain occupation
            var current_strength_f = f32(new_strength) / 255.0;
            
            // Harsh logistics penalty in difficult terrain - mountains are expensive to hold
            // Ocean/plains (penalty=0.05-0.15): 95-85% strength retention
            // Hills (penalty=0.45): 55% strength retention  
            // Mountains (penalty=0.95): 5% strength retention (unsustainable)
            let logistics_penalty = 1.0 - (terrain_penalty * 0.95);
            let penalized_strength = current_strength_f * logistics_penalty;
            
            // Strength generation heavily biased towards low altitude (coastal preference)
            // Mountains generate almost no strength, encouraging abandonment
            let base_generation = terrain_factor / 20.0;
            let terrain_generation_penalty = 1.0 - (terrain_penalty * 0.9); // 90% reduction in mountains
            let strength_generation = base_generation * terrain_generation_penalty;
            
            let final_strength = penalized_strength + strength_generation;
            current_strength_f = final_strength;
            
            // Smart tactical decision making with proper threat assessment
            // Calculate total defensive requirements against ALL potential threats
            var total_defensive_requirement = 0.0;
            
            // Sum up all neighboring enemy threats (need 3:1 advantage to defend successfully)
            for (var dir = 0u; dir < 6u; dir++) {
                let neighbor_offset = get_hex_neighbor_offset(dir, pos.y);
                let neighbor_pos = pos + neighbor_offset;
                let neighbor_cell = get_cell(neighbor_pos);
                let neighbor_empire = float_to_u8(neighbor_cell.r);
                let neighbor_strength = f32(float_to_u8(neighbor_cell.g)) / 255.0;
                
                // Check if neighbor is water
                let neighbor_terrain = get_terrain(neighbor_pos);
                let neighbor_is_water = neighbor_terrain.r < ocean_cutoff;
                
                if (!neighbor_is_water && neighbor_empire != 0u && neighbor_empire != new_empire_id) {
                    // Enemy cell - need to be able to defend against potential attacks
                    // Account for the 3:1 advantage attackers need, so we need 1/3 strength to defend
                    total_defensive_requirement += neighbor_strength / 3.0;
                }
            }
            
            // Add terrain-based defensive buffer (mountains require more reserves)
            let base_buffer = 0.05; // 5% base safety buffer
            let terrain_buffer_multiplier = 1.0 + (terrain_penalty * terrain_penalty * 2.0);
            let defensive_buffer = total_defensive_requirement * (base_buffer * terrain_buffer_multiplier);
            let total_required_defense = total_defensive_requirement + defensive_buffer;
            
            // Available strength for offensive operations after ensuring defense
            let available_for_action = current_strength_f - total_required_defense;
            
            if (available_for_action > 0.05) { // Need meaningful strength to act
                // Get aggression parameter for this empire
                let empire_params = get_empire_params(new_empire_id);
                let aggression = empire_params.g; // Green channel contains aggression
                
                // Calculate attack threshold based on aggression AND terrain difficulty
                let base_multiplier = 3.0;  // Conservative 3:1 advantage requirement
                let aggressive_multiplier = 0.5;  // Even aggressive empires need 0.5:1 minimum
                let aggression_threshold = mix(base_multiplier, aggressive_multiplier, aggression);
                
                // Terrain penalty makes attacks much harder in mountains
                let terrain_attack_penalty = 1.0 + (terrain_penalty * terrain_penalty * 7.0);
                let final_attack_threshold = aggression_threshold * terrain_attack_penalty;
                
                // Check if we can safely attack the weakest enemy without compromising defense
                if (min_enemy_direction < 6u) {
                    let required_attack_strength = min_enemy_strength * final_attack_threshold;
                    let strength_after_attack = current_strength_f - required_attack_strength;
                    
                    // Verify we can still defend after the attack
                    if (available_for_action >= required_attack_strength && strength_after_attack >= total_required_defense) {
                        // Safe to attack! Calculate attack strength as 4-bit value (0-15)
                        let scaled_attack_strength = u32(clamp(required_attack_strength * 15.0, 1.0, 15.0));
                        new_action = min_enemy_direction | (1u << 7u) | (scaled_attack_strength << 3u);
                    } else {
                        // Can't safely attack, check for reinforcement opportunities
                        new_action = 6u; // Invalid direction = no action
                    }
                } else {
                    // No enemies to attack, consider reinforcement
                    new_action = 6u; // Invalid direction = no action initially
                }
                
                // If no safe attack was possible, consider reinforcement 
                if (new_action == 6u && max_need_direction < 6u && max_friendly_need > (f32(need) / 255.0)) {
                    let my_need = f32(need) / 255.0;
                    let need_ratio = max_friendly_need / max(my_need, 0.01); // Avoid division by zero
                    
                    // Only reinforce if the need difference is significant enough to justify the risk
                    if (need_ratio > 1.2) { // At least 20% more need
                        // Calculate safe transfer amount that won't compromise our defense
                        let base_transfer_rate = 0.2; // Conservative transfer rate
                        let need_factor = 1.0 - (my_need * 0.8); // Keep more if we also have high need
                        
                        // Terrain penalty for reinforcement efficiency
                        let transfer_efficiency = 1.0 - (terrain_penalty * 0.95);
                        
                        let safe_transfer_rate = base_transfer_rate * need_factor * transfer_efficiency;
                        let proposed_transfer = available_for_action * clamp(safe_transfer_rate, 0.05, 0.4);
                        
                        // Verify transfer won't compromise defense
                        if (current_strength_f - proposed_transfer >= total_required_defense) {
                            // Safe to reinforce
                            let scaled_transfer = u32(clamp(proposed_transfer * 15.0, 1.0, 15.0));
                            new_action = max_need_direction | (2u << 7u) | (scaled_transfer << 3u);
                            current_strength_f -= proposed_transfer;
                        } else {
                            // Not safe to reinforce
                            new_action = 6u; // No action
                        }
                    } else {
                        // Need difference not significant enough
                        new_action = 6u; // No action
                    }
                }
            } else {
                // Not enough available strength for any action
                new_action = 6u; // No action (invalid direction indicates no action)
            }
            
            // Update final strength
            new_strength = u32(clamp(current_strength_f * 255.0, 0.0, 255.0));
            
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
