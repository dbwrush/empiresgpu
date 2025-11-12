@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba16unorm, write>;

@group(0) @binding(2)
var<uniform> frame_data: u32;

@group(0) @binding(3)
var terrain_texture: texture_2d<f32>;

@group(0) @binding(4)
var empire_params_texture: texture_2d<f32>;

@group(0) @binding(5)
var aux_input_texture: texture_2d<f32>; // RGBA16Unorm auxiliary input texture (age data in red channel)

@group(0) @binding(6)
var aux_output_texture: texture_storage_2d<rgba16unorm, write>; // RGBA16Unorm auxiliary output texture

// Diplomacy system: Real-time atomic counters for empire interactions
// 2048x2048 matrix, 5 u32 counters per empire pair
struct DiplomacyCounters {
    attack_count: atomic<u32>,      // Number of attacks from A to B
    attack_strength: atomic<u32>,   // Total attack strength (accumulated)
    reinforce_count: atomic<u32>,   // Number of reinforcements from A to B
    reinforce_strength: atomic<u32>, // Total reinforce strength (accumulated)
    pressure_count: atomic<u32>,    // Border pressure events (stronger cell reports)
}

@group(0) @binding(7)
var<storage, read_write> diplomacy_counters: array<DiplomacyCounters>; // 2048*2048 = 4194304 entries

@group(0) @binding(8)
var<storage, read> diplomacy_relations: array<f32>; // 2048*2048 relations matrix (-1.0 to 1.0)

@group(0) @binding(9)
var<storage, read> boat_landings: array<u32>; // Boat attack buffer: upper 16 bits = empire_id, lower 16 bits = strength

@group(0) @binding(10)
var<storage, read_write> boat_spawns: array<atomic<u32>>; // Boat spawn buffer: upper 16 bits = empire_id, lower 16 bits = direction

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

// Get diplomatic relation between two empires
// Returns: -1.0 = hostile/war, 0.0 = neutral, 1.0 = allied
fn get_diplomacy(empire_a: u32, empire_b: u32) -> f32 {
    if (empire_a == 0u || empire_b == 0u || empire_a > 2047u || empire_b > 2047u) {
        return 0.0; // Neutral for invalid empires
    }
    if (empire_a == empire_b) {
        return 1.0; // Self-relation is always maximum
    }
    
    // Access symmetric matrix (relations are the same both ways)
    // Use (min, max) ordering to ensure we always access the same cell
    let a_idx = empire_a - 1u;
    let b_idx = empire_b - 1u;
    let row = min(a_idx, b_idx);
    let col = max(a_idx, b_idx);
    let idx = row * 2048u + col;
    return diplomacy_relations[idx];
}

// Record a diplomatic event (attack or reinforcement to another empire)
// sample_rate: 1.0 = always record, 0.1 = record 10% of the time (for expected events)
fn record_diplomatic_event(actor_empire: u32, target_empire: u32, is_attack: bool, strength: u32, sample_rate: f32, random_seed: u32) {
    if (actor_empire == 0u || target_empire == 0u || actor_empire > 2047u || target_empire > 2047u) {
        return; // Invalid empires
    }
    if (actor_empire == target_empire) {
        return; // Don't record same-empire events
    }
    
    // Probabilistic sampling for expected events (reduces CPU load)
    if (sample_rate < 1.0) {
        let sample_threshold = u32(sample_rate * 100.0);
        let random_value = rng_range(actor_empire, target_empire, random_seed, 0u, 100u);
        if (random_value >= sample_threshold) {
            return; // Skip this event (sampled out)
        }
    }
    
    // Access symmetric matrix using (min, max) ordering
    let a_idx = actor_empire - 1u;
    let t_idx = target_empire - 1u;
    let row = min(a_idx, t_idx);
    let col = max(a_idx, t_idx);
    let idx = row * 2048u + col;
    
    if (is_attack) {
        atomicAdd(&diplomacy_counters[idx].attack_count, 1u);
        atomicAdd(&diplomacy_counters[idx].attack_strength, strength);
    } else {
        atomicAdd(&diplomacy_counters[idx].reinforce_count, 1u);
        atomicAdd(&diplomacy_counters[idx].reinforce_strength, strength);
    }
}

// Get auxiliary data (age, boat_need) at position with wrapping - returns vec2<f32>
fn get_aux_data_full(pos: vec2<i32>) -> vec2<f32> {
    let dims = textureDimensions(aux_input_texture);
    let wrapped_pos = vec2<i32>(
        (pos.x + i32(dims.x)) % i32(dims.x),
        (pos.y + i32(dims.y)) % i32(dims.y)
    );
    let aux_data = textureLoad(aux_input_texture, wrapped_pos, 0);
    return vec2<f32>(aux_data.r, aux_data.g); // age in red, boat_need in green
}

// Get auxiliary data (age) at position with wrapping - returns normalized float 0.0-1.0
fn get_aux_data(pos: vec2<i32>) -> f32 {
    return get_aux_data_full(pos).x;
}

// Convert float [0,1] to u16 [0,65535]
fn float_to_u16(val: f32) -> u32 {
    return u32(clamp(val * 65535.0, 0.0, 65535.0));
}

// Convert u16 [0,65535] to float [0,1]
fn u16_to_float(val: u32) -> f32 {
    return f32(val) / 65535.0;
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
    let empire_id = float_to_u16(current_cell.r);
    let strength = float_to_u16(current_cell.g);
    let need = float_to_u16(current_cell.b);
    let action = float_to_u16(current_cell.a);
    
    // Read current auxiliary data (age and boat need)
    let aux_data = get_aux_data_full(pos);
    let current_age = aux_data.x;
    var boat_need = aux_data.y; // 0.0 to 1.0, represents desire to send boats
    
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
    var new_age = current_age; // Default: preserve current age
    
    // If this is water, clear any empire presence
    if (is_water) {
        new_empire_id = 0u;
        new_strength = 0u;
        new_need = 0u;
        new_action = 0u;
        new_age = 0.0;
    } else {
        // Handle unclaimed land cells - set initial strength based on terrain
        if (empire_id == 0u && strength == 0u) {
            // Convert terrain altitude to initial strength (16-bit range)
            // DEBUG: Much lower initial strength to allow expansion testing
            new_strength = u32(altitude * 3000.0); // DEBUG: Much weaker initial strength (scaled for 16-bit)
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
            let neighbor_empire = float_to_u16(neighbor_cell.r);
            let neighbor_strength = f32(float_to_u16(neighbor_cell.g)) / 65535.0;
            let neighbor_action = float_to_u16(neighbor_cell.a);
            
            // If neighbor has a valid action targeting this cell
            if (neighbor_empire != 0u && neighbor_action != 0u) {
                let neighbor_direction = neighbor_action & 7u; // Extract direction (bits 0-2)
                let amount_12bit = (neighbor_action >> 3u) & 4095u; // Extract amount (bits 3-14, 12-bit)
                let is_reinforce = (neighbor_action >> 15u) & 1u; // Extract action type (bit 15)
                
                // Skip if no action (amount = 0)
                if (amount_12bit == 0u) {
                    continue;
                }
                
                // Skip invalid directions (6-7)
                if (neighbor_direction >= 6u) {
                    continue;
                }
                
                // Calculate where the neighbor is acting
                let neighbor_target_offset = get_hex_neighbor_offset(neighbor_direction, neighbor_pos.y);
                let neighbor_target = neighbor_pos + neighbor_target_offset;
                
                // If the neighbor is acting on this cell
                if (neighbor_target.x == pos.x && neighbor_target.y == pos.y) {
                    // Convert 12-bit amount back to 0.0-1.0 range
                    let action_amount = f32(amount_12bit) / 4095.0;
                    
                    if (is_reinforce == 0u) {
                        // Attack action
                        if (neighbor_empire != empire_id) {
                            // Apply defender's terrain penalty - attacking into mountains is harder
                            // Mountains provide strong defensive bonus (quadratic penalty to attacker)
                            let defender_terrain_penalty = 1.0 - (terrain_penalty * terrain_penalty * 0.8);
                            let effective_attack = action_amount * defender_terrain_penalty;
                            
                            total_attack_strength += effective_attack;
                            
                            // Remember attacking empire (last one wins if multiple)
                            if (!found_attacker) {
                                attacking_empire = neighbor_empire;
                                found_attacker = true;
                            }
                        }
                    } else if (is_reinforce == 1u && neighbor_empire == empire_id) {
                        // Reinforcement action from same empire
                        // Apply receiver's terrain penalty - but much reduced to boost reinforcement efficiency
                        let receiver_terrain_penalty = 1.0 - (terrain_penalty * 0.1);  // Reduced from 0.7
                        let effective_reinforcement = action_amount * receiver_terrain_penalty;
                        
                        total_reinforcement_strength += effective_reinforcement;
                    }
                }
            }
        }
        
        // BOAT LANDINGS: Check if any boats are landing at this cell
        let cell_index = u32(pos.y) * dims.x + u32(pos.x);
        let boat_landing = boat_landings[cell_index];
        if (boat_landing != 0u) {
            let boat_empire = (boat_landing >> 16u) & 0xFFFFu;  // Upper 16 bits
            let boat_strength_u16 = boat_landing & 0xFFFFu;      // Lower 16 bits
            let boat_strength_f = f32(boat_strength_u16) / 65535.0;
            
            if (boat_empire != 0u) {
                // Determine action based on empire relationships
                let relation = get_diplomacy(boat_empire, empire_id);
                var is_attack = false;
                var should_act = true;
                
                if (empire_id == 0u) {
                    // Unclaimed territory - always conquer
                    is_attack = true;
                } else if (boat_empire == empire_id) {
                    // Same empire - always reinforce
                    is_attack = false;
                } else if (relation >= 0.3) {
                    // Allied territory (relation >= 0.3) - reinforce
                    is_attack = false;
                } else if (relation <= -0.3) {
                    // Enemy territory (relation <= -0.3) - attack
                    is_attack = true;
                } else {
                    // Neutral territory (-0.3 < relation < 0.3)
                    // Decision based on boat empire's aggression
                    let boat_params = get_empire_params(boat_empire);
                    let aggression = boat_params.r; // Red channel = aggression
                    
                    // Random roll against aggression threshold
                    let roll = rng_range(boat_empire, empire_id, frame_data, 12345u, 100u);
                    let aggression_threshold = u32(aggression * 100.0);
                    is_attack = roll < aggression_threshold;
                }
                
                if (is_attack) {
                    // ATTACK: Apply defender's terrain penalty to boat attack
                    let defender_terrain_penalty = 1.0 - (terrain_penalty * terrain_penalty * 0.8);
                    let effective_boat_attack = boat_strength_f * defender_terrain_penalty;
                    
                    total_attack_strength += effective_boat_attack;
                    
                    // Boat empire becomes potential attacker
                    if (!found_attacker) {
                        attacking_empire = boat_empire;
                        found_attacker = true;
                    }
                    
                    // BOAT NEED TRACKING: Cell is being attacked by boat - increase boat need
                    // This signals that ocean routes are active and cells should respond with their own boats
                    if (empire_id != 0u && empire_id != boat_empire) {
                        boat_need = min(boat_need + 0.5, 1.0); // Significant boost when attacked by boat
                    }
                    
                    // Record attack event for diplomacy (harms relations)
                    if (empire_id != 0u) {
                        record_diplomatic_event(boat_empire, empire_id, true, boat_strength_u16, 1.0, frame_data);
                    }
                } else {
                    // REINFORCE: Add boat strength to cell (friendly action)
                    if (empire_id != 0u) {
                        total_reinforcement_strength += boat_strength_f;
                        
                        // Record reinforcement for diplomacy (improves relations)
                        record_diplomatic_event(boat_empire, empire_id, false, boat_strength_u16, 1.0, frame_data);
                    }
                }
            }
        }
        
        // Apply reinforcements and combat resolution
        var current_strength_f = f32(new_strength) / 65535.0;
        current_strength_f += total_reinforcement_strength;
        
        // New attrition-based combat resolution
        if (total_attack_strength > 0.0) {
            // Calculate casualties: attackers inflict damage at 3:1 ratio (defender advantage)
            let casualties = total_attack_strength / 3.0;
            current_strength_f -= casualties;
            
            // Check if this results in conquest
            // Defender is conquered if their strength drops to 0 or below
            if (current_strength_f <= 0.0) {
                // Full conquest: attacker wins and occupies with remaining strength
                new_empire_id = attacking_empire;
                // Conqueror gets the excess attack strength after overcoming the defender
                let remaining_attack_strength = total_attack_strength - (f32(strength) / 65535.0 * 3.0);
                new_strength = u32(clamp(remaining_attack_strength * 65535.0, 1000.0, 65535.0)); // Minimum 1000 strength
                new_need = 3000u; // Reset need for newly conquered cell
                new_age = 0.0; // Reset age for newly conquered cell
                
                // BOAT NEED ON CONQUEST: Count ocean neighbors (like Bevy reference)
                // Set boat_need based on coastline accessibility (6 - ocean_neighbors)
                // More ocean = higher boat_need to signal naval capability
                var ocean_neighbors = 0u;
                for (var dir: u32 = 0u; dir < 6u; dir++) {
                    let offset = get_hex_neighbor_offset(dir, pos.y);
                    let neighbor_pos = pos + offset;
                    if (neighbor_pos.x >= 0 && neighbor_pos.x < i32(dims.x) && 
                        neighbor_pos.y >= 0 && neighbor_pos.y < i32(dims.y)) {
                        let neighbor_terrain = get_terrain(neighbor_pos);
                        let neighbor_altitude = neighbor_terrain.r;
                        if (neighbor_altitude < ocean_cutoff) {
                            ocean_neighbors++;
                        }
                    }
                }
                // Reference: boat_need = 6.0 - coastlines.len() (fewer land neighbors = more coastal)
                // We invert: more ocean neighbors = more boat_need
                // Scale to 0.0-2.0 range (0 ocean = 0.0, 6 ocean = 2.0)
                boat_need = f32(ocean_neighbors) / 3.0; // 0 to 2.0
            } else {
                // Attrition only: defender survives but is weakened
                current_strength_f = max(current_strength_f, 0.01); // Minimum survival strength
                new_strength = u32(clamp(current_strength_f * 65535.0, 1.0, 65535.0));
                // Empire ID and need remain unchanged
            }
        } else {
            // No combat, just apply reinforcements
            new_strength = u32(clamp(current_strength_f * 65535.0, 0.0, 65535.0));
        }
        
        // PHASE 1.5: Peaceful territory transfers to reduce border gore
        // If surrounded by allied neighbors, small chance of peaceful transfer
        if (new_empire_id != 0u && total_attack_strength == 0.0) {
            // First pass: find the ally with the most neighbors (max 6 comparisons)
            var count_0 = 0u; var empire_0 = 0u;
            var count_1 = 0u; var empire_1 = 0u;
            var count_2 = 0u; var empire_2 = 0u;
            var count_3 = 0u; var empire_3 = 0u;
            var count_4 = 0u; var empire_4 = 0u;
            var count_5 = 0u; var empire_5 = 0u;
            
            for (var dir = 0u; dir < 6u; dir++) {
                let neighbor_offset = get_hex_neighbor_offset(dir, pos.y);
                let neighbor_pos = pos + neighbor_offset;
                let neighbor_cell = get_cell(neighbor_pos);
                let neighbor_empire = float_to_u16(neighbor_cell.r);
                
                // Check if neighbor is an ally (not self, not unclaimed)
                if (neighbor_empire != 0u && neighbor_empire != new_empire_id) {
                    let relation = get_diplomacy(new_empire_id, neighbor_empire);
                    if (relation >= 0.25) { // Allied threshold
                        // Track up to 6 different allied empires
                        if (empire_0 == 0u) { empire_0 = neighbor_empire; count_0 = 1u; }
                        else if (neighbor_empire == empire_0) { count_0++; }
                        else if (empire_1 == 0u) { empire_1 = neighbor_empire; count_1 = 1u; }
                        else if (neighbor_empire == empire_1) { count_1++; }
                        else if (empire_2 == 0u) { empire_2 = neighbor_empire; count_2 = 1u; }
                        else if (neighbor_empire == empire_2) { count_2++; }
                        else if (empire_3 == 0u) { empire_3 = neighbor_empire; count_3 = 1u; }
                        else if (neighbor_empire == empire_3) { count_3++; }
                        else if (empire_4 == 0u) { empire_4 = neighbor_empire; count_4 = 1u; }
                        else if (neighbor_empire == empire_4) { count_4++; }
                        else if (empire_5 == 0u) { empire_5 = neighbor_empire; count_5 = 1u; }
                        else if (neighbor_empire == empire_5) { count_5++; }
                    }
                }
            }
            
            // Find the maximum count
            var max_count = count_0;
            var dominant_ally = empire_0;
            if (count_1 > max_count) { max_count = count_1; dominant_ally = empire_1; }
            if (count_2 > max_count) { max_count = count_2; dominant_ally = empire_2; }
            if (count_3 > max_count) { max_count = count_3; dominant_ally = empire_3; }
            if (count_4 > max_count) { max_count = count_4; dominant_ally = empire_4; }
            if (count_5 > max_count) { max_count = count_5; dominant_ally = empire_5; }
            
            // Check for peaceful transfer based on neighbor count
            if (max_count >= 4u && dominant_ally != 0u) {
                // Calculate transfer probability based on neighbor count
                var transfer_chance = 0u;
                if (max_count == 4u) {
                    transfer_chance = 1u;   // 1% chance
                } else if (max_count == 5u) {
                    transfer_chance = 5u;   // 5% chance
                } else if (max_count == 6u) {
                    transfer_chance = 10u;  // 10% chance
                }
                
                // Random check for transfer
                let random_roll = rng_range(new_empire_id, dominant_ally, frame_data + u32(pos.x * 1000 + pos.y), 1u, 100u);
                if (random_roll <= transfer_chance) {
                    // Peaceful transfer to dominant ally
                    new_empire_id = dominant_ally;
                    // Keep strength and age, reset need slightly
                    new_need = u32(f32(new_need) * 0.8);
                }
            }
        }
        
        // PHASE 2: Generate strength, calculate need, and make decisions for occupied cells
        if (new_empire_id != 0u) {
            // Calculate need based on enemy pressure and friendly support
            var calculated_need = 0.0;
            var max_enemy_strength = 0.0;
            var min_enemy_strength = 2.0;
            var min_enemy_direction = 6u;
            var min_unclaimed_strength = 2.0;
            var min_unclaimed_direction = 6u;
            var min_neutral_strength = 2.0;
            var min_neutral_direction = 6u;
            var min_neutral_empire = 0u;
            var max_friendly_need = 0.0;
            var max_need_direction = 6u;
            var friendly_neighbors = 0u;
            var enemy_neighbors = 0u;
            
            // Track allied neighbors separately for border need reduction
            var allied_neighbors = 0u;
            
            // Analyze all neighbors for need calculation
            for (var dir = 0u; dir < 6u; dir++) {
                let neighbor_offset = get_hex_neighbor_offset(dir, pos.y);
                let neighbor_pos = pos + neighbor_offset;
                let neighbor_cell = get_cell(neighbor_pos);
                let neighbor_empire = float_to_u16(neighbor_cell.r);
                let neighbor_strength = f32(float_to_u16(neighbor_cell.g)) / 65535.0;
                let neighbor_need = f32(float_to_u16(neighbor_cell.b)) / 65535.0;
                
                // Check if neighbor is water
                let neighbor_terrain = get_terrain(neighbor_pos);
                let neighbor_is_water = neighbor_terrain.r < ocean_cutoff;
                
                if (!neighbor_is_water) {
                    if (neighbor_empire == new_empire_id) {
                        // Friendly neighbor (same empire)
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
                        // Different empire or unclaimed neighbor
                        if (neighbor_empire == 0u) {
                            // Unclaimed territory - track as expansion target
                            // No friction with unclaimed (ocean) borders
                            enemy_neighbors++;
                            calculated_need += neighbor_strength; // Add pressure from unclaimed
                            calculated_need -= 0.9 * neighbor_strength; // But easier to expand
                            
                            // Track weakest unclaimed cell for expansion
                            if (neighbor_strength < min_unclaimed_strength) {
                                min_unclaimed_strength = neighbor_strength;
                                min_unclaimed_direction = dir;
                            } else if (neighbor_strength == min_unclaimed_strength) {
                                if (rng_range(global_id.x, global_id.y, frame_data, dir * 11u, 2u) == 0u) {
                                    min_unclaimed_direction = dir;
                                }
                            }
                        } else {
                            // Another empire - check diplomatic relations
                            let relation = get_diplomacy(new_empire_id, neighbor_empire);
                            
                            if (relation > 0.3) {
                                // ALLIED: No defensive need generated - treat as friendly border
                                // Need propagates FROM allies to guide reinforcements to their fights
                                allied_neighbors++;
                                
                                // Allow need to propagate FROM allies (with small 15% penalty for cross-border coordination)
                                // This allows reinforcements to flow freely through allied territory to reach the actual fight
                                let ally_need_contribution = neighbor_need * 0.85;
                                if (ally_need_contribution > max_friendly_need) {
                                    max_friendly_need = ally_need_contribution;
                                    max_need_direction = dir;
                                }
                                
                                // NO defensive need added - allied borders should be transparent to need flow
                            } else if (relation >= -0.3) {
                                // NEUTRAL: Need scales with tension level
                                // relation = -0.3 to 0.3: from hostile threshold to allied threshold
                                // Normalized tension: 0.0 (very friendly) to 1.0 (hostile threshold)
                                let tension = (0.3 - relation) / 0.6; // Range: 0.0 to 1.0
                                
                                // Need pressure scales from 0.1x (friendly neutral) to 1.0x (tense neutral)
                                let pressure_multiplier = 0.1 + (tension * 0.9);
                                calculated_need += neighbor_strength * pressure_multiplier;
                                
                                enemy_neighbors++;
                            } else {
                                // HOSTILE: Full pressure (current behavior)
                                enemy_neighbors++;
                                calculated_need += neighbor_strength;
                            }
                        }
                        
                        // Track enemy strength for attack decisions
                        // Hostile threshold varies by empire aggression personality
                        let relation = get_diplomacy(new_empire_id, neighbor_empire);
                        
                        // Get personality for hostile threshold adjustment
                        let empire_params = get_empire_params(new_empire_id);
                        let aggression = empire_params.g;
                        
                        // Hostile threshold varies by aggression:
                        // Low aggression (0.0): attack at -0.3 (very hostile)
                        // Medium aggression (0.5): attack at -0.2
                        // High aggression (1.0): attack at -0.1 (slightly negative)
                        let hostile_threshold = mix(-0.3, -0.1, aggression);
                        
                        if (relation < hostile_threshold) {
                            // This is a HOSTILE neighbor - valid attack target
                            if (neighbor_strength > max_enemy_strength) {
                                max_enemy_strength = neighbor_strength;
                            }
                            if (neighbor_strength < min_enemy_strength) {
                                min_enemy_strength = neighbor_strength;
                                min_enemy_direction = dir;
                            } else if (neighbor_strength == min_enemy_strength) {
                                // If strengths are equal, randomly choose between them to avoid directional bias
                                if (rng_range(global_id.x, global_id.y, frame_data, dir * 10u, 2u) == 0u) {
                                    min_enemy_direction = dir;
                                }
                            }
                        } else if (relation >= hostile_threshold && relation <= 0.3) {
                            // This is a NEUTRAL neighbor - potential target for aggressive empires
                            if (neighbor_strength < min_neutral_strength) {
                                min_neutral_strength = neighbor_strength;
                                min_neutral_direction = dir;
                                min_neutral_empire = neighbor_empire;
                            } else if (neighbor_strength == min_neutral_strength) {
                                // If strengths are equal, randomly choose between them
                                if (rng_range(global_id.x, global_id.y, frame_data, dir * 11u, 2u) == 0u) {
                                    min_neutral_direction = dir;
                                    min_neutral_empire = neighbor_empire;
                                }
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
            
                        // GRADIENT-BASED need propagation with AGGRESSIVE terrain penalties
            // Mountains should severely block need propagation, forcing routes around them
            
            // Distance cost per cell (normalized 0-1 space)
            // We want mountains to be 50-100x worse than coastal areas
            let base_distance_cost = 0.002; // Base cost per hop in good terrain (500 cell range)
            
            // Exponential terrain scaling: mountains should nearly block propagation
            // terrain_penalty ranges from ~0.05 (ocean/coast) to ~0.95 (mountains)
            // Use exponential function: 2^(penalty * 6) gives ~1.4x at coast, ~64x at mountains
            let terrain_exponential = pow(2.0, terrain_penalty * 6.0); // 1.4x to 64x multiplier
            let distance_cost = base_distance_cost * terrain_exponential;
            
            // For extreme mountains (penalty > 0.8), add multiplicative decay as well
            var propagated_need = max(max_friendly_need - distance_cost, 0.0);
            if (terrain_penalty > 0.8) {
                // Severe terrain: combine linear decay with multiplicative decay
                let multiplicative_decay = 1.0 - (terrain_penalty * 0.5); // 60% to 10% retention
                propagated_need = propagated_need * multiplicative_decay;
            }
            
            calculated_need = calculated_need + propagated_need;
            
            // Update need value (16-bit range now)
            new_need = u32(clamp(calculated_need * 65535.0, 0.0, 65535.0));
            
            // Apply severe terrain penalties to discourage mountain occupation
            var current_strength_f = f32(new_strength) / 65535.0;
            
            // Harsh logistics penalty in difficult terrain - mountains are expensive to hold
            // Ocean/plains (penalty=0.05-0.15): 95-85% strength retention
            // Hills (penalty=0.45): 55% strength retention  
            // Mountains (penalty=0.95): 5% strength retention (unsustainable)
            let logistics_penalty = 1.0 - (terrain_penalty * 0.95);
            let penalized_strength = current_strength_f * logistics_penalty;
            
            // Strength generation heavily biased towards low altitude (coastal preference)
            // Mountains generate almost no strength, encouraging abandonment
            // INCREASED from /20.0 to /15.0 to break stalemates faster
            let base_generation = terrain_factor / 15.0;
            let terrain_generation_penalty = 1.0 - (terrain_penalty * 0.9); // 90% reduction in mountains
            let strength_generation = base_generation * terrain_generation_penalty;
            
            let final_strength = penalized_strength + strength_generation;
            current_strength_f = final_strength;
            
            // Smart tactical decision making - defend against strongest single enemy only
            // Only the strongest enemy can realistically attack us in one turn
            
            // Find the strongest enemy neighbor (already calculated above as max_enemy_strength)
            // We need MUCH LESS than 1/3 defense - be more aggressive, break stalemates
            let required_defense = max_enemy_strength * 0.15; // Only keep 15% of enemy strength as defense
            
            // Add small terrain-based defensive buffer for mountains
            let base_buffer = 0.01; // 1% base safety buffer (reduced from 2%)
            let terrain_buffer_multiplier = 1.0 + (terrain_penalty * terrain_penalty * 0.5); // Reduced
            let defensive_buffer = required_defense * (base_buffer * terrain_buffer_multiplier);
            let total_required_defense = required_defense + defensive_buffer;
            
            // Available strength for offensive operations after ensuring defense
            let available_for_action = current_strength_f - total_required_defense;
            
            if (available_for_action > 0.01) { // Need some strength to act (lowered threshold)
                // Get aggression parameter for this empire
                let empire_params = get_empire_params(new_empire_id);
                let aggression = empire_params.g; // Green channel contains aggression (0.0 to 1.0)
                
                // Calculate attack ratio tolerance based on aggression
                // aggression = 0.0: only attack with 3:1 advantage (guaranteed victory)
                // aggression = 1.0: attack with any available strength (1:1 or worse ratios)
                // REDUCED requirements to be more aggressive
                let conservative_multiplier = 2.0;  // Need 2:1 advantage (reduced from 3:1)
                let aggressive_multiplier = 0.1;    // Willing to attack with terrible odds
                let attack_ratio_requirement = mix(conservative_multiplier, aggressive_multiplier, aggression);
                
                // Terrain penalty still makes attacks harder in mountains (but reduced)
                let terrain_attack_penalty = 1.0 + (terrain_penalty * terrain_penalty * 2.0); // Reduced from 4.0
                let final_attack_ratio = attack_ratio_requirement * terrain_attack_penalty;
                
                // DECISION TREE: Attack vs Reinforce
                // Priority 1: Attack hostile enemies if we have them
                // Priority 2: Expand into unclaimed territory
                // Priority 3: Reinforce if we can't attack
                
                var should_attack = false;
                var attack_strength = 0.0;
                var attack_target_direction = 6u;
                var target_strength = 0.0;
                
                // First priority: Attack hostile neighbors
                if (min_enemy_direction < 6u) {
                    attack_target_direction = min_enemy_direction;
                    target_strength = min_enemy_strength;
                    
                    // Get target empire for diplomacy modifier
                    let target_offset = get_hex_neighbor_offset(min_enemy_direction, pos.y);
                    let target_pos = pos + target_offset;
                    let target_cell = get_cell(target_pos);
                    let target_empire = float_to_u16(target_cell.r);
                    
                    var diplomacy_attack_modifier = 1.0;
                    if (target_empire != 0u && target_empire != new_empire_id) {
                        let relation = get_diplomacy(new_empire_id, target_empire);
                        // relation: -1.0 (hostile) to 1.0 (allied)
                        // For hostile relations, modify attack strength
                        // More hostile = lower threshold (easier to attack)
                        // More aggressive = more willing to commit strength
                        let base_modifier = 1.0 - (relation * 0.7); // Range: 0.3 to 1.7
                        
                        // Aggression makes empires commit more strength at earlier hostility levels
                        // High aggression: willing to attack even at -0.1 relation with full strength
                        // Low aggression: cautious even at -1.0 relation
                        let aggression_boost = aggression * 0.5; // 0.0 to 0.5 bonus
                        diplomacy_attack_modifier = base_modifier + aggression_boost;
                    }
                    
                    let ideal_attack_strength = target_strength * final_attack_ratio * diplomacy_attack_modifier;
                    
                    // If we're MUCH stronger than the enemy, commit ALL available strength (break stalemates!)
                    let strength_ratio = available_for_action / max(target_strength, 0.01);
                    var final_attack_strength = min(available_for_action, ideal_attack_strength);
                    if (strength_ratio > 3.0) {
                        // We're overwhelming - commit everything to finish them
                        final_attack_strength = available_for_action * 0.95;
                    }
                    attack_strength = final_attack_strength;
                    
                    // Attack if we have meaningful strength OR enemy is extremely weak
                    let min_attack_threshold = 1.0 / 4095.0;
                    if (attack_strength > min_attack_threshold || target_strength < 0.01) {
                        should_attack = true;
                    }
                } else if (min_unclaimed_direction < 6u) {
                    // Second priority: Expand into unclaimed territory
                    attack_target_direction = min_unclaimed_direction;
                    target_strength = min_unclaimed_strength;
                    
                    // Unclaimed is MUCH easier to take - just need any strength
                    // This ensures empires expand into unclaimed territory readily
                    attack_strength = available_for_action * 0.8; // Use most available strength
                    
                    let min_attack_threshold = 1.0 / 4095.0;
                    if (attack_strength > min_attack_threshold) {
                        should_attack = true;
                    }
                } else if (min_neutral_direction < 6u && aggression > 0.5) {
                    // Third priority: High-aggression empires attack weak neutrals
                    // This creates more conflicts across the map
                    // LOWERED threshold from 0.6 to 0.5 (50% of empires can do this)
                    let strength_ratio = available_for_action / max(min_neutral_strength, 0.01);
                    
                    // Require strength advantage to attack neutrals
                    // MUCH MORE AGGRESSIVE: aggression = 0.5: need 2.5:1, aggression = 1.0: need 1.5:1
                    let required_advantage = mix(2.5, 1.5, (aggression - 0.5) / 0.5);
                    
                    if (strength_ratio >= required_advantage) {
                        attack_target_direction = min_neutral_direction;
                        target_strength = min_neutral_strength;
                        
                        // Use moderate strength for neutral attacks (not full commitment)
                        attack_strength = available_for_action * 0.5;
                        
                        let min_attack_threshold = 1.0 / 4095.0;
                        if (attack_strength > min_attack_threshold) {
                            should_attack = true;
                        }
                    }
                }
                
                if (should_attack) {
                    // Launch attack
                    let amount_12bit = u32(clamp(max(attack_strength, 1.0 / 4095.0) * 4095.0, 1.0, 4095.0));
                    new_action = attack_target_direction | (amount_12bit << 3u); // Bit 15 = 0 for attack
                    current_strength_f -= max(attack_strength, 1.0 / 4095.0);
                    
                    // Record diplomatic event if attacking another empire (not unclaimed)
                    let target_offset = get_hex_neighbor_offset(attack_target_direction, pos.y);
                    let target_pos = pos + target_offset;
                    let target_cell = get_cell(target_pos);
                    let target_empire = float_to_u16(target_cell.r);
                    if (target_empire != 0u && target_empire != new_empire_id) {
                        record_diplomatic_event(new_empire_id, target_empire, true, amount_12bit, 1.0, frame_data);
                    }
                } else if (max_need_direction < 6u && max_friendly_need > (f32(need) / 65535.0)) {
                    // No valid attack - reinforce instead

                    let my_need = f32(need) / 65535.0;
                    
                    // With gradient-based propagation, adjacent cells have very similar needs
                    // Use absolute difference instead of ratio to detect gradient direction
                    let need_difference = max_friendly_need - my_need;
                    
                    // Only reinforce if neighbor has ANY higher need (gradient flows uphill)
                    // This works because our gradient decay is very gentle (0.0008 per hop)
                    if (need_difference > 0.0001) { // Tiny threshold to avoid floating point errors
                        // Check if we're reinforcing an ally (different empire)
                        let reinforce_offset = get_hex_neighbor_offset(max_need_direction, pos.y);
                        let reinforce_pos = pos + reinforce_offset;
                        let reinforce_cell = get_cell(reinforce_pos);
                        let reinforce_empire = float_to_u16(reinforce_cell.r);
                        
                        var cross_empire_penalty = 1.0;
                        var allow_cross_empire = false;
                        
                        if (reinforce_empire != 0u && reinforce_empire != new_empire_id) {
                            // Different empire - check diplomacy
                            let relation = get_diplomacy(new_empire_id, reinforce_empire);
                            // Only allow cross-empire reinforcement if allied (relation > 0.3)
                            if (relation > 0.3) {
                                allow_cross_empire = true;
                                // Allied reinforcements are less efficient (resources lost in transfer)
                                cross_empire_penalty = 0.3 + (relation * 0.4); // 30% to 70% efficiency
                            }
                        } else if (reinforce_empire == new_empire_id || reinforce_empire == 0u) {
                            // Same empire or unclaimed - always allow
                            allow_cross_empire = true;
                            cross_empire_penalty = 1.0; // Full efficiency
                        }
                        
                        if (allow_cross_empire) {
                            // Calculate safe transfer amount that won't compromise our defense
                            let base_transfer_rate = 0.2; // Conservative transfer rate
                            let need_factor = 1.0 - (my_need * 0.8); // Keep more if we also have high need
                            
                            // Terrain penalty for reinforcement efficiency
                            let transfer_efficiency = 1.0 - (terrain_penalty * 0.95);
                            
                            let safe_transfer_rate = base_transfer_rate * need_factor * transfer_efficiency * cross_empire_penalty;
                            let proposed_transfer = available_for_action * clamp(safe_transfer_rate, 0.05, 0.4);
                            
                            // Verify transfer won't compromise defense
                            if (current_strength_f - proposed_transfer >= total_required_defense) {
                                // Safe to reinforce
                                // NEW 16-bit encoding:
                                // Bits 0-2: direction (0-5)
                                // Bits 3-14: amount (0-4095, 12 bits)
                                // Bit 15: action type (0=attack, 1=reinforce)
                                
                                let amount_12bit = u32(clamp(proposed_transfer * 4095.0, 1.0, 4095.0));
                                new_action = max_need_direction | (amount_12bit << 3u) | (1u << 15u); // Bit 15 = 1 for reinforce
                                current_strength_f -= proposed_transfer;
                                
                                // Record diplomatic event if reinforcing another empire
                                if (reinforce_empire != 0u && reinforce_empire != new_empire_id) {
                                    record_diplomatic_event(new_empire_id, reinforce_empire, false, amount_12bit, 1.0, frame_data);
                                }
                            } else {
                                // Not safe to reinforce
                                new_action = 0u; // No action
                            }
                        } else {
                            // Not allowed to reinforce this empire (not allied)
                            new_action = 0u;
                        }
                    } else {
                        // Need difference not significant enough
                        new_action = 0u; // No action
                    }
                }
            } else {
                // Not enough available strength for any action
                new_action = 0u; // No action
            }
            
            // Update final strength
            new_strength = u32(clamp(current_strength_f * 65535.0, 0.0, 65535.0));
            
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
    
    // Increment age for cells that have an empire and didn't just get conquered
    // (new_age is already set to 0.0 during conquest above)
    if (new_empire_id != 0u) {
        if (new_age < 0.0001) {
            // Just conquered or initial spawn - start at a small visible value
            // With 16-bit precision, we can use much smaller values (1/65535 ≈ 0.000015)
            new_age = 0.001; // Start at visible red
        } else {
            // Existing cell - increment age logarithmically
            // With 16-bit precision, we can use much smaller increments
            let base_rate = 0.0001; // Small increment per frame (~10,000 frames to reach 1.0)
            let increment = base_rate / (new_age + 1.0);
            new_age = min(new_age + increment, 1.0); // Cap at 1.0
        }
    }
    
    // BOAT SPAWNING LOGIC
    // Only land cells with an empire can spawn boats
    if (new_empire_id != 0u && !is_water) {
        // Update boat need: decays very slowly (99.8% retention per frame)
        // Takes ~350 frames to halve - sustains naval campaigns
        boat_need *= 0.998;
        
        // IMPORTANT: Add tiny baseline boat_need for coastal cells to ensure exploration
        // Count ocean neighbors to determine if this is a coastal cell
        var ocean_neighbors = 0u;
        for (var dir: u32 = 0u; dir < 6u; dir++) {
            let offset = get_hex_neighbor_offset(dir, pos.y);
            let neighbor_pos = pos + offset;
            if (neighbor_pos.x >= 0 && neighbor_pos.x < i32(dims.x) && 
                neighbor_pos.y >= 0 && neighbor_pos.y < i32(dims.y)) {
                let neighbor_terrain = get_terrain(neighbor_pos);
                let neighbor_altitude = neighbor_terrain.r;
                if (neighbor_altitude < ocean_cutoff) {
                    ocean_neighbors++;
                }
            }
        }
        
        // Ensure coastal cells always have minimum boat_need for exploration
        if (ocean_neighbors > 0u && boat_need < 0.1) {
            boat_need = max(boat_need, 0.1); // Minimum 0.1 for any coastal cell
        }
        
        // Try to spawn a boat if:
        // 1. We have good strength (> 30000, ~46% of max)
        // 2. We're coastal (have adjacent ocean)
        let strength_f = f32(new_strength) / 65535.0;
        if (new_strength > 30000u && ocean_neighbors > 0u) {
            // Very aggressive spawn rates with baseline probing
            // Add a constant baseline chance (5%) plus scaled boat_need
            // This ensures even low boat_need cells send occasional exploratory boats
            // Use power curve (exponent 0.4) for even stronger low-value boost
            // Examples:
            //  - boat_need=0.10: ~5% baseline + ~6% scaled = ~11%
            //  - boat_need=0.50: ~5% baseline + ~33% scaled = ~38%
            //  - boat_need=1.0:  ~5% baseline + ~50% scaled = ~55%
            let adjusted_need = pow(boat_need, 0.4);
            let spawn_chance = u32(min(5.0 + adjusted_need * 50.0, 100.0));
            
            // Check all 6 hex directions for adjacent ocean cells
            var boats_spawned_this_frame = 0u;
            for (var dir: u32 = 0u; dir < 6u; dir++) {
                let offset = get_hex_neighbor_offset(dir, pos.y);
                let neighbor_pos = pos + offset;
                
                // Check bounds
                if (neighbor_pos.x >= 0 && neighbor_pos.x < i32(dims.x) && 
                    neighbor_pos.y >= 0 && neighbor_pos.y < i32(dims.y)) {
                    
                    let neighbor_terrain = get_terrain(neighbor_pos);
                    let neighbor_altitude = neighbor_terrain.r;
                    let neighbor_is_water = neighbor_altitude < ocean_cutoff;
                    
                    // If this neighbor is ocean, try to spawn a boat there
                    if (neighbor_is_water) {
                        // Each ocean neighbor gets independent spawn roll
                        let spawn_rng = pcg_hash(global_id.x, global_id.y ^ (dir << 16u), frame_data);
                        if ((spawn_rng % 100u) < spawn_chance) {
                            // Spawn a boat!
                            let neighbor_idx = u32(neighbor_pos.y) * dims.x + u32(neighbor_pos.x);
                            let spawn_data = (new_empire_id << 16u) | dir; // Empire ID in upper 16 bits, direction in lower 16 bits
                            atomicMax(&boat_spawns[neighbor_idx], spawn_data);
                            
                            boats_spawned_this_frame++;
                            // Only spawn one boat per frame
                            break;
                        }
                    }
                }
            }
            
            // Only reduce boat_need if we actually spawned boats
            if (boats_spawned_this_frame > 0u) {
                // Miniscule reduction - almost continuous spawning
                boat_need = max(boat_need - 0.01, 0.0);
            }
        }
    }
    
    // Write the new cell state
    let output_color = vec4<f32>(
        u16_to_float(new_empire_id),
        u16_to_float(new_strength),
        u16_to_float(new_need),
        u16_to_float(new_action)
    );
    
    textureStore(output_texture, pos, output_color);
    
    // Write auxiliary data (age, boat_need) to auxiliary texture
    textureStore(aux_output_texture, pos, vec4<f32>(new_age, boat_need, 0.0, 1.0));
}
