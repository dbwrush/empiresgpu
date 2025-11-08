use std::collections::HashMap;
use rayon::prelude::*;

/// Real-time diplomacy state processor
/// Reads atomic counters from GPU, applies diplomatic rules, updates relations
pub struct DiplomacyState {
    /// Current diplomatic relations matrix (symmetric, -1.0 to 1.0)
    /// Key: (min_empire, max_empire) to ensure symmetry
    relations: HashMap<(u16, u16), f32>,
    
    /// Track empire statistics for context-aware diplomacy
    last_territory_sizes: HashMap<u16, u32>,
    
    /// Suppress logging during bulk initialization
    suppress_logging: bool,
    
    /// Reusable buffer for GPU upload (2048×2048 f32s = 16MB)
    /// Cached to avoid repeated allocations
    gpu_buffer: Vec<f32>,
    
    /// Sparse tracking for propagation performance
    /// allies[empire_id] = list of allied empire IDs (relation >= 0.3)
    allies: HashMap<u16, Vec<u16>>,
    /// enemies[empire_id] = list of enemy empire IDs (relation <= -0.3)
    enemies: HashMap<u16, Vec<u16>>,
}

impl DiplomacyState {
    pub fn new() -> Self {
        let mut state = Self {
            relations: HashMap::new(),
            last_territory_sizes: HashMap::new(),
            suppress_logging: true, // Suppress during initialization
            gpu_buffer: vec![0.0f32; 2048 * 2048], // Pre-allocate 16MB buffer
            allies: HashMap::new(),
            enemies: HashMap::new(),
        };
        state.initialize_neutral_relations();
        state.suppress_logging = false; // Re-enable after init
        state
    }
    
    /// Initialize all relations to neutral (0.0)
    fn initialize_neutral_relations(&mut self) {
        // Initialize relations for all possible empire pairs (1-2047)
        for a in 1u16..2048 {
            for b in (a + 1)..2048 {
                self.set_relation(a, b, 0.0);
            }
        }
        
        println!("🤝 Initialized all empire pairs to neutral relations (0.0)");
    }
    
    /// Get relation between two empires
    pub fn get_relation(&self, a: u16, b: u16) -> f32 {
        if a == 0 || b == 0 { return 0.0; } // Invalid
        if a == b { return 1.0; } // Self
        
        let key = if a < b { (a, b) } else { (b, a) };
        *self.relations.get(&key).unwrap_or(&0.0)
    }
    
    /// Set relation between two empires (clamped to -1.0 to 1.0)
    /// Logs when relations cross thresholds: -0.3 (hostile/neutral) or 0.3 (neutral/allied)
    fn set_relation(&mut self, a: u16, b: u16, value: f32) {
        if a == 0 || b == 0 || a == b { return; }
        
        let key = if a < b { (a, b) } else { (b, a) };
        let old_value = self.get_relation(a, b);
        let new_value = value.clamp(-1.0, 1.0);
        
        // Check for threshold crossings (skip during bulk initialization)
        if !self.suppress_logging {
            const HOSTILE_THRESHOLD: f32 = -0.25;  // Lowered from -0.3
            const ALLIED_THRESHOLD: f32 = 0.25;    // Lowered from 0.3
            
            let old_state = if old_value < HOSTILE_THRESHOLD {
                "hostile"
            } else if old_value > ALLIED_THRESHOLD {
                "allied"
            } else {
                "neutral"
            };
            
            let new_state = if new_value < HOSTILE_THRESHOLD {
                "hostile"
            } else if new_value > ALLIED_THRESHOLD {
                "allied"
            } else {
                "neutral"
            };
            
            // Log state changes
            if old_state != new_state {
                println!("🔔 Empires {} & {}: {} → {} (relation: {:.2} → {:.2})",
                         a, b, old_state, new_state, old_value, new_value);
            }
        }
        
        self.relations.insert(key, new_value);
        
        // Update ally/enemy tracking lists
        self.update_alliance_tracking(a, b, old_value, new_value);
    }
    
    /// Update sparse ally/enemy lists when relations change
    /// OPTIMIZED: Lowered thresholds to ±0.25 for more dynamic tracking
    fn update_alliance_tracking(&mut self, a: u16, b: u16, old_value: f32, new_value: f32) {
        const HOSTILE_THRESHOLD: f32 = -0.25;  // Lowered from -0.3
        const ALLIED_THRESHOLD: f32 = 0.25;    // Lowered from 0.3
        
        // Remove old associations
        if old_value >= ALLIED_THRESHOLD {
            // Was allied, remove from both ally lists
            if let Some(allies_a) = self.allies.get_mut(&a) {
                allies_a.retain(|&x| x != b);
            }
            if let Some(allies_b) = self.allies.get_mut(&b) {
                allies_b.retain(|&x| x != a);
            }
        } else if old_value <= HOSTILE_THRESHOLD {
            // Was hostile, remove from both enemy lists
            if let Some(enemies_a) = self.enemies.get_mut(&a) {
                enemies_a.retain(|&x| x != b);
            }
            if let Some(enemies_b) = self.enemies.get_mut(&b) {
                enemies_b.retain(|&x| x != a);
            }
        }
        
        // Add new associations
        if new_value >= ALLIED_THRESHOLD {
            // Now allied, add to both ally lists
            self.allies.entry(a).or_insert_with(Vec::new).push(b);
            self.allies.entry(b).or_insert_with(Vec::new).push(a);
        } else if new_value <= HOSTILE_THRESHOLD {
            // Now hostile, add to both enemy lists
            self.enemies.entry(a).or_insert_with(Vec::new).push(b);
            self.enemies.entry(b).or_insert_with(Vec::new).push(a);
        }
    }
    
    /// Apply a relation change with personality and existing relationship dampening
    /// 
    /// Logic:
    /// - Good relations + similar personalities = negative changes dampened, positive changes amplified
    /// - Bad relations + different personalities = negative changes amplified, positive changes dampened
    /// 
    /// This creates natural "inertia" where friendships are resilient and enmities are hard to fix
    pub fn apply_relation_change(
        &mut self,
        empire_a: u16,
        empire_b: u16,
        raw_change: f32,
        personality_similarity: f32, // 0.0 = completely different, 1.0 = identical
    ) {
        let current_relation = self.get_relation(empire_a, empire_b);
        
        // Normalized current relation: -1.0 to 1.0
        // We want positive relations to dampen negative changes and amplify positive changes
        
        // Calculate dampening factor based on current relations
        // relation_factor ranges from -1.0 (hostile) to 1.0 (allied)
        let relation_factor = current_relation;
        
        // Calculate dampening factor based on personality similarity
        // similarity_factor ranges from -1.0 (different) to 1.0 (similar)
        let similarity_factor = personality_similarity * 2.0 - 1.0; // Map [0,1] to [-1,1]
        
        // Combined dampening: average of relation and similarity factors
        let combined_factor = (relation_factor + similarity_factor) / 2.0; // -1.0 to 1.0
        
        // Apply dampening:
        // - For negative changes (raw_change < 0):
        //   - High combined_factor (friendly + similar) -> dampen (multiply by smaller value)
        //   - Low combined_factor (hostile + different) -> amplify (multiply by larger value)
        // - For positive changes (raw_change > 0):
        //   - High combined_factor (friendly + similar) -> amplify
        //   - Low combined_factor (hostile + different) -> dampen
        
        let modified_change = if raw_change < 0.0 {
            // Negative change: dampen if friendly/similar, amplify if hostile/different
            // combined_factor of 1.0 -> multiply by 0.5 (50% dampening)
            // combined_factor of -1.0 -> multiply by 2.0 (200% amplification)
            let multiplier = 1.0 - (combined_factor * 0.5); // 0.5 to 1.5
            raw_change * multiplier
        } else {
            // Positive change: amplify if friendly/similar, dampen if hostile/different
            // combined_factor of 1.0 -> multiply by 1.5 (150% amplification)
            // combined_factor of -1.0 -> multiply by 0.5 (50% dampening)
            let multiplier = 1.0 + (combined_factor * 0.5); // 0.5 to 1.5
            raw_change * multiplier
        };
        
        // Apply the modified change
        let new_relation = current_relation + modified_change;
        self.set_relation(empire_a, empire_b, new_relation);
    }
    
    /// Process diplomatic counters from GPU and update relations
    /// counters: flat array of u32 [attack_count, attack_strength, reinforce_count, reinforce_strength, pressure_count, ...]
    /// for each empire pair (2048×2048 = 4,194,304 pairs × 5 values = 20,971,520 u32s)
    /// personality_diff_fn: Function that returns personality difference (0.0-1.0) for two empires
    /// OPTIMIZED: Early exit for empty regions, parallel row processing
    pub fn process_counters<F>(
        &mut self, 
        counters: &[u32], 
        territory_sizes: &HashMap<u16, u32>,
        personality_diff_fn: F,
    ) 
    where
        F: Fn(u16, u16) -> f32 + Sync + Send,
    {
        // Parallel processing: Collect all relation updates from all empire pairs
        // We process rows in parallel, with each row processing its columns sequentially
        // OPTIMIZED: Skip rows with no events
        let updates: Vec<(u16, u16, f32)> = (0..2048u16)
            .into_par_iter()
            .flat_map(|row| {
                let mut row_updates = Vec::new();
                
                // Early exit: Check if this entire row has any events at all
                let row_start = (row as usize) * 2048 * 5;
                let row_end = row_start + (2048 * 5);
                if row_end <= counters.len() {
                    // Quick scan: Does this row have ANY non-zero values?
                    let has_events = counters[row_start..row_end].iter().any(|&x| x > 0);
                    if !has_events {
                        return row_updates; // Skip entire row
                    }
                }
                
                for col in (row + 1)..2048u16 {
                    let idx = (row as usize) * 2048 + (col as usize);
                    let base = idx * 5; // 5 counters per pair
                    
                    if base + 4 >= counters.len() {
                        break;
                    }
                    
                    let attack_count = counters[base];
                    let attack_strength = counters[base + 1];
                    let reinforce_count = counters[base + 2];
                    let reinforce_strength = counters[base + 3];
                    let pressure_count = counters[base + 4];
                    
                    // Skip if no events
                    if attack_count == 0 && reinforce_count == 0 && pressure_count == 0 {
                        continue;
                    }
                    
                    let empire_a = row + 1; // Empire IDs are 1-indexed
                    let empire_b = col + 1;
                    
                    // Calculate personality difference for this pair
                    let personality_diff = personality_diff_fn(empire_a, empire_b);
                    
                    // Calculate the relation delta (without applying it yet)
                    let delta = self.calculate_relation_delta(
                        empire_a,
                        empire_b,
                        attack_count,
                        attack_strength,
                        reinforce_count,
                        reinforce_strength,
                        pressure_count,
                        personality_diff,
                        territory_sizes,
                    );
                    
                    if delta.abs() > 0.0001 {
                        row_updates.push((empire_a, empire_b, delta));
                    }
                }
                
                row_updates
            })
            .collect();
        
        // Apply all updates sequentially using dampened relation changes
        for (empire_a, empire_b, delta) in updates {
            // Get personality similarity (1.0 - difference)
            let personality_diff = personality_diff_fn(empire_a, empire_b);
            let personality_similarity = 1.0 - personality_diff;
            
            self.apply_relation_change(empire_a, empire_b, delta, personality_similarity);
        }
        
        // Apply "enemy of my enemy is my friend" alliance formation
        // This creates natural alliance blocs against common threats
        self.propagate_common_enemy_alliances(&personality_diff_fn);
        
        // Debug: Count total allies and enemies
        let total_allies: usize = self.allies.values().map(|v| v.len()).sum();
        let total_enemies: usize = self.enemies.values().map(|v| v.len()).sum();
        let total_relations = self.relations.len();
        let possible_relations = (2047 * 2048) / 2; // All possible empire pairs
        println!("📊 Relations: {} pairs tracked ({:.1}% of possible), {} ally links, {} enemy links", 
                 total_relations, (total_relations as f32 / possible_relations as f32) * 100.0,
                 total_allies / 2, total_enemies / 2);
        
        // Update territory tracking
        self.last_territory_sizes = territory_sizes.clone();
    }
    
    /// Process attack reactions: when A attacks B, propagate to allies and enemies
    /// - Allies of B should hate A (you attacked my friend)
    /// - Allies of A should hate B (they attacked my friend)
    /// - Enemies of B should like A (you attacked my enemy)
    /// - Enemies of A should like B (they attacked my enemy)
    pub fn propagate_attack_reactions<F>(&mut self, counters: &[u32], personality_diff_fn: F)
    where
        F: Fn(u16, u16) -> f32,
    {
        let mut relation_changes = Vec::new();
        let mut attacks_found = 0;
        
        // Scan through all empire pairs to find attacks
        for row in 0..2048u16 {
            for col in (row + 1)..2048u16 {
                let empire_a = row + 1;
                let empire_b = col + 1;
                
                // Calculate index into counter array
                let idx = (row as usize) * 2048 + (col as usize);
                let base_idx = idx * 5;
                
                if base_idx + 4 >= counters.len() {
                    continue;
                }
                
                let attack_count = counters[base_idx];
                
                // If there was an attack between A and B
                if attack_count > 0 {
                    attacks_found += 1;
                    let intensity = (attack_count as f32 / 256.0).min(1.0);
                    
                    // Get allies and enemies of both A and B using sparse tracking
                    let allies_of_a = self.allies.get(&empire_a);
                    let enemies_of_a = self.enemies.get(&empire_a);
                    let allies_of_b = self.allies.get(&empire_b);
                    let enemies_of_b = self.enemies.get(&empire_b);
                    
                    // Rule 1: "You attacked my friend" - Allies of B should hate A
                    if let Some(b_allies) = allies_of_b {
                        for &ally in b_allies.iter() {
                            if ally != empire_a {
                                let personality_similarity = 1.0 - personality_diff_fn(empire_a, ally);
                                let penalty = -0.35 * intensity;  // Massively increased from -0.15
                                relation_changes.push((empire_a, ally, penalty, personality_similarity));
                            }
                        }
                    }
                    
                    // Rule 2: "They attacked my friend" - Allies of A should hate B  
                    if let Some(a_allies) = allies_of_a {
                        for &ally in a_allies.iter() {
                            if ally != empire_b {
                                let personality_similarity = 1.0 - personality_diff_fn(empire_b, ally);
                                let penalty = -0.35 * intensity;  // Massively increased from -0.15
                                relation_changes.push((empire_b, ally, penalty, personality_similarity));
                            }
                        }
                    }
                    
                    // Rule 3: "You attacked my enemy" - Enemies of B should like A
                    if let Some(b_enemies) = enemies_of_b {
                        for &enemy in b_enemies.iter() {
                            if enemy != empire_a {
                                let personality_similarity = 1.0 - personality_diff_fn(empire_a, enemy);
                                let bonus = 0.25 * intensity;  // Massively increased from 0.12
                                relation_changes.push((empire_a, enemy, bonus, personality_similarity));
                            }
                        }
                    }
                    
                    // Rule 4: "They attacked my enemy" - Enemies of A should like B
                    if let Some(a_enemies) = enemies_of_a {
                        for &enemy in a_enemies.iter() {
                            if enemy != empire_b {
                                let personality_similarity = 1.0 - personality_diff_fn(empire_b, enemy);
                                let bonus = 0.25 * intensity;  // Massively increased from 0.12
                                relation_changes.push((empire_b, enemy, bonus, personality_similarity));
                            }
                        }
                    }
                }
            }
        }
        
        // Debug logging
        if attacks_found > 0 {
            println!("⚔️ Propagation: {} attacks, {} third-party reactions", 
                     attacks_found, relation_changes.len());
        }
        
        // Apply all relation changes with dampening
        for (empire_a, empire_c, change, personality_similarity) in relation_changes {
            self.apply_relation_change(empire_a, empire_c, change, personality_similarity);
        }
    }
    
    /// Calculate the relation delta for an empire pair (used by parallel processing)
    fn calculate_relation_delta(
        &self,
        empire_a: u16,
        empire_b: u16,
        attack_count: u32,
        attack_strength: u32,
        reinforce_count: u32,
        reinforce_strength: u32,
        pressure_count: u32,
        personality_diff: f32,
        territory_sizes: &HashMap<u16, u32>,
    ) -> f32 {
        let current_relation = self.get_relation(empire_a, empire_b);
        let mut new_relation = current_relation;
        
        // Personality-based modifiers:
        // - Negative events (attacks, pressure) amplified by difference (similar empires less likely to fight)
        // - Positive events (reinforcements, peace) amplified by similarity (similar empires ally easier)
        let similarity = 1.0 - personality_diff;  // 0.0 = different, 1.0 = similar
        let difference_amplifier = 1.0 + personality_diff * 5.0;  // 1.0 to 6.0 (much stronger)
        let similarity_amplifier = 1.0 + similarity * 5.0;       // 1.0 to 6.0 (much stronger)
        
        // Rule 1: Attacks degrade relations rapidly
        // Philosophy: Direct conflict is the strongest driver of hostility
        // Amplified by personality difference (different empires clash more severely)
        if attack_count > 0 {
            // Scale by both frequency (count) and intensity (strength)
            let attack_impact = (attack_count as f32).sqrt() * 0.025  // Massively increased
                              + (attack_strength as f32).sqrt() * 0.003; // Massively increased
            let modified_impact = attack_impact * difference_amplifier;
            new_relation -= modified_impact.min(0.5); // Max -0.5 per cycle (instant hostility)
        }
        
        // Rule 1b: Border pressure can degrade OR improve relations based on personality
        // Philosophy: Border interactions are opportunities for friction OR friendship
        // - Different personalities: tension and resentment (negative)
        // - Similar personalities: cultural exchange and understanding (positive)
        // MUCH stronger effect to quickly push neutrals into conflicts or alliances
        if pressure_count > 0 {
            let pressure_sqrt = (pressure_count as f32).sqrt();
            
            // Raised threshold to 0.7 - empires need to be VERY similar to bond
            // With Euclidean distance metric, most empires will fall below this
            if similarity > 0.7 {
                // Very similar (>70%) -> border contact improves relations
                let friendship_bonus = pressure_sqrt * 0.025 * similarity_amplifier;
                new_relation += friendship_bonus.min(0.2); // Max +0.2 per cycle (very fast alliance)
            } else {
                // All other empires (<=70% similar) -> border contact degrades relations
                // EXTREMELY strong effect to rapidly create conflicts
                let pressure_impact = pressure_sqrt * 0.08 * difference_amplifier;
                new_relation -= pressure_impact.min(0.5); // Max -0.5 per cycle (instant hostility)
            }
        }
        
        // Rule 2: Reinforcements improve relations (but slower than attacks degrade)
        // Philosophy: Helping builds trust, but trust takes time
        // Amplified by personality similarity (similar empires bond easier)
        if reinforce_count > 0 {
            // Only count reinforcements if empires aren't already allied
            // (prevents runaway positive feedback)
            if current_relation < 0.8 {
                let reinforce_impact = (reinforce_count as f32).sqrt() * 0.002  // Increased
                                     + (reinforce_strength as f32).sqrt() * 0.0002; // Increased
                let modified_impact = reinforce_impact * similarity_amplifier;
                new_relation += modified_impact.min(0.03); // Max +0.03 per cycle
            }
        }
        
        // Rule 3: Mutual non-aggression slowly improves relations
        // Philosophy: Peaceful coexistence breeds familiarity (but removed - now only border contact matters)
        // REMOVED - relations only change through actual interactions
        
        // Rule 4: Desperation and surrender
        // Philosophy: Losing empires may seek peace to survive
        let a_size = territory_sizes.get(&empire_a).copied().unwrap_or(0);
        let b_size = territory_sizes.get(&empire_b).copied().unwrap_or(0);
        let a_last_size = self.last_territory_sizes.get(&empire_a).copied().unwrap_or(a_size);
        let b_last_size = self.last_territory_sizes.get(&empire_b).copied().unwrap_or(b_size);
        
        let a_losing = a_last_size.saturating_sub(a_size);
        let b_losing = b_last_size.saturating_sub(b_size);
        
        // If empire is losing badly (>5% of previous size) and currently hostile
        let a_desperate = a_last_size > 0 && (a_losing * 100) / a_last_size > 5;
        let b_desperate = b_last_size > 0 && (b_losing * 100) / b_last_size > 5;
        
        if a_desperate && attack_count > 0 && current_relation < -0.2 {
            // A is desperate and under attack - rapidly improve relations (surrender/vassalage)
            new_relation += 0.03;
        }
        if b_desperate && attack_count > 0 && current_relation < -0.2 {
            new_relation += 0.03;
        }
        
        // NO natural decay - relations only change through interactions
        
        // Return the delta
        new_relation - current_relation
    }
    
    /// "Enemy of my enemy is my friend" - form alliances against common threats
    /// This creates natural alliance blocs without arbitrary ideological grouping
    /// OPTIMIZED: Uses sparse enemy tracking instead of O(n³) full scan
    fn propagate_common_enemy_alliances<F>(&mut self, personality_diff_fn: &F)
    where
        F: Fn(u16, u16) -> f32,
    {
        let mut alliance_boosts = Vec::new();
        let mut mutual_enemies_found = 0;
        
        // OPTIMIZED: Iterate through empires that HAVE enemies (sparse)
        // Instead of checking all 2048 empires for each hostile pair
        for (&empire_a, enemies_of_a) in &self.enemies {
            // For each enemy of A
            for &empire_b in enemies_of_a {
                // Find other empires that also hate B (use sparse tracking!)
                if let Some(enemies_of_b) = self.enemies.get(&empire_b) {
                    // Iterate through enemies of B
                    for &empire_c in enemies_of_b {
                        if empire_c == empire_a { continue; }
                        
                        mutual_enemies_found += 1;
                        // Both A and C hate B! They should become friendlier
                        let relation_ac = self.get_relation(empire_a, empire_c);
                        
                        // Only boost if not already allied (prevent runaway)
                        if relation_ac < 0.7 {
                            let relation_ab = self.get_relation(empire_a, empire_b);
                            let relation_cb = self.get_relation(empire_c, empire_b);
                            
                            // Boost scales with how much they both hate the common enemy
                            let mutual_hatred = (relation_ab.abs() + relation_cb.abs()) / 2.0;
                            let boost = (mutual_hatred * 0.2).min(0.3); // Massively stronger boost
                            alliance_boosts.push((empire_a, empire_c, boost));
                        }
                    }
                }
            }
        }
        
        let empires_with_enemies = self.enemies.len();
        println!("🤝 Common enemy scan: {} empires with enemies, {} mutual enemy situations, {} alliance boosts", 
                 empires_with_enemies, mutual_enemies_found, alliance_boosts.len());
        
        // Apply alliance boosts using dampened relation changes
        for (a, c, boost) in alliance_boosts {
            let personality_diff = personality_diff_fn(a, c);
            let personality_similarity = 1.0 - personality_diff;
            self.apply_relation_change(a, c, boost, personality_similarity);
        }
    }
    
    /// Convert relations to flat f32 array for GPU upload (2048×2048)
    /// Reuses cached buffer to avoid repeated 16MB allocations
    pub fn to_buffer(&mut self) -> &[f32] {
        // Clear buffer to zeros (fast memset)
        self.gpu_buffer.fill(0.0);
        
        // Update relations (sequential - HashMap iteration is already fast)
        for ((a, b), relation) in &self.relations {
            let idx_ab = (*a as usize - 1) * 2048 + (*b as usize - 1);
            let idx_ba = (*b as usize - 1) * 2048 + (*a as usize - 1);
            self.gpu_buffer[idx_ab] = *relation;
            self.gpu_buffer[idx_ba] = *relation;
        }
        
        // Set self-relations to 1.0 (sequential - only 2048 writes)
        for i in 0..2048 {
            self.gpu_buffer[i * 2048 + i] = 1.0;
        }
        
        &self.gpu_buffer
    }
}

impl Default for DiplomacyState {
    fn default() -> Self {
        Self::new()
    }
}
