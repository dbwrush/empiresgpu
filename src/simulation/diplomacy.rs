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
    
    /// Track when borders last changed for stagnation detection
    /// Key: (min_empire, max_empire), Value: frames since last territory change
    border_stagnation: HashMap<(u16, u16), u32>,
    
    /// Track last interaction time for war weariness
    /// Key: (min_empire, max_empire), Value: frames since last interaction (attack/reinforce)
    last_interaction: HashMap<(u16, u16), u32>,
    
    /// Suppress logging during bulk initialization
    suppress_logging: bool,
    
    /// Reusable buffer for GPU upload (2048×2048 f32s = 16MB)
    /// Cached to avoid repeated allocations
    gpu_buffer: Vec<f32>,
    
    /// Sparse tracking for propagation performance
    /// allies[empire_id] = list of allied empire IDs (relation >= 0.65)
    allies: HashMap<u16, Vec<u16>>,
    /// enemies[empire_id] = list of enemy empire IDs (relation <= -0.1)
    enemies: HashMap<u16, Vec<u16>>,
}

impl DiplomacyState {
    pub fn new() -> Self {
        let mut state = Self {
            relations: HashMap::new(),
            last_territory_sizes: HashMap::new(),
            border_stagnation: HashMap::new(),
            last_interaction: HashMap::new(),
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
        // OPTIMIZED: Don't pre-populate HashMap with 2M+ entries
        // Relations default to 0.0 via get_relation() when not present
        // Only add entries when relations change from neutral
        // This saves ~32MB of memory and seconds of startup time
        
        println!("Relations initialized (lazy - will populate on first change)");
    }
    
    /// Get relation between two empires
    pub fn get_relation(&self, a: u16, b: u16) -> f32 {
        if a == 0 || b == 0 { return 0.0; } // Invalid
        if a == b { return 1.0; } // Self
        
        let key = if a < b { (a, b) } else { (b, a) };
        *self.relations.get(&key).unwrap_or(&0.0)
    }
    
    /// Set relation between two empires (clamped to -1.0 to 1.0)
    /// Logs when relations cross thresholds: -0.1 (hostile/neutral) or 0.65 (neutral/allied)
    fn set_relation(&mut self, a: u16, b: u16, value: f32) {
        if a == 0 || b == 0 || a == b { return; }
        
        let key = if a < b { (a, b) } else { (b, a) };
        let old_value = self.get_relation(a, b);
        let new_value = value.clamp(-1.0, 1.0);
        
        // Check for threshold crossings (skip during bulk initialization)
        if !self.suppress_logging {
            const HOSTILE_THRESHOLD: f32 = -0.1;   // Match most aggressive GPU empires
            const ALLIED_THRESHOLD: f32 = 0.65;    // Increased from 0.25/0.3 - alliances are now VERY hard to form
            
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
                println!("Empires {} & {}: {} → {} (relation: {:.2} → {:.2})",
                         a, b, old_state, new_state, old_value, new_value);
            }
        }
        
        self.relations.insert(key, new_value);
        
        // Update ally/enemy tracking lists
        self.update_alliance_tracking(a, b, old_value, new_value);
    }
    
    /// Update alliance tracking when a specific relation changes
    fn update_alliance_tracking(&mut self, a: u16, b: u16, old_value: f32, new_value: f32) {
        const ALLIED_THRESHOLD: f32 = 0.65;  // Increased from 0.3 - alliances are now VERY hard to form
        const HOSTILE_THRESHOLD: f32 = -0.1;   // Match most aggressive GPU empires
        
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
    
    /// Update sparse ally/enemy lists - rebuild from scratch
    /// OPTIMIZED: Lowered thresholds to ±0.25 for more dynamic tracking
    fn update_ally_enemy_lists(&mut self) {
        self.allies.clear();
        self.enemies.clear();
        
        const ALLIED_THRESHOLD: f32 = 0.65;  // Increased from 0.3 - alliances are now VERY hard to form
        const HOSTILE_THRESHOLD: f32 = -0.1;   // Match most aggressive GPU empires
        
        for (&(a, b), &relation) in &self.relations {
            if relation >= ALLIED_THRESHOLD {
                self.allies.entry(a).or_insert_with(Vec::new).push(b);
                self.allies.entry(b).or_insert_with(Vec::new).push(a);
            } else if relation <= HOSTILE_THRESHOLD {
                self.enemies.entry(a).or_insert_with(Vec::new).push(b);
                self.enemies.entry(b).or_insert_with(Vec::new).push(a);
            }
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
        
        // NEW: Add dampening based on absolute relation value
        // Make it much harder to reach extreme values (±0.65 threshold for allied/hostile)
        // This prevents runaway alliance/war spirals
        let abs_relation = current_relation.abs();
        let extreme_dampening = if abs_relation > 0.6 {
            // VERY strong dampening when approaching ±1.0
            // 0.6 -> 0.8 (20% dampening)
            // 0.7 -> 0.5 (50% dampening)
            // 0.8 -> 0.2 (80% dampening)
            // 0.9 -> 0.05 (95% dampening)
            (1.0 - abs_relation) * 2.0 // Linear from 0.8 at 0.6 to 0.2 at 0.9, 0.0 at 1.0
        } else if abs_relation > 0.4 {
            // Strong dampening in the "allied/hostile zone"
            // 0.4 -> 1.0 (full effect)
            // 0.5 -> 0.85 (15% dampening)
            // 0.6 -> 0.7 (30% dampening)
            1.0 - ((abs_relation - 0.4) * 1.5) // From 1.0 at 0.4 to 0.7 at 0.6
        } else if abs_relation > 0.25 {
            // Moderate dampening approaching the threshold
            // 0.25 -> 1.0 (full effect)
            // 0.3 -> 0.85 (15% dampening at alliance threshold)
            // 0.4 -> 0.7 (30% dampening)
            1.0 - ((abs_relation - 0.25) * 2.0) // From 1.0 at 0.25 to 0.7 at 0.4
        } else {
            1.0 // No dampening for neutral relations
        };
        
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
            raw_change * multiplier * extreme_dampening // Apply extreme dampening
        } else {
            // Positive change: amplify if friendly/similar, dampen if hostile/different
            // combined_factor of 1.0 -> multiply by 1.5 (150% amplification)
            // combined_factor of -1.0 -> multiply by 0.5 (50% dampening)
            let multiplier = 1.0 + (combined_factor * 0.5); // 0.5 to 1.5
            raw_change * multiplier * extreme_dampening // Apply extreme dampening
        };
        
        // NEW: Gradual drift toward neutrality over time
        // This prevents relations from getting permanently stuck at extremes
        // Strength scales with absolute value (stronger pull when further from neutral)
        // This is very subtle but helps maintain dynamic diplomacy
        let neutral_drift = if current_relation.abs() > 0.1 {
            // Pull toward 0.0, strength proportional to distance from neutral
            // At ±0.2: drift of ±0.0004 per frame
            // At ±0.5: drift of ±0.001 per frame  
            // At ±0.8: drift of ±0.0016 per frame
            // At ±1.0: drift of ±0.002 per frame
            let drift_strength = 0.01; // Increased from 0.002 to make neutral pull stronger per request
            -current_relation.signum() * current_relation.abs() * drift_strength
        } else {
            0.0  // No drift when already near neutral
        };
        
        // Apply the modified change plus neutral drift
        let new_relation = current_relation + modified_change + neutral_drift;
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
        frame_count: u32,  // Add frame counter for throttling expensive operations
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
                    
                    // Skip if no events
                    if attack_count == 0 && reinforce_count == 0 {
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
        
        // OPTIMIZED: Build a HashSet of pairs that had interactions for O(1) lookup
        let mut interacted_pairs = std::collections::HashSet::new();
        for &(a, b, _) in &updates {
            let key = if a < b { (a, b) } else { (b, a) };
            interacted_pairs.insert(key);
        }
        
        // Track interactions only for empires that are enemies (sparse)
        // Increment counter for all enemy pairs, reset when we see activity
        for (&empire_a, enemies_list) in &self.enemies {
            for &empire_b in enemies_list {
                if empire_a >= empire_b { continue; } // Process each pair once
                
                let key = (empire_a, empire_b);
                
                if interacted_pairs.contains(&key) {
                    // Reset interaction timer
                    self.last_interaction.insert(key, 0);
                } else {
                    // No interaction this frame, increment counter
                    let count = self.last_interaction.entry(key).or_insert(0);
                    *count += 1;
                }
            }
        }
        
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
        
        // Break up conflicting alliances (empires allied to both sides of a war)
        self.break_conflicting_alliances(&personality_diff_fn);
        
        // OPTIMIZATION: Only run expensive decay/weariness systems every 30 frames
        // These are slower-moving diplomatic forces that don't need frame-perfect updates
        if frame_count % 30 == 0 {
            // Apply war weariness for stagnant conflicts
            self.apply_war_weariness(territory_sizes, &personality_diff_fn);
        }
        
        // Debug: Count total allies and enemies
        let total_allies: usize = self.allies.values().map(|v| v.len()).sum();
        let total_enemies: usize = self.enemies.values().map(|v| v.len()).sum();
        let total_relations = self.relations.len();
        let possible_relations = (2047 * 2048) / 2; // All possible empire pairs
        println!("Relations: {} pairs tracked ({:.1}% of possible), {} ally links, {} enemy links", 
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
    /// OPTIMIZED: Parallel processing with early exit, limited propagation depth
    pub fn propagate_attack_reactions<F>(&mut self, counters: &[u32], personality_diff_fn: F)
    where
        F: Fn(u16, u16) -> f32 + Sync + Send,
    {
        // OPTIMIZATION: Limit the number of allies/enemies we process to prevent O(n³) explosion
        const MAX_ALLIES_TO_PROCESS: usize = 8;
        const MAX_ENEMIES_TO_PROCESS: usize = 8;
        
        // OPTIMIZED: Parallel scan for attacks, early exit for empty rows
        let relation_changes: Vec<_> = (0..2048u16)
            .into_par_iter()
            .flat_map(|row| {
                let mut changes = Vec::new();
                
                // Early exit: Check if this row has any attacks
                let row_start = (row as usize) * 2048 * 5;
                let row_end = row_start + (2048 * 5);
                if row_end <= counters.len() {
                    let has_attacks = counters[row_start..row_end]
                        .chunks_exact(5)
                        .any(|chunk| chunk[0] > 0); // attack_count is first
                    if !has_attacks {
                        return changes; // Skip entire row
                    }
                }
                
                for col in (row + 1)..2048u16 {
                    let empire_a = row + 1;
                    let empire_b = col + 1;
                    
                    let idx = (row as usize) * 2048 + (col as usize);
                    let base_idx = idx * 5;
                    
                    if base_idx + 4 >= counters.len() {
                        break;
                    }
                    
                    let attack_count = counters[base_idx];
                    
                    // If there was an attack between A and B
                    if attack_count > 0 {
                        let intensity = (attack_count as f32 / 256.0).min(1.0);
                        
                        // Get allies and enemies - need to capture for parallel closure
                        let allies_of_a = self.allies.get(&empire_a).cloned();
                        let enemies_of_a = self.enemies.get(&empire_a).cloned();
                        let allies_of_b = self.allies.get(&empire_b).cloned();
                        let enemies_of_b = self.enemies.get(&empire_b).cloned();
                        
                        // Rule 1: "You attacked my friend" - Allies of B should hate A
                        // OPTIMIZATION: Limit to MAX_ALLIES_TO_PROCESS most important allies
                        // HEAVILY REDUCED: Was -0.35 → -0.03 → now -0.005 to prevent boat skirmishes from affecting third parties
                        if let Some(b_allies) = &allies_of_b {
                            for &ally in b_allies.iter().take(MAX_ALLIES_TO_PROCESS) {
                                if ally != empire_a {
                                    let personality_similarity = 1.0 - personality_diff_fn(empire_a, ally);
                                    let penalty = -0.005 * intensity;  // Reduced from -0.03 (83% further reduction)
                                    changes.push((empire_a, ally, penalty, personality_similarity));
                                }
                            }
                        }
                        
                        // Rule 2: "They attacked my friend" - Allies of A should hate B  
                        // OPTIMIZATION: Limit to MAX_ALLIES_TO_PROCESS most important allies
                        // HEAVILY REDUCED: Was -0.35 → -0.03 → now -0.005 to prevent boat skirmishes from affecting third parties
                        if let Some(a_allies) = &allies_of_a {
                            for &ally in a_allies.iter().take(MAX_ALLIES_TO_PROCESS) {
                                if ally != empire_b {
                                    let personality_similarity = 1.0 - personality_diff_fn(empire_b, ally);
                                    let penalty = -0.005 * intensity;  // Reduced from -0.03 (83% further reduction)
                                    changes.push((empire_b, ally, penalty, personality_similarity));
                                }
                            }
                        }
                        
                        // Rule 3: "You attacked my enemy" - Enemies of B should like A
                        // BUT ONLY if the enemy has recently fought B (not stagnant)
                        // OPTIMIZATION: Limit to MAX_ENEMIES_TO_PROCESS most active enemies
                        // HEAVILY REDUCED: Was 0.25 → 0.01 → now 0.002 to prevent boat skirmishes from creating alliances
                        if let Some(b_enemies) = &enemies_of_b {
                            for &enemy in b_enemies.iter().take(MAX_ENEMIES_TO_PROCESS) {
                                if enemy != empire_a {
                                    // Check if this enemy has recently fought B
                                    let key = if enemy < empire_b { (enemy, empire_b) } else { (empire_b, enemy) };
                                    let stagnation = self.border_stagnation.get(&key).copied().unwrap_or(0);
                                    
                                    // Only boost if conflict is active (< 30 frames stagnant)
                                    if stagnation < 30 {
                                        let personality_similarity = 1.0 - personality_diff_fn(empire_a, enemy);
                                        let bonus = 0.002 * intensity;  // Reduced from 0.01 (80% further reduction)
                                        changes.push((empire_a, enemy, bonus, personality_similarity));
                                    } else if stagnation < 100 {
                                        let personality_similarity = 1.0 - personality_diff_fn(empire_a, enemy);
                                        let bonus = 0.0005 * intensity;  // Reduced from 0.003 (83% further reduction)
                                        changes.push((empire_a, enemy, bonus, personality_similarity));
                                    }
                                }
                            }
                        }
                        
                        // Rule 4: "They attacked my enemy" - Enemies of A should like B
                        // BUT ONLY if the enemy has recently fought A (not stagnant)
                        // OPTIMIZATION: Limit to MAX_ENEMIES_TO_PROCESS most active enemies
                        // HEAVILY REDUCED: Was 0.25 → 0.01 → now 0.002 to prevent boat skirmishes from creating alliances
                        if let Some(a_enemies) = &enemies_of_a {
                            for &enemy in a_enemies.iter().take(MAX_ENEMIES_TO_PROCESS) {
                                if enemy != empire_b {
                                    let key = if enemy < empire_a { (enemy, empire_a) } else { (empire_a, enemy) };
                                    let stagnation = self.border_stagnation.get(&key).copied().unwrap_or(0);
                                    
                                    if stagnation < 30 {
                                        let personality_similarity = 1.0 - personality_diff_fn(empire_b, enemy);
                                        let bonus = 0.002 * intensity;  // Reduced from 0.01 (80% further reduction)
                                        changes.push((empire_b, enemy, bonus, personality_similarity));
                                    } else if stagnation < 100 {
                                        let personality_similarity = 1.0 - personality_diff_fn(empire_b, enemy);
                                        let bonus = 0.0005 * intensity;  // Reduced from 0.003 (83% further reduction)
                                        changes.push((empire_b, enemy, bonus, personality_similarity));
                                    }
                                }
                            }
                        }
                    }
                }
                
                changes
            })
            .collect();
        
        // Debug logging
        let attacks_found = relation_changes.len();
        if attacks_found > 0 {
            println!("Propagation: {} third-party reactions", attacks_found);
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
        
        // Rule 2: Reinforcements improve relations (but EXTREMELY slow - alliances are rare)
        // Philosophy: Trust is very hard to build, requires sustained cooperation
        // Amplified by personality similarity (similar empires bond easier)
        if reinforce_count > 0 {
            // Only count reinforcements if empires aren't already allied
            // (prevents runaway positive feedback)
            if current_relation < 0.8 {
                // EXTREMELY REDUCED: Alliances should be exceptionally rare
                let reinforce_impact = (reinforce_count as f32).sqrt() * 0.0001  // Cut by 50% from 0.0002
                                     + (reinforce_strength as f32).sqrt() * 0.00001; // Cut by 50% from 0.00002
                let modified_impact = reinforce_impact * similarity_amplifier;
                new_relation += modified_impact.min(0.0025); // Cut by 50% from 0.005
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
            new_relation += 0.015; // Cut by 50% from 0.03
        }
        if b_desperate && attack_count > 0 && current_relation < -0.2 {
            new_relation += 0.015; // Cut by 50% from 0.03
        }
        
        // Rule 5: Natural drift toward conflict based on personality difference
        // Philosophy: Empires with different values naturally develop tensions over time
        // Similar empires (high similarity) have little drift
        // Different empires (low similarity) drift toward hostility
        if current_relation > -0.5 && current_relation < 0.3 {
            // Only apply drift in the neutral/slightly hostile range
            // Don't make existing wars worse or existing alliances decay (that's handled elsewhere)
            
            // Drift strength varies by personality difference
            // MASSIVELY INCREASED to ensure borders turn into conflicts quickly
            // Very different empires (personality_diff = 1.0, similarity = 0.0):
            //   - difference_amplifier = 6.0, drift = -0.3 per frame (instant war)
            // Moderately different (personality_diff = 0.5, similarity = 0.5):
            //   - difference_amplifier = 3.5, drift = -0.175 per frame
            // Very similar (personality_diff = 0.1, similarity = 0.9):
            //   - difference_amplifier = 1.5, drift = -0.075 per frame
            let base_drift = -0.05; // Increased from -0.01 to -0.05 (5x stronger)
            let personality_drift = base_drift * difference_amplifier;
            
            new_relation += personality_drift.max(-0.3); // Cap at -0.3 per frame (instant war)
        }
        
        // NO other natural decay - relations only change through interactions
        
        // Return the delta
        new_relation - current_relation
    }
    
    /// "Enemy of my enemy is my friend" - form alliances against common threats
    /// This creates natural alliance blocs without arbitrary ideological grouping
    /// OPTIMIZED: Uses sparse enemy tracking, alliance decay integrated
    fn propagate_common_enemy_alliances<F>(&mut self, personality_diff_fn: &F)
    where
        F: Fn(u16, u16) -> f32,
    {
        let mut alliance_boosts = Vec::new();
        let mut alliance_decay_changes: Vec<(u16, u16, f32, f32, usize)> = Vec::new();
        let mut mutual_enemies_found = 0;
        
        // OPTIMIZATION: Limit scope to reduce O(n³) complexity
        // Process all empires with at least one enemy (allow any empire to seek allies)
        let empires_with_multiple_enemies: Vec<_> = self.enemies.iter()
            .filter(|(_, enemies)| enemies.len() >= 1)
            .map(|(empire, _)| *empire)
            .collect();
        
        // Instead of checking all 2048 empires for each hostile pair
        for &empire_a in &empires_with_multiple_enemies {
            let enemies_of_a = match self.enemies.get(&empire_a) {
                Some(enemies) => enemies,
                None => continue,
            };
            
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
                        // Also require that they're not already hostile to each other
                        if relation_ac < 0.5 && relation_ac > -0.2 {
                            let relation_ab = self.get_relation(empire_a, empire_b);
                            let relation_cb = self.get_relation(empire_c, empire_b);
                            
                            // CRITICAL CHECK: Prevent conflicting allegiances
                            // Don't ally A and C if A is at war with someone C is allied with (or vice versa)
                            let has_conflicting_allegiances = {
                                let mut conflict = false;
                                
                                // Check if A is allied with any of C's enemies
                                if let Some(c_enemies) = self.enemies.get(&empire_c) {
                                    if let Some(a_allies) = self.allies.get(&empire_a) {
                                        for &c_enemy in c_enemies {
                                            if a_allies.contains(&c_enemy) {
                                                conflict = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                                
                                // Check if C is allied with any of A's enemies
                                if !conflict {
                                    if let Some(a_enemies) = self.enemies.get(&empire_a) {
                                        if let Some(c_allies) = self.allies.get(&empire_c) {
                                            for &a_enemy in a_enemies {
                                                if c_allies.contains(&a_enemy) {
                                                    conflict = true;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                                
                                conflict
                            };
                            
                            // Only proceed if no conflicting allegiances
                            if !has_conflicting_allegiances {
                                // Boost scales with how much they both hate the common enemy
                                // REDUCED but buffed to encourage more alliance formation
                                let mutual_hatred = (relation_ab.abs() + relation_cb.abs()) / 2.0;
                                
                                // Lower threshold - empires with moderate mutual hatred can ally
                                if mutual_hatred > 0.6 { // Reduced from 0.9 - easier alliance formation
                                    let boost = (mutual_hatred * 0.004).min(0.008); // Buffed from 0.003/0.006
                                    alliance_boosts.push((empire_a, empire_c, boost));
                                }
                            }
                        }
                    }
                }
            }
        }
        
        let empires_with_enemies = self.enemies.len();
        println!("Common enemy scan: {} empires with enemies, {} mutual enemy situations, {} alliance boosts", 
                 empires_with_enemies, mutual_enemies_found, alliance_boosts.len());
        
        // Apply alliance boosts using dampened relation changes
        for (a, c, boost) in alliance_boosts {
            let personality_diff = personality_diff_fn(a, c);
            let personality_similarity = 1.0 - personality_diff;
            self.apply_relation_change(a, c, boost, personality_similarity);
        }
        
        // ALLIANCE DECAY: Integrated into this function to reuse enemy lookups
        // Process alliance decay for all allied pairs
        for (&empire_a, allies_of_a) in &self.allies {
            for &empire_b in allies_of_a {
                if empire_a >= empire_b { continue; } // Only process each pair once
                
                // OPTIMIZED: Reuse enemy lookups we already have
                let enemies_of_a = self.enemies.get(&empire_a);
                let enemies_of_b = self.enemies.get(&empire_b);
                
                // Fast shared enemy count using iterators
                let shared_enemies = match (enemies_of_a, enemies_of_b) {
                    (Some(ea), Some(eb)) => {
                        // Use a HashSet for O(1) lookups instead of O(n) contains
                        let eb_set: std::collections::HashSet<_> = eb.iter().collect();
                        ea.iter().filter(|e| eb_set.contains(e)).count()
                    }
                    _ => 0,
                };
                
                let current_relation = self.get_relation(empire_a, empire_b);
                
                // AGGRESSIVE alliance cleanup - alliances decay quickly without shared purpose
                // Calculate decay based on shared enemies
                let decay_strength = if shared_enemies == 0 {
                    // No shared enemies - alliance losing purpose, decay VERY aggressively
                    if current_relation > 0.8 {
                        -0.08  // Doubled from -0.05 for very strong alliances
                    } else if current_relation > 0.5 {
                        -0.15  // Increased from -0.10 - faster decay
                    } else {
                        -0.25  // Increased from -0.15 - extremely fast decay for weak alliances
                    }
                } else if shared_enemies == 1 {
                    -0.08  // Doubled from -0.04 - much stronger decay even with one shared enemy
                } else if shared_enemies == 2 {
                    -0.03  // Tripled from -0.01 - significant decay even with 2 enemies
                } else {
                    -0.01  // NEW: Even 3+ shared enemies see some decay
                };
                
                if decay_strength < 0.0 {
                    let personality_diff = personality_diff_fn(empire_a, empire_b);
                    let personality_similarity = 1.0 - personality_diff;
                    alliance_decay_changes.push((empire_a, empire_b, decay_strength, personality_similarity, shared_enemies));
                }
                
                // ADDITIONAL: Former allies with different personalities develop tension
                if shared_enemies == 0 && current_relation < 0.4 {
                    let personality_diff = personality_diff_fn(empire_a, empire_b);
                    if personality_diff > 0.5 { // Lowered threshold from 0.6
                        let tension = -0.02 * personality_diff; // Doubled from -0.01
                        let personality_similarity = 1.0 - personality_diff;
                        alliance_decay_changes.push((empire_a, empire_b, tension, personality_similarity, shared_enemies));
                    }
                }
            }
        }
        
        if !alliance_decay_changes.is_empty() {
            let total_decaying = alliance_decay_changes.len();
            let no_shared_enemies = alliance_decay_changes.iter()
                .filter(|(_, _, _, _, shared)| *shared == 0)
                .count();
            
            println!("Alliance decay: {} alliances weakening ({} with no shared enemies)", 
                     total_decaying, no_shared_enemies);
        }
        
        // Apply alliance decay
        for (a, b, decay, personality_similarity, _) in alliance_decay_changes {
            self.apply_relation_change(a, b, decay, personality_similarity);
        }
    }
    
    /// Break up conflicting alliances - empires allied to both sides of a war
    /// When A and B are at war, if C is allied to both, we need to force C to pick a side
    fn break_conflicting_alliances<F>(&mut self, personality_diff_fn: &F)
    where
        F: Fn(u16, u16) -> f32,
    {
        let mut conflicts_found = 0;
        let mut alliance_breaks = Vec::new();
        
        // Check all enemy pairs
        for (&empire_a, enemies_of_a) in &self.enemies {
            for &empire_b in enemies_of_a {
                if empire_a >= empire_b { continue; } // Process each pair once
                
                // A and B are at war. Check if any empire is allied to both
                if let Some(allies_of_a) = self.allies.get(&empire_a) {
                    if let Some(allies_of_b) = self.allies.get(&empire_b) {
                        // Find empires allied to both A and B
                        for &potential_conflict in allies_of_a {
                            if allies_of_b.contains(&potential_conflict) {
                                // Empire is allied to both sides of a war!
                                conflicts_found += 1;
                                
                                // Force them to pick a side based on:
                                // 1. Stronger relation
                                // 2. Personality similarity
                                let relation_to_a = self.get_relation(potential_conflict, empire_a);
                                let relation_to_b = self.get_relation(potential_conflict, empire_b);
                                
                                let personality_sim_a = 1.0 - personality_diff_fn(potential_conflict, empire_a);
                                let personality_sim_b = 1.0 - personality_diff_fn(potential_conflict, empire_b);
                                
                                // Score = relation * 0.7 + personality * 0.3
                                let score_a = relation_to_a * 0.7 + personality_sim_a * 0.3;
                                let score_b = relation_to_b * 0.7 + personality_sim_b * 0.3;
                                
                                // Break alliance with the weaker side
                                let break_with = if score_a > score_b {
                                    empire_b
                                } else {
                                    empire_a
                                };
                                
                                // Apply strong negative penalty to break the alliance
                                // This should drop relation below the ally threshold (0.65)
                                let personality_similarity = 1.0 - personality_diff_fn(potential_conflict, break_with);
                                alliance_breaks.push((potential_conflict, break_with, -0.4, personality_similarity));
                            }
                        }
                    }
                }
            }
        }
        
        if conflicts_found > 0 {
            println!("Conflicting alliances: {} conflicts detected, breaking {} alliances", 
                     conflicts_found, alliance_breaks.len());
        }
        
        // Apply alliance breaks
        for (a, b, penalty, personality_similarity) in alliance_breaks {
            self.apply_relation_change(a, b, penalty, personality_similarity);
        }
    }
    
    /// Apply war weariness for stagnant conflicts
    /// When empires are at war but don't interact for many cycles, relations drift toward neutral
    /// Uses last_interaction tracker to detect when enemies have stopped fighting
    /// This helps end unproductive stalemates and allows former enemies to normalize relations
    fn apply_war_weariness<F>(&mut self, _territory_sizes: &HashMap<u16, u32>, personality_diff_fn: F)
    where
        F: Fn(u16, u16) -> f32,
    {
        // OPTIMIZATION: More aggressive war cleanup - reduced from 60 to 30 frames
        const NO_INTERACTION_THRESHOLD: u32 = 30; // Reduced from 60 (~0.5 second at 60fps)
        const WEARINESS_BASE: f32 = 0.016;          // Increased from 0.008 - faster drift
        const WEARINESS_MAX: f32 = 0.08;            // Increased from 0.04 - faster max drift
        
        let mut weariness_changes = Vec::new();
        
        // Check all enemy pairs for lack of interaction
        for (&empire_a, enemies_list) in &self.enemies {
            for &empire_b in enemies_list {
                if empire_a >= empire_b { continue; } // Process each pair once
                
                let key = (empire_a, empire_b);
                
                // Check how long since last interaction
                let frames_since_interaction = self.last_interaction.get(&key).copied().unwrap_or(0);
                
                // Apply war weariness if no interaction for a while
                if frames_since_interaction >= NO_INTERACTION_THRESHOLD {
                    let relation = self.get_relation(empire_a, empire_b);
                    
                    // Only apply weariness to hostile relations (< 0)
                    if relation < 0.0 {
                        // Weariness strength scales with how long there's been no interaction
                        let extra_time = (frames_since_interaction - NO_INTERACTION_THRESHOLD) as f32;
                        let weariness_strength = (WEARINESS_BASE + extra_time * 0.0005).min(WEARINESS_MAX);
                        
                        // Drift toward neutral (positive change for negative relations)
                        let personality_diff = personality_diff_fn(empire_a, empire_b);
                        let personality_similarity = 1.0 - personality_diff;
                        
                        weariness_changes.push((empire_a, empire_b, weariness_strength, personality_similarity, frames_since_interaction));
                    }
                }
            }
        }
        
        // Clean up interaction tracking for pairs that are no longer enemies
        self.last_interaction.retain(|&(a, b), _| {
            // Keep if either is an enemy of the other
            let a_enemies = self.enemies.get(&a);
            let b_enemies = self.enemies.get(&b);
            
            a_enemies.map_or(false, |enemies| enemies.contains(&b)) ||
            b_enemies.map_or(false, |enemies| enemies.contains(&a))
        });
        
        if !weariness_changes.is_empty() {
            let avg_no_interaction: f32 = weariness_changes.iter()
                .map(|(_, _, _, _, frames)| *frames as f32)
                .sum::<f32>() / weariness_changes.len() as f32;
            
            println!("War weariness: {} inactive conflicts normalizing (avg {} frames without interaction)", 
                     weariness_changes.len(), avg_no_interaction as u32);
        }
        
        // Apply war weariness
        for (a, b, weariness, personality_similarity, _) in weariness_changes {
            self.apply_relation_change(a, b, weariness, personality_similarity);
        }
    }
    
    /// Convert relations to flat f32 array for GPU upload (2048×2048)
    /// Reuses cached buffer to avoid repeated 16MB allocations
    /// OPTIMIZED: Parallel clear and update
    pub fn to_buffer(&mut self) -> &[f32] {
        // Fast parallel clear
        use rayon::prelude::*;
        self.gpu_buffer.par_chunks_mut(8192).for_each(|chunk| {
            chunk.fill(0.0);
        });
        
        // Update non-neutral relations (sparse - only a small fraction)
        for ((a, b), relation) in &self.relations {
            let idx_ab = (*a as usize - 1) * 2048 + (*b as usize - 1);
            let idx_ba = (*b as usize - 1) * 2048 + (*a as usize - 1);
            self.gpu_buffer[idx_ab] = *relation;
            self.gpu_buffer[idx_ba] = *relation;
        }
        
        // Set self-relations to 1.0 (sequential is fine for 2048 writes)
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
