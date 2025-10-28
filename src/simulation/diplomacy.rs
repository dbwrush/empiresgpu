use std::collections::HashMap;

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
}

impl DiplomacyState {
    pub fn new() -> Self {
        let mut state = Self {
            relations: HashMap::new(),
            last_territory_sizes: HashMap::new(),
            suppress_logging: true, // Suppress during initialization
        };
        state.initialize_neutral_relations();
        state.suppress_logging = false; // Re-enable after init
        state
    }
    
    /// Initialize all relations to neutral (0.0)
    fn initialize_neutral_relations(&mut self) {
        // Initialize relations for all possible empire pairs (1-255)
        for a in 1u16..256 {
            for b in (a + 1)..256 {
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
            const HOSTILE_THRESHOLD: f32 = -0.3;
            const ALLIED_THRESHOLD: f32 = 0.3;
            
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
    }
    
    /// Process diplomatic counters from GPU and update relations
    /// counters: flat array of u32 [attack_count, attack_strength, reinforce_count, reinforce_strength, ...]
    /// for each empire pair (256×256 = 65536 pairs × 4 values = 262144 u32s)
    pub fn process_counters(&mut self, counters: &[u32], territory_sizes: &HashMap<u16, u32>) {
        // Process each empire pair
        for row in 0..256u16 {
            for col in (row + 1)..256u16 { // Only process upper triangle (symmetric)
                let idx = (row as usize) * 256 + (col as usize);
                let base = idx * 4; // 4 counters per pair
                
                if base + 3 >= counters.len() {
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
                
                // Apply diplomatic rules
                self.apply_rules(
                    empire_a,
                    empire_b,
                    attack_count,
                    attack_strength,
                    reinforce_count,
                    reinforce_strength,
                    territory_sizes,
                );
            }
        }
        
        // Update territory tracking
        self.last_territory_sizes = territory_sizes.clone();
    }
    
    fn apply_rules(
        &mut self,
        a: u16,
        b: u16,
        attack_count: u32,
        attack_strength: u32,
        reinforce_count: u32,
        reinforce_strength: u32,
        territory_sizes: &HashMap<u16, u32>,
    ) {
        let current_relation = self.get_relation(a, b);
        let mut new_relation = current_relation;
        
        // Rule 1: Attacks degrade relations rapidly
        // Philosophy: Direct conflict is the strongest driver of hostility
        if attack_count > 0 {
            // Scale by both frequency (count) and intensity (strength)
            let attack_impact = (attack_count as f32).sqrt() * 0.001 
                              + (attack_strength as f32).sqrt() * 0.0001;
            new_relation -= attack_impact.min(0.02); // Max -0.02 per cycle
            
            // Rule 1a: Allies of victim also dislike attacker (diplomatic ripple effect)
            self.propagate_hostility(a, b, attack_impact * 0.3);
        }
        
        // Rule 2: Reinforcements improve relations (but slower than attacks degrade)
        // Philosophy: Helping builds trust, but trust takes time
        if reinforce_count > 0 {
            // Only count reinforcements if empires aren't already allied
            // (prevents runaway positive feedback)
            if current_relation < 0.8 {
                let reinforce_impact = (reinforce_count as f32).sqrt() * 0.0005
                                     + (reinforce_strength as f32).sqrt() * 0.00005;
                new_relation += reinforce_impact.min(0.01); // Max +0.01 per cycle (slower than attack degradation)
            }
        }
        
        // Rule 3: Mutual non-aggression slowly improves relations
        // Philosophy: Peaceful coexistence breeds familiarity
        if attack_count == 0 && current_relation > -0.5 && current_relation < 0.5 {
            // Neutral/slightly friendly empires become friendlier when not fighting
            new_relation += 0.0002; // Very gradual drift toward alliance
        }
        
        // Rule 4: Desperation and surrender
        // Philosophy: Losing empires may seek peace to survive
        let a_size = territory_sizes.get(&a).copied().unwrap_or(0);
        let b_size = territory_sizes.get(&b).copied().unwrap_or(0);
        let a_last_size = self.last_territory_sizes.get(&a).copied().unwrap_or(a_size);
        let b_last_size = self.last_territory_sizes.get(&b).copied().unwrap_or(b_size);
        
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
        
        // Rule 5: Very slow natural decay toward neutral (forgotten grudges)
        // Philosophy: Over very long periods, old conflicts are forgotten
        let decay_rate = 0.00005; // Extremely slow
        if new_relation > 0.1 {
            new_relation -= decay_rate;
        } else if new_relation < -0.1 {
            new_relation += decay_rate;
        }
        
        self.set_relation(a, b, new_relation);
    }
    
    /// Propagate hostility: allies of victim dislike attacker
    fn propagate_hostility(&mut self, attacker: u16, victim: u16, amount: f32) {
        // Find allies of victim (relation > 0.3)
        let mut victim_allies = Vec::new();
        for ((a, b), relation) in &self.relations {
            if *relation > 0.3 {
                if *a == victim {
                    victim_allies.push(*b);
                } else if *b == victim {
                    victim_allies.push(*a);
                }
            }
        }
        
        // Each ally dislikes the attacker more
        for ally in victim_allies {
            if ally != attacker {
                let current = self.get_relation(attacker, ally);
                self.set_relation(attacker, ally, current - amount);
            }
        }
    }
    
    /// Convert relations to flat f32 array for GPU upload (256×256)
    pub fn to_buffer(&self) -> Vec<f32> {
        let mut buffer = vec![0.0f32; 256 * 256];
        
        for ((a, b), relation) in &self.relations {
            // Store symmetrically
            let idx_ab = (*a as usize - 1) * 256 + (*b as usize - 1);
            let idx_ba = (*b as usize - 1) * 256 + (*a as usize - 1);
            buffer[idx_ab] = *relation;
            buffer[idx_ba] = *relation;
        }
        
        // Self-relations are always 1.0
        for i in 0..256 {
            buffer[i * 256 + i] = 1.0;
        }
        
        buffer
    }
}

impl Default for DiplomacyState {
    fn default() -> Self {
        Self::new()
    }
}
