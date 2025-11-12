use wgpu::util::DeviceExt;
use crate::graphics::{GraphicsContext, load_shader};
use noise::{NoiseFn, Simplex};
use std::collections::HashMap;
use rand::Rng;
use std::sync::{Arc, Mutex};
use crossbeam::channel::{Sender, Receiver, unbounded};

mod diplomacy;
use diplomacy::DiplomacyState;

/// Message sent to diplomacy worker thread
#[derive(Clone)]
struct DiplomacyWorkRequest {
    counters: Vec<u32>,
    personality_diffs: Vec<f32>,
    frame_count: u32,
}

/// Message received from diplomacy worker thread
struct DiplomacyWorkResponse {
    relations_buffer: Vec<f32>,
}

/// Boat entity - managed on CPU, integrates with GPU simulation
#[derive(Debug, Clone)]
struct Boat {
    position: (f32, f32),  // Floating point position for smooth movement
    empire_id: u16,
    strength: u16,
    direction: u8,         // 0-5 for hex directions
    age: u32,              // Frames since spawn
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RenderMode {
    Empires = 0,
    Strength = 1,
    Need = 2,
    Action = 3,
    Age = 4,
    Diplomacy = 5,
    BoatNeed = 6,
}

// Generate terrain data using simplex noise with cylindrical world wrapping
fn generate_terrain_data(size: u32) -> Vec<u8> {
    println!("Generating terrain data with cylindrical world wrapping...");
    
    // Multiple noise layers for varied terrain with random seeds
    let mut rng = rand::rng();
    let seed1: u32 = rng.random();
    let seed2: u32 = rng.random();
    let seed3: u32 = rng.random();
    println!("Terrain seeds: {}, {}, {}", seed1, seed2, seed3);
    
    let noise = Simplex::new(seed1);
    let noise2 = Simplex::new(seed2);
    let noise3 = Simplex::new(seed3);
    
    let mut terrain_data = Vec::with_capacity((size * size * 4) as usize);
    
    // Parameters for world wrapping
    let loop_dist = size / 16; // Distance from edge where wrapping interpolation occurs
    let _ocean_cutoff = 0.53; // Terrain below this becomes ocean (used in shaders)
    
    for y in 0..size {
        for x in 0..size {
            let mut elevation = get_elevation(&noise, &noise2, &noise3, x, y, size);
            
            // Apply cylindrical world wrapping at edges
            if x < loop_dist || x > size - loop_dist {
                let opp_prop = if x < loop_dist {
                    x as f32 / loop_dist as f32 * -0.5 + 0.5
                } else {
                    (size - x - 1) as f32 / loop_dist as f32 * -0.5 + 0.5
                };
                
                let opp_x = size - x - 1;
                let opp_elevation = get_elevation(&noise, &noise2, &noise3, opp_x, y, size);
                elevation = elevation * (1.0 - opp_prop) + opp_elevation * opp_prop;
            }
            
            // Store elevation directly as 0-255, let shader handle ocean cutoff
            let altitude = (elevation * 255.0).clamp(0.0, 255.0) as u8;
            
            // Calculate terrain penalty using smooth functions for need propagation and reinforcements
            // Higher elevation = more difficult terrain = exponentially higher penalty
            // Ocean (below 0.53) has very low penalty for naval movement
            // Mountains (above 0.8) have severe penalties to discourage border gore
            let terrain_penalty = if elevation < 0.53 {
                // Ocean/water - excellent for naval logistics and need propagation
                0.05
            } else {
                // Land elevation penalty using smooth exponential curve
                // Normalize land elevation from 0.53-1.0 to 0.0-1.0 range
                let land_elevation = (elevation - 0.53) / (1.0 - 0.53);
                
                // Exponential penalty curve: starts low for plains, becomes severe for mountains
                // At plains (land_elevation=0.0): penalty = 0.15 (slight penalty)
                // At mid-hills (land_elevation=0.5): penalty = 0.45 (moderate penalty) 
                // At high mountains (land_elevation=1.0): penalty = 0.95 (severe penalty)
                let base_penalty = 0.15;
                let mountain_penalty = 0.95;
                let curve_steepness = 2.5; // Controls how quickly penalty increases with elevation
                
                base_penalty + (mountain_penalty - base_penalty) * land_elevation.powf(curve_steepness)
            };
            
            let penalty_u8 = (terrain_penalty * 255.0f32).clamp(0.0f32, 255.0f32) as u8;
            
            // Store as RGBA: R=altitude, G=unused, B=unused, A=terrain_penalty
            terrain_data.push(altitude);
            terrain_data.push(0u8); // Unused channel (humidity in future)
            terrain_data.push(0u8); // Unused channel (temperature in future)
            terrain_data.push(penalty_u8); // Terrain penalty for need propagation
        }
    }
    
    println!("Terrain generation complete with cylindrical wrapping!");
    terrain_data
}

fn get_elevation(noise: &Simplex, noise2: &Simplex, noise3: &Simplex, x: u32, y: u32, _size: u32) -> f32 {
    let mut x_coord = x as f32;
    
    // Apply hex grid offset for odd rows
    if y % 2 == 1 {
        x_coord += 0.5;
    }
    
    // Multiple octaves of noise for detailed terrain
    let mut e = noise.get([x_coord as f64 / 128.0, y as f64 / 128.0]) as f32 * 16.0 + 
                noise.get([x_coord as f64 / 64.0, y as f64 / 64.0]) as f32 * 8.0 + 
                noise2.get([x_coord as f64 / 32.0, y as f64 / 32.0]) as f32 * 4.0 + 
                noise3.get([x_coord as f64 / 16.0, y as f64 / 16.0]) as f32 * 2.0 + 
                noise3.get([x_coord as f64 / 8.0, y as f64 / 8.0]) as f32;
    
    e /= 64.0; // Normalize
    e += 0.5;  // Shift to 0-1 range
    
    e.clamp(0.0, 1.0)
}

pub struct EmpireSimulation {
    pub texture_a: wgpu::Texture,
    pub texture_b: wgpu::Texture,
    pub terrain_texture: wgpu::Texture,  // Read-only terrain data (R=altitude, G=unused, B=unused, A=terrain_penalty)
    pub empire_params_texture: wgpu::Texture,  // Read-only per-empire parameters (aggression, diplomacy, etc.)
    pub aux_texture_a: wgpu::Texture,    // Auxiliary data texture A (currently: age tracking, future: other dynamic data)
    pub aux_texture_b: wgpu::Texture,    // Auxiliary data texture B (ping-pong with A)
    
    // Diplomacy system buffers (declared before bind groups that use them for proper drop order)
    diplomacy_counters_buffer: wgpu::Buffer,  // Atomic counters for diplomatic events (2048×2048×5 u32s)
    diplomacy_staging_buffer: wgpu::Buffer,   // Staging buffer for reading counter data from GPU
    diplomacy_relations_buffer: wgpu::Buffer, // Current relations matrix (2048×2048 f32s)
    perspective_empire_buffer: wgpu::Buffer,  // Uniform buffer for perspective empire
    
    // Reusable buffer for zeroing counters (allocated once, ~83MB)
    zero_counters_buffer: Vec<u32>,
    
    // Boat system
    boats: Vec<Boat>,                       // Active boats (CPU-managed)
    boat_landings_buffer: wgpu::Buffer,     // GPU buffer for boat landings
    boat_landings_cpu: Vec<u32>,            // CPU staging for boat landings
    boat_spawns_buffer: wgpu::Buffer,       // GPU buffer for boat spawns from cells
    boat_spawns_staging_buffer: wgpu::Buffer, // CPU staging for reading boat spawns
    terrain_data: Vec<u8>,                  // Cached terrain for collision detection
    empire_ownership: Vec<u16>,             // Simplified CPU-side empire ownership map (updated periodically)
    empire_readback_buffer: wgpu::Buffer,   // Staging buffer for reading empire data from GPU
    empire_pending_readback: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,  // Pending empire readback
    
    pub compute_bind_group_a: wgpu::BindGroup,
    pub compute_bind_group_b: wgpu::BindGroup,
    pub display_bind_group_a: wgpu::BindGroup,
    pub display_bind_group_b: wgpu::BindGroup,
    pub aux_bind_group_a: wgpu::BindGroup,       // Bind group for auxiliary texture A rendering
    pub aux_bind_group_b: wgpu::BindGroup,       // Bind group for auxiliary texture B rendering
    pub diplomacy_bind_group_a: wgpu::BindGroup,  // Bind group for diplomacy view A
    pub diplomacy_bind_group_b: wgpu::BindGroup,  // Bind group for diplomacy view B
    pub terrain_bind_group: wgpu::BindGroup,     // For terrain rendering
    pub compute_pipeline: wgpu::ComputePipeline,
    pub render_pipeline_empires: wgpu::RenderPipeline,
    pub render_pipeline_strength: wgpu::RenderPipeline,
    pub render_pipeline_need: wgpu::RenderPipeline,
    pub render_pipeline_action: wgpu::RenderPipeline,
    pub render_pipeline_age: wgpu::RenderPipeline,
    pub render_pipeline_diplomacy: wgpu::RenderPipeline,
    pub render_pipeline_boat_need: wgpu::RenderPipeline,
    pub terrain_pipeline: wgpu::RenderPipeline,  // For terrain rendering
    pub boat_pipeline: wgpu::RenderPipeline,     // For boat rendering
    pub boat_instance_buffer: wgpu::Buffer,      // Instance buffer for boat rendering
    pub current_render_mode: RenderMode,
    pub current_is_a: bool,
    pub frame_count: u32,
    pub simulation_speed: u32,
    pub is_paused: bool,
    pub game_size: u32,
    pub frame_uniform_buffer: wgpu::Buffer,
    pub num_empires: u16,  // Track number of empires for parameters texture size (16-bit for 65535 empires)
    pub perspective_empire: u16,  // Which empire to view diplomacy from (0 = none selected)
    
    // Empire personality traits (stored CPU-side for quick access)
    empire_personalities: std::collections::HashMap<u16, EmpirePersonality>,
    // Cached personality difference matrix (2048×2048 normalized differences, computed once)
    personality_diff_cache: Vec<f32>,
    
    // Diplomacy state
    diplomacy_state: DiplomacyState,          // CPU-side diplomacy processor
    diplomacy_pending_map: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>, // Async map receiver
    
    // Diplomacy worker thread communication
    diplomacy_work_sender: Sender<DiplomacyWorkRequest>,   // Send work to background thread
    diplomacy_result_receiver: Receiver<DiplomacyWorkResponse>,  // Receive results from background thread
    diplomacy_processing: bool,  // Track if diplomacy is currently being processed
    
    // Boat spawns async readback
    boat_spawns_pending_map: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

/// Empire personality traits (0-255 for each)
#[derive(Debug, Clone, Copy)]
struct EmpirePersonality {
    aggression: u8,   // 0=peaceful, 255=warlike
    expansion: u8,    // 0=defensive/isolationist, 255=expansionist/colonial
    cooperation: u8,  // 0=selfish/competitive, 255=cooperative/altruistic
}

impl EmpireSimulation {
    pub fn new(graphics: &GraphicsContext, camera_bind_group_layout: &wgpu::BindGroupLayout, requested_size: u32) -> Self {
        // Check GPU texture size limits and constrain if necessary
        let max_texture_size = graphics.device.limits().max_texture_dimension_2d;
        let game_size = if requested_size > max_texture_size {
            println!("WARNING: Requested simulation size {}x{} exceeds GPU limit of {}x{}", 
                requested_size, requested_size, max_texture_size, max_texture_size);
            println!("         Constraining to maximum supported size: {}x{}", max_texture_size, max_texture_size);
            println!("         For larger simulations, consider using storage buffers instead of textures.");
            max_texture_size
        } else {
            requested_size
        };
        
        // Generate terrain data first
        let terrain_data = generate_terrain_data(game_size);
        
        println!("Creating Empire simulation textures with size {}x{}...", game_size, game_size);
        
        // Generate initial empire map with random empires on land cells
        let mut initial_data = vec![0u16; (game_size * game_size * 4) as usize]; // 16-bit values
        let mut empire_id_counter = 1u16; // Start from 1 since 0 means unclaimed (16-bit now)
        
        // Use a simple RNG for spawning
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        // Generate random seeds for empire initialization
        let mut rng = rand::rng();
        let shuffle_seed: u32 = rng.random();
        let spawn_seed: u32 = rng.random();
        println!("Empire initialization seeds - shuffle: {}, spawn: {}", shuffle_seed, spawn_seed);
        
        // First pass: collect all valid land positions
        let mut land_positions = Vec::new();
        for y in 0..game_size {
            for x in 0..game_size {
                let terrain_idx = ((y * game_size + x) * 4) as usize;
                let elevation = terrain_data[terrain_idx] as f32 / 255.0;
                let is_land = elevation > 0.53; // Same ocean cutoff as terrain generation
                
                if is_land {
                    land_positions.push((x, y));
                }
            }
        }
        
        // Shuffle the land positions using a deterministic shuffle based on hashing
        for i in (1..land_positions.len()).rev() {
            let mut hasher = DefaultHasher::new();
            (i, shuffle_seed).hash(&mut hasher); // Seed for shuffle consistency
            let j = (hasher.finish() as usize) % (i + 1);
            land_positions.swap(i, j);
        }
        
        // Spawn empires on shuffled land positions
        let empire_spawn_chance = 0.005; // 0.5% chance per land cell
        let max_empires = 2048u16; // Hard limit for diplomacy system
        for (x, y) in &land_positions {
            // Stop spawning if we've hit the maximum empire limit
            if empire_id_counter > max_empires {
                println!("  -> Reached maximum empire limit ({}), stopping spawn", max_empires);
                break;
            }
            
            let mut hasher = DefaultHasher::new();
            (*x, *y, spawn_seed).hash(&mut hasher);
            let random_val = (hasher.finish() % 1000) as f32 / 1000.0;
            
            if random_val < empire_spawn_chance {
                let idx = ((y * game_size + x) * 4) as usize;
                initial_data[idx] = empire_id_counter;     // R: Empire ID
                initial_data[idx + 1] = 10000u16;          // G: Strength (scaled up from 200 to ~10000 for 16-bit)
                println!("  -> Spawning Empire {} at ({}, {})", empire_id_counter, x, y);
                empire_id_counter += 1;
            }
        }
        
        println!("  -> Spawned {} empires total", empire_id_counter - 1);
        
        // Create two identical textures for ping-pong
        let texture_desc = wgpu::TextureDescriptor {
            label: Some("Empire Simulation Texture"),
            size: wgpu::Extent3d {
                width: game_size,
                height: game_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm, // 16-bit for extended range (0-65535)
            usage: wgpu::TextureUsages::TEXTURE_BINDING 
                | wgpu::TextureUsages::STORAGE_BINDING 
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC, // Allow reading back from GPU
            view_formats: &[],
        };
        
        // Convert u16 data to bytes for texture upload
        let initial_data_bytes: Vec<u8> = initial_data.iter()
            .flat_map(|&value| value.to_ne_bytes())
            .collect();
        
        let texture_a = graphics.device.create_texture_with_data(
            &graphics.queue,
            &texture_desc,
            wgpu::util::TextureDataOrder::LayerMajor,
            &initial_data_bytes,
        );
        
        let texture_b = graphics.device.create_texture(&texture_desc);
        
        // Create terrain texture (read-only) using previously generated data
        let terrain_texture_desc = wgpu::TextureDescriptor {
            label: Some("Terrain Texture"),
            size: wgpu::Extent3d {
                width: game_size,
                height: game_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm, // Use RGBA format, ignore alpha channel
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };
        
        let terrain_texture = graphics.device.create_texture_with_data(
            &graphics.queue,
            &terrain_texture_desc,
            wgpu::util::TextureDataOrder::LayerMajor,
            &terrain_data,
        );
        
        // Create empire parameters texture (NxN where N is number of empires)
        // Each pixel contains parameters for one empire vs all empires
        // Maximum of 2048 empires supported for diplomacy and personality systems
        let max_empires = 2048u32;
        let num_spawned_empires = (empire_id_counter - 1) as u16;
        
        // Hard limit check - program cannot run with more empires than diplomacy system supports
        if num_spawned_empires > max_empires as u16 {
            panic!("CRITICAL: {} empires spawned, but diplomacy system only supports {}. Reduce empire count or increase MAX_EMPIRES.", 
                   num_spawned_empires, max_empires);
        }
        
        println!("Creating empire parameters texture ({}x{}) for {} empires...", max_empires, max_empires, num_spawned_empires);
        
        // Generate random seeds for empire personality traits
        let diplomacy_seed: u32 = rng.random();
        let aggression_seed: u32 = rng.random();
        let expansion_seed: u32 = rng.random();
        let cooperation_seed: u32 = rng.random();
        println!("Empire personality seeds - diplomacy: {}, aggression: {}, expansion: {}, cooperation: {}", 
                 diplomacy_seed, aggression_seed, expansion_seed, cooperation_seed);
        
        // Generate empire parameters data
        // Channel layout: R=diplomacy (opinion of other empire, 0-255), G=aggression (0-255), 
        // B=expansion (0=defensive/isolationist, 255=expansionist/colonial), 
        // A=cooperation (0=selfish/competitive, 255=cooperative/altruistic)
        let mut empire_params_data = Vec::with_capacity((max_empires * max_empires * 4) as usize);
        
        // Initialize the parameters texture
        for y in 0..max_empires {
            for x in 0..max_empires {
                let empire_id = (y + 1) as u16; // Empire IDs start from 1 (16-bit now)
                let target_empire_id = (x + 1) as u16;
                
                let (diplomacy, aggression, expansion, cooperation) = if empire_id <= num_spawned_empires {
                    // Generate random diplomacy opinion for this empire vs target empire
                    let mut hasher = DefaultHasher::new();
                    (empire_id, target_empire_id, diplomacy_seed).hash(&mut hasher); // Seed for consistency
                    let diplomacy_random = (hasher.finish() % 256) as u8;
                    
                    // Generate random aggression for this empire (same for all targets)
                    let mut hasher2 = DefaultHasher::new();
                    (empire_id, aggression_seed).hash(&mut hasher2);
                    let aggression_random = (hasher2.finish() % 256) as u8;
                    
                    // Generate random expansion trait (0=defensive, 255=expansionist)
                    let mut hasher3 = DefaultHasher::new();
                    (empire_id, expansion_seed).hash(&mut hasher3);
                    let expansion_random = (hasher3.finish() % 256) as u8;
                    
                    // Generate random cooperation trait (0=selfish, 255=cooperative)
                    let mut hasher4 = DefaultHasher::new();
                    (empire_id, cooperation_seed).hash(&mut hasher4);
                    let cooperation_random = (hasher4.finish() % 256) as u8;
                    
                    (diplomacy_random, aggression_random, expansion_random, cooperation_random)
                } else {
                    (128u8, 128u8, 128u8, 128u8) // Neutral values for unused empire slots
                };
                
                empire_params_data.extend_from_slice(&[
                    diplomacy,    // R: Diplomacy/opinion (not used yet, but ready for future)
                    aggression,   // G: Aggression level (affects combat threshold)
                    expansion,    // B: Expansion drive (0=defensive, 255=expansionist)
                    cooperation,  // A: Cooperation level (0=selfish, 255=cooperative)
                ]);
            }
        }
        
        // Build empire personalities HashMap for CPU-side access
        let mut empire_personalities = std::collections::HashMap::new();
        for empire_id in 1..=num_spawned_empires {
            // Extract personality from the same data we generated
            // The texture is organized as: for each row (empire), there are max_empires columns (targets)
            // Row index = empire_id - 1, and we can use any column (traits are same for all columns)
            // We'll use column 0 for simplicity
            let row = (empire_id - 1) as usize;
            let col = 0usize;
            let idx = (row * (max_empires as usize) + col) * 4; // 4 bytes per pixel (RGBA)
            
            let aggression = empire_params_data[idx + 1];   // G channel
            let expansion = empire_params_data[idx + 2];    // B channel
            let cooperation = empire_params_data[idx + 3];  // A channel
            
            empire_personalities.insert(empire_id, EmpirePersonality {
                aggression,
                expansion,
                cooperation,
            });
        }
        
        let empire_params_texture_desc = wgpu::TextureDescriptor {
            label: Some("Empire Parameters Texture"),
            size: wgpu::Extent3d {
                width: max_empires,
                height: max_empires,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };
        
        let empire_params_texture = graphics.device.create_texture_with_data(
            &graphics.queue,
            &empire_params_texture_desc,
            wgpu::util::TextureDataOrder::LayerMajor,
            &empire_params_data,
        );
        
        // Create auxiliary textures for dynamic data (age tracking, etc.)
        // Use RGBA16Unorm format for better precision (matches main simulation texture)
        println!("Creating auxiliary textures for age tracking and future dynamic data...");
        
        let aux_texture_desc = wgpu::TextureDescriptor {
            label: Some("Auxiliary Data Texture"),
            size: wgpu::Extent3d {
                width: game_size,
                height: game_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm, // 16-bit per channel for better precision
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };
        
        // Initialize auxiliary data (all zeros = age 0)
        let aux_data = vec![0u8; (game_size * game_size * 8) as usize]; // 8 bytes per pixel for RGBA16
        
        let aux_texture_a = graphics.device.create_texture_with_data(
            &graphics.queue,
            &aux_texture_desc,
            wgpu::util::TextureDataOrder::LayerMajor,
            &aux_data,
        );
        
        // Initialize aux_texture_b with the same data (both textures must start with same state)
        let aux_texture_b = graphics.device.create_texture_with_data(
            &graphics.queue,
            &aux_texture_desc,
            wgpu::util::TextureDataOrder::LayerMajor,
            &aux_data,
        );
        
        // Create texture views
        let game_view_a = texture_a.create_view(&wgpu::TextureViewDescriptor::default());
        let game_view_b = texture_b.create_view(&wgpu::TextureViewDescriptor::default());
        let terrain_view = terrain_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let empire_params_view = empire_params_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let aux_view_a = aux_texture_a.create_view(&wgpu::TextureViewDescriptor::default());
        let aux_view_b = aux_texture_b.create_view(&wgpu::TextureViewDescriptor::default());
        
        let texture_sampler = graphics.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        // Create frame counter uniform buffer
        let frame_uniform_buffer = graphics.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Frame Counter Uniform Buffer"),
            size: 4, // u32 = 4 bytes
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create diplomacy buffers
        // Counters: 2048×2048 empire pairs × 5 u32s per pair = 20,971,520 u32s = 83,886,080 bytes (~80 MB)
        let diplomacy_counters_buffer = graphics.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Diplomacy Counters Buffer"),
            size: (2048 * 2048 * 5 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Staging buffer for reading back counter data from GPU
        let diplomacy_staging_buffer = graphics.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Diplomacy Staging Buffer"),
            size: (2048 * 2048 * 5 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Relations: 2048×2048 f32 values = 16,777,216 bytes (~16 MB)
        let diplomacy_relations_buffer = graphics.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Diplomacy Relations Buffer"),
            size: (2048 * 2048 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Perspective empire for diplomacy view (u32)
        let perspective_empire_buffer = graphics.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Perspective Empire Buffer"),
            contents: bytemuck::cast_slice(&[0u32]), // Start with no empire selected
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        // Initialize counters to zero
        graphics.queue.write_buffer(
            &diplomacy_counters_buffer,
            0,
            bytemuck::cast_slice(&vec![0u32; 2048 * 2048 * 5]),
        );
        
        // Initialize relations to zero (neutral)
        graphics.queue.write_buffer(
            &diplomacy_relations_buffer,
            0,
            bytemuck::cast_slice(&vec![0.0f32; 2048 * 2048]),
        );
        
        // Create boat landing buffer (one u32 per cell for boat attacks)
        // Format: upper 16 bits = empire_id, lower 16 bits = strength
        let boat_landings_buffer = graphics.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Boat Landings Buffer"),
            size: ((game_size * game_size) as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Initialize boat landings to zero
        graphics.queue.write_buffer(
            &boat_landings_buffer,
            0,
            bytemuck::cast_slice(&vec![0u32; (game_size * game_size) as usize]),
        );
        
        // Create boat spawns buffer (one u32 per cell for GPU-initiated boat spawns)
        // Format: upper 16 bits = empire_id, lower 16 bits = direction (0-5)
        let boat_spawns_buffer = graphics.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Boat Spawns Buffer"),
            size: ((game_size * game_size) as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Create staging buffer for reading boat spawns from GPU
        let boat_spawns_staging_buffer = graphics.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Boat Spawns Staging Buffer"),
            size: ((game_size * game_size) as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create staging buffer for reading empire ownership from GPU
        // RGBA16Unorm texture: each pixel = 4 u16s, we only need R channel for empire_id
        let empire_readback_buffer = graphics.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Empire Readback Buffer"),
            size: (game_size * game_size * 8) as u64,  // 4 channels * 2 bytes per u16
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create compute bind group layout
        let compute_bind_group_layout = graphics.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Empire Simulation Compute Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Unorm, // Match shader 16-bit format
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 9: Boat landings buffer (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 10: Boat spawns buffer (read-write storage with atomics)
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Load compute shader
        let compute_shader = load_shader(&graphics.device, "shaders/empire_compute.wgsl");
        
        // Create compute pipeline
        let compute_pipeline = graphics.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Empire Simulation Compute Pipeline"),
            layout: Some(&graphics.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Empire Simulation Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        // Create compute bind groups for ping-pong (A -> B and B -> A)
        let compute_bind_group_a = graphics.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Game Bind Group A->B"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&game_view_a),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&game_view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: frame_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&terrain_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&empire_params_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&aux_view_a),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&aux_view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: diplomacy_counters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: diplomacy_relations_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: boat_landings_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: boat_spawns_buffer.as_entire_binding(),
                },
            ],
        });
        
        let compute_bind_group_b = graphics.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Game Bind Group B->A"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&game_view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&game_view_a),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: frame_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&terrain_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&empire_params_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&aux_view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&aux_view_a),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: diplomacy_counters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: diplomacy_relations_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: boat_landings_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: boat_spawns_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Create display bind group layout (for rendering to screen)
        let display_bind_group_layout = graphics.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Display Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        
        // Create display bind groups
        let display_bind_group_a = graphics.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Display Bind Group A"),
            layout: &display_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&game_view_a),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
            ],
        });
        
        let display_bind_group_b = graphics.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Display Bind Group B"),
            layout: &display_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&game_view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
            ],
        });
        
        // Create auxiliary bind groups for rendering auxiliary data (age, etc.)
        let aux_bind_group_a = graphics.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Auxiliary Bind Group A"),
            layout: &display_bind_group_layout, // Reuse same layout as display 
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&aux_view_a),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
            ],
        });
        
        let aux_bind_group_b = graphics.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Auxiliary Bind Group B"),
            layout: &display_bind_group_layout, // Reuse same layout as display
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&aux_view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
            ],
        });
        
        // Create shared pipeline layout for all render modes
        let render_pipeline_layout = graphics.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Empire Simulation Pipeline Layout"),
            bind_group_layouts: &[&display_bind_group_layout, camera_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Helper function to create render pipeline with different shaders
        let create_render_pipeline = |shader_path: &str, label: &str| {
            let shader = load_shader(&graphics.device, shader_path);
            graphics.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[crate::graphics::Vertex::desc()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: graphics.config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            })
        };
        
        // Create render pipelines for different visualization modes
        let render_pipeline_empires = create_render_pipeline("shaders/empire_render.wgsl", "Empire Render Pipeline");
        let render_pipeline_strength = create_render_pipeline("shaders/empire_render_strength.wgsl", "Strength Render Pipeline");
        let render_pipeline_need = create_render_pipeline("shaders/empire_render_need.wgsl", "Need Render Pipeline");
        let render_pipeline_action = create_render_pipeline("shaders/empire_render_action.wgsl", "Action Render Pipeline");
        let render_pipeline_age = create_render_pipeline("shaders/empire_render_age.wgsl", "Age Render Pipeline");
        let render_pipeline_boat_need = create_render_pipeline("shaders/empire_render_boat_need.wgsl", "Boat Need Render Pipeline");
        
        // Create diplomacy perspective pipeline (has different bind group layout)
        let diplomacy_shader = load_shader(&graphics.device, "shaders/empire_render_diplomacy.wgsl");
        
        // Diplomacy bind group layout: texture, sampler, relations buffer, perspective uniform
        let diplomacy_display_layout = graphics.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Diplomacy Display Bind Group Layout"),
            entries: &[
                // Binding 0: Texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // Binding 1: Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Binding 2: Diplomacy relations buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: Perspective empire uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Create diplomacy bind groups (for A and B textures)
        let diplomacy_bind_group_a = graphics.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Diplomacy Bind Group A"),
            layout: &diplomacy_display_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&game_view_a),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: diplomacy_relations_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: perspective_empire_buffer.as_entire_binding(),
                },
            ],
        });
        
        let diplomacy_bind_group_b = graphics.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Diplomacy Bind Group B"),
            layout: &diplomacy_display_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&game_view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: diplomacy_relations_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: perspective_empire_buffer.as_entire_binding(),
                },
            ],
        });
        
        let render_pipeline_diplomacy = graphics.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Diplomacy Render Pipeline"),
            layout: Some(&graphics.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Diplomacy Pipeline Layout"),
                bind_group_layouts: &[&diplomacy_display_layout, camera_bind_group_layout],
                push_constant_ranges: &[],
            })),
            vertex: wgpu::VertexState {
                module: &diplomacy_shader,
                entry_point: Some("vs_main"),
                buffers: &[crate::graphics::Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &diplomacy_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: graphics.config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });
        
        // Create terrain bind group for rendering
        let terrain_bind_group = graphics.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain Bind Group"),
            layout: &display_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&terrain_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
            ],
        });
        
        // Load terrain render shader
        let terrain_shader = load_shader(&graphics.device, "shaders/terrain_render.wgsl");
        
        // Create terrain render pipeline
        let terrain_pipeline = graphics.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Terrain Render Pipeline"),
            layout: Some(&graphics.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Terrain Pipeline Layout"),
                bind_group_layouts: &[&display_bind_group_layout, camera_bind_group_layout],
                push_constant_ranges: &[],
            })),
            vertex: wgpu::VertexState {
                module: &terrain_shader,
                entry_point: Some("vs_main"),
                buffers: &[crate::graphics::Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &terrain_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: graphics.config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });
        
        println!("Empire simulation and terrain pipelines created! Ready to simulate.");
        
        // Create boat render pipeline
        // Boats use a simple layout: only camera bind group (no display textures needed)
        let boat_shader = load_shader(&graphics.device, "shaders/boat_render.wgsl");
        let boat_pipeline = graphics.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Boat Render Pipeline"),
            layout: Some(&graphics.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Boat Pipeline Layout"),
                bind_group_layouts: &[camera_bind_group_layout],  // Only camera at group 0
                push_constant_ranges: &[],
            })),
            vertex: wgpu::VertexState {
                module: &boat_shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<[f32; 5]>() as u64,  // position (2) + color (3)
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x2,  // position
                            },
                            wgpu::VertexAttribute {
                                offset: std::mem::size_of::<[f32; 2]>() as u64,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x3,  // color
                            },
                        ],
                    },
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &boat_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: graphics.config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),  // Need alpha blending for transparency
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,  // No culling for boats
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });
        
        // Create boat instance buffer (max 10,000 boats)
        let boat_instance_buffer = graphics.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Boat Instance Buffer"),
            size: (50000 * std::mem::size_of::<[f32; 5]>()) as u64,  // Support up to 50k boats
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Spawn diplomacy worker thread for async processing
        let (work_sender, work_receiver) = crossbeam::channel::unbounded::<DiplomacyWorkRequest>();
        let (result_sender, result_receiver) = crossbeam::channel::unbounded::<DiplomacyWorkResponse>();
        
        std::thread::spawn(move || {
            let mut diplomacy_state = DiplomacyState::new();
            
            loop {
                // Wait for work request
                match work_receiver.recv() {
                    Ok(request) => {
                        // Process diplomacy on this background thread
                        let territory_sizes = std::collections::HashMap::new();
                        
                        diplomacy_state.process_counters(
                            &request.counters,
                            &territory_sizes,
                            |a, b| {
                                let idx = (a as usize) * 2048 + (b as usize);
                                request.personality_diffs.get(idx).copied().unwrap_or(0.5)
                            },
                            request.frame_count,
                        );
                        
                        diplomacy_state.propagate_attack_reactions(&request.counters, |a, b| {
                            let idx = (a as usize) * 2048 + (b as usize);
                            request.personality_diffs.get(idx).copied().unwrap_or(0.5)
                        });
                        
                        // Send back the updated relations buffer
                        let relations_buffer = diplomacy_state.to_buffer().to_vec();
                        let _ = result_sender.send(DiplomacyWorkResponse {
                            relations_buffer,
                        });
                    }
                    Err(_) => {
                        // Channel closed, exit thread
                        break;
                    }
                }
            }
        });
        
        let mut sim = Self {
            texture_a,
            texture_b,
            terrain_texture,
            empire_params_texture,
            aux_texture_a,
            aux_texture_b,
            compute_bind_group_a,
            compute_bind_group_b,
            display_bind_group_a,
            display_bind_group_b,
            aux_bind_group_a,
            aux_bind_group_b,
            diplomacy_bind_group_a,
            diplomacy_bind_group_b,
            terrain_bind_group,
            compute_pipeline,
            render_pipeline_empires,
            render_pipeline_strength,
            render_pipeline_need,
            render_pipeline_action,
            render_pipeline_age,
            render_pipeline_diplomacy,
            render_pipeline_boat_need,
            terrain_pipeline,
            boat_pipeline,
            boat_instance_buffer,
            current_render_mode: RenderMode::Empires,
            current_is_a: true,
            frame_count: 0,
            simulation_speed: 1, // Update every frame for immediate feedback
            is_paused: false,
            game_size,
            frame_uniform_buffer,
            num_empires: num_spawned_empires,
            perspective_empire: 0,
            empire_personalities,
            personality_diff_cache: vec![0.5f32; 2048 * 2048], // Will be computed after initialization
            diplomacy_counters_buffer,
            diplomacy_staging_buffer,
            diplomacy_relations_buffer,
            perspective_empire_buffer,
            zero_counters_buffer: vec![0u32; 2048 * 2048 * 5],  // Pre-allocate reusable buffer (~83MB)
            boats: Vec::new(),  // No boats initially
            boat_landings_buffer,
            boat_landings_cpu: vec![0u32; (game_size * game_size) as usize],
            boat_spawns_buffer,
            boat_spawns_staging_buffer,
            terrain_data: terrain_data.clone(),  // Keep terrain for boat collision detection
            empire_ownership: vec![0u16; (game_size * game_size) as usize],  // Initially no owners
            empire_readback_buffer,
            empire_pending_readback: None,
            diplomacy_state: DiplomacyState::new(),
            diplomacy_pending_map: None,
            diplomacy_work_sender: work_sender,
            diplomacy_result_receiver: result_receiver,
            diplomacy_processing: false,
            boat_spawns_pending_map: None,
        };
        
        // Pre-compute personality difference cache (static for the entire game)
        // Use Euclidean distance instead of average to make differences more pronounced
        // This way empires need to be similar on ALL axes to be compatible
        for (&a, pa) in &sim.empire_personalities {
            if a > 2047 { continue; }
            for (&b, pb) in &sim.empire_personalities {
                if b > 2047 { continue; }
                let aggression_diff = (pa.aggression as i16 - pb.aggression as i16).abs() as f32;
                let expansion_diff = (pa.expansion as i16 - pb.expansion as i16).abs() as f32;
                let cooperation_diff = (pa.cooperation as i16 - pb.cooperation as i16).abs() as f32;
                
                // Euclidean distance: sqrt(a² + b² + c²)
                // Normalized by sqrt(3 * 255²) = ~441.67 (max possible distance)
                let euclidean_dist = (aggression_diff.powi(2) + expansion_diff.powi(2) + cooperation_diff.powi(2)).sqrt();
                let max_distance = (3.0_f32 * 255.0_f32.powi(2)).sqrt(); // ~441.67
                let normalized = euclidean_dist / max_distance;
                
                let idx = (a as usize) * 2048 + (b as usize);
                sim.personality_diff_cache[idx] = normalized;
            }
        }
        
        // Upload initial random relations to GPU
        let initial_relations = sim.diplomacy_state.to_buffer();
        graphics.queue.write_buffer(
            &sim.diplomacy_relations_buffer,
            0,
            bytemuck::cast_slice(&initial_relations),
        );
        
        sim
    }
    
    // Update boat positions and handle spawning/landing
    fn update_boats(&mut self, graphics: &GraphicsContext) {
        // Helper function for boat rendering: convert hue to RGB
        fn hue_to_rgb(hue: f32) -> (f32, f32, f32) {
            let h = hue / 60.0;
            let x = 1.0 - (h % 2.0 - 1.0).abs();
            
            let (r, g, b) = match h as u32 {
                0 => (1.0, x, 0.0),
                1 => (x, 1.0, 0.0),
                2 => (0.0, 1.0, x),
                3 => (0.0, x, 1.0),
                4 => (x, 0.0, 1.0),
                _ => (1.0, 0.0, x),
            };
            
            (r, g, b)
        }
        
        // Clear boat landings buffer
        self.boat_landings_cpu.fill(0);
        
        let mut landed_count = 0;
        
        // Move existing boats and check for landings
        let mut debug_land_encounters = 0;
        let mut debug_continued_boats = 0;
        
        // Create RNG once for all boats (PERFORMANCE: don't create per boat!)
        let mut rng = rand::rng();
        
        self.boats.retain_mut(|boat| {
            boat.age += 1;
            
            // No age-based despawning - boats should eventually hit land
            
            // Move boat on hex grid (discrete cell-to-cell movement like Bevy version)
            // Boats move every frame for fast movement
            const MOVE_INTERVAL: u32 = 1; // Move every frame
            if boat.age % MOVE_INTERVAL != 0 {
                return true; // Don't move yet, keep boat
            }
            
            // Rare chance to drift in an adjacent direction (2% chance each way)
            // This doesn't change the boat's stored direction, just the movement this frame
            let mut use_direction = boat.direction;
            let drift_check = rng.random_range(0..100);
            if drift_check < 2 {
                // Drift clockwise (to the right)
                use_direction = (use_direction + 1) % 6;
            } else if drift_check < 4 {
                // Drift counter-clockwise (to the left)
                use_direction = if use_direction == 0 { 5 } else { use_direction - 1 };
            }
            // Note: boat.direction remains unchanged, so boat generally travels straight
            
            // Calculate new position based on hex grid movement
            let current_x = boat.position.0.round() as i32;
            let current_y = boat.position.1.round() as i32;
            let (mut new_x, mut new_y) = Self::hex_neighbor(current_x, current_y, use_direction);
            
            // Apply world wrapping on both axes (matching shader behavior)
            let game_size_i32 = self.game_size as i32;
            new_x = ((new_x % game_size_i32) + game_size_i32) % game_size_i32;
            new_y = ((new_y % game_size_i32) + game_size_i32) % game_size_i32;
            
            // Calculate indices into RGBA terrain data (4 bytes per pixel)
            let new_cell_idx = ((new_y as usize) * (self.game_size as usize) + (new_x as usize)) * 4;
            
            if new_cell_idx >= self.terrain_data.len() {
                // Invalid cell - remove boat
                return false;
            }
            
            // Read altitude from R channel (first byte of RGBA)
            let altitude = self.terrain_data[new_cell_idx];
            // Ocean cutoff: 0.53 * 255 = 135
            let is_land = altitude >= 135;
            
            // Debug: Check current position too - remove boats that are on land
            let current_cell_idx = ((current_y as usize) * (self.game_size as usize) + (current_x as usize)) * 4;
            let current_altitude = if current_cell_idx < self.terrain_data.len() {
                self.terrain_data[current_cell_idx]
            } else {
                0
            };
            let current_is_land = current_altitude >= 135;
            
            // Sanity check: if boat is currently on land, remove it immediately
            if current_is_land {
                if self.frame_count % 120 == 0 {
                    println!("WARNING: Removing boat at ({}, {}) - currently on land (altitude {})", current_x, current_y, current_altitude);
                }
                return false;
            }
            
            if is_land {
                debug_land_encounters += 1;
                
                // Boat encounters land! Record landing for GPU to process
                // GPU will decide attack vs reinforce based on empire relations and personality
                // Format: upper 16 bits = empire_id, lower 16 bits = strength
                let landing_data = ((boat.empire_id as u32) << 16) | (boat.strength as u32);
                
                // Calculate the cell index for boat_landings_cpu buffer (1 u32 per cell)
                let landing_cell_idx = (new_y as usize) * (self.game_size as usize) + (new_x as usize);
                
                // Use atomicMax-style logic: only update if our landing is stronger or same empire
                let existing = self.boat_landings_cpu[landing_cell_idx];
                let existing_empire = (existing >> 16) as u16;
                let existing_strength = (existing & 0xFFFF) as u16;
                
                // If same empire or we're stronger, record this landing
                if existing_empire == boat.empire_id || boat.strength > existing_strength {
                    self.boat_landings_cpu[landing_cell_idx] = landing_data;
                    
                    // Debug: Log successful landings with details
                    if self.frame_count % 60 == 0 {
                        let boat_strength_normalized = boat.strength as f32 / 65535.0;
                        let attack_damage = boat_strength_normalized / 3.0;
                        println!("Boat landing at ({}, {}): Empire {} with strength {} ({:.2}% normalized)", 
                            new_x, new_y, boat.empire_id, boat.strength, boat_strength_normalized * 100.0);
                        println!("   Attack damage = {:.4} ({:.2}%). Conquers cells with strength < {:.4}", 
                            attack_damage, attack_damage * 100.0, attack_damage);
                    }
                }
                
                landed_count += 1;
                
                // Remove boat (GPU handles the actual attack/reinforce logic)
                return false;
            }
            
            // Still on water - update position to new cell
            debug_continued_boats += 1;
            boat.position.0 = new_x as f32;
            boat.position.1 = new_y as f32;
            // Note: boat.direction is NOT updated - it stays constant except for drift on individual moves
            
            // Keep boat alive
            true
        });
        
        // Debug output every 2 seconds
        if self.frame_count % 120 == 0 {
            if self.boats.len() > 0 || landed_count > 0 {
                println!("Boats: {} active, {} landed this cycle (moved: {}, hit land: {})", 
                    self.boats.len(), landed_count, debug_continued_boats, debug_land_encounters);
                // Show positions of first 10 boats for easy finding
                let boats_to_show = self.boats.len().min(10);
                for i in 0..boats_to_show {
                    let boat = &self.boats[i];
                    println!("   Boat {}: pos=({:.1}, {:.1}), empire={}, dir={}", 
                        i+1, boat.position.0, boat.position.1, boat.empire_id, boat.direction);
                }
                if self.boats.len() > 10 {
                    println!("   ... and {} more boats", self.boats.len() - 10);
                }
            }
        }
        
        // Upload boat landings to GPU
        graphics.queue.write_buffer(
            &self.boat_landings_buffer,
            0,
            bytemuck::cast_slice(&self.boat_landings_cpu),
        );
        
        // Update boat instance buffer for rendering
        // Upload actual boat positions and colors
        {
            let mut instance_data = Vec::new();
            
            const MAX_BOATS: usize = 50000;  // Match buffer size
            let boats_to_render = self.boats.len().min(MAX_BOATS);
            
            for boat in self.boats.iter().take(boats_to_render) {
                // Get the empire color using the same hash-based system as the shader
                let (r, g, b) = Self::get_empire_color(boat.empire_id);
                
                instance_data.push([boat.position.0, boat.position.1, r, g, b]);
            }
            
            // Only write if we have boats
            if !instance_data.is_empty() {
                graphics.queue.write_buffer(
                    &self.boat_instance_buffer,
                    0,
                    bytemuck::cast_slice(&instance_data),
                );
            }
            
            // Debug output
            if self.frame_count % 60 == 0 && !self.boats.is_empty() {
                println!("Rendering {} boats on screen (total: {})", instance_data.len(), self.boats.len());
            }
        }
    }
    
    // Helper function to convert HSV to RGB
    fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
        let c = v * s;
        let h_prime = h / 60.0;
        let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
        let m = v - c;
        
        let (r, g, b) = if h_prime < 1.0 {
            (c, x, 0.0)
        } else if h_prime < 2.0 {
            (x, c, 0.0)
        } else if h_prime < 3.0 {
            (0.0, c, x)
        } else if h_prime < 4.0 {
            (0.0, x, c)
        } else if h_prime < 5.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };
        
        (r + m, g + m, b + m)
    }
    
    // Hash function matching the shader's hash_empire_id
    fn hash_empire_id(id: u32) -> u32 {
        let mut x = id;
        x = ((x >> 16) ^ x).wrapping_mul(0x45d9f3b);
        x = ((x >> 16) ^ x).wrapping_mul(0x45d9f3b);
        x = (x >> 16) ^ x;
        x
    }
    
    // Get empire color matching the shader's get_empire_color function
    fn get_empire_color(empire_id: u16) -> (f32, f32, f32) {
        if empire_id == 0 {
            return (0.0, 0.0, 0.0); // Black for unclaimed
        }
        
        let hash_val = Self::hash_empire_id(empire_id as u32);
        
        // Generate HSV values from hash (matching shader logic)
        let hue = (hash_val % 360) as f32;
        let saturation = 0.7 + ((hash_val >> 8) % 30) as f32 / 100.0;
        let value = 0.6 + ((hash_val >> 16) % 20) as f32 / 100.0;
        
        Self::hsv_to_rgb(hue, saturation, value)
    }
    
    // Helper function to get hex neighbor coordinates (even-r offset)
    // Direction mapping matches shader: 0=East, 1=NE, 2=NW, 3=West, 4=SW, 5=SE
    fn hex_neighbor(x: i32, y: i32, direction: u8) -> (i32, i32) {
        let is_even_row = y % 2 == 0;
        match direction {
            0 => {               // East
                (x + 1, y)
            }
            1 => {               // Northeast
                if is_even_row {
                    (x + 1, y - 1)  // Even row: right and up
                } else {
                    (x, y - 1)      // Odd row: up only
                }
            }
            2 => {               // Northwest
                if is_even_row {
                    (x, y - 1)      // Even row: up only
                } else {
                    (x - 1, y - 1)  // Odd row: left and up
                }
            }
            3 => {               // West
                (x - 1, y)
            }
            4 => {               // Southwest
                if is_even_row {
                    (x, y + 1)      // Even row: down only
                } else {
                    (x - 1, y + 1)  // Odd row: left and down
                }
            }
            5 => {               // Southeast
                if is_even_row {
                    (x + 1, y + 1)  // Even row: right and down
                } else {
                    (x, y + 1)      // Odd row: down only
                }
            }
            _ => (x, y)
        }
    }
    
    pub fn update(&mut self, graphics: &GraphicsContext) {
        if self.is_paused {
            return;
        }
        
        self.frame_count += 1;
        if self.frame_count % self.simulation_speed != 0 {
            return;
        }
        
        // Update frame counter for compute shader randomness
        graphics.queue.write_buffer(
            &self.frame_uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.frame_count]),
        );
        
        // Periodic debug output (every 5 seconds at 60fps)
        if self.frame_count % 300 == 0 {
            println!("Simulation running... Frame {}", self.frame_count);
        }
        
        let mut encoder = graphics.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Empire Simulation Compute Encoder"),
        });

        // Clear boat spawns buffer before compute pass
        let buffer_size = (self.game_size * self.game_size) as u64 * std::mem::size_of::<u32>() as u64;
        encoder.clear_buffer(&self.boat_spawns_buffer, 0, Some(buffer_size));

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Empire Simulation Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.compute_pipeline);
            
            // Use the appropriate bind group for ping-pong
            let compute_bind_group = if self.current_is_a {
                &self.compute_bind_group_a // A -> B
            } else {
                &self.compute_bind_group_b // B -> A
            };
            
            compute_pass.set_bind_group(0, compute_bind_group, &[]);
            
            // Dispatch compute shader with dynamic workgroup count based on texture size
            // Uses ceiling division to ensure full coverage for any texture size with 8x8 workgroups
            let workgroup_count = (self.game_size + 8 - 1) / 8;
            compute_pass.dispatch_workgroups(workgroup_count, workgroup_count, 1);
        }

        graphics.queue.submit(std::iter::once(encoder.finish()));
        
        // Process boat spawns from GPU (async readback pipeline)
        // State machine: None → Copy+Request → Poll → Process → None
        const MAX_BOAT_CAPACITY: usize = 50000;
        
        if let Some(rx) = &self.boat_spawns_pending_map {
            // Try to receive without blocking
            if let Ok(result) = rx.try_recv() {
                // Mapping completed! Process the spawn data
                if result.is_ok() {
                    let staging_slice = self.boat_spawns_staging_buffer.slice(..);
                    let spawn_data = staging_slice.get_mapped_range();
                    let spawn_u32s: &[u32] = bytemuck::cast_slice(&spawn_data);
                    
                    let mut spawned_count = 0;
                    
                    // Process each cell's spawn request
                    for y in 0..self.game_size {
                        for x in 0..self.game_size {
                            let cell_idx = (y * self.game_size + x) as usize;
                            let spawn_value = spawn_u32s[cell_idx];
                            
                            // Non-zero value means this cell wants to spawn a boat
                            if spawn_value != 0 {
                                let empire_id = ((spawn_value >> 16) & 0xFFFF) as u16;
                                let direction = (spawn_value & 0xFF) as u8;
                                
                                // The GPU already wrote the spawn request to the OCEAN cell's index
                                // So (x, y) IS the ocean position where we should spawn the boat
                                // The direction tells us which way the boat should move (away from the coast)
                                
                                // Check if we're at max boat capacity (50k limit)
                                const MAX_BOAT_CAPACITY: usize = 50000;
                                if self.boats.len() >= MAX_BOAT_CAPACITY {
                                    // Skip spawning - we're at capacity
                                    continue;
                                }
                                
                                // Verify this is actually ocean before spawning
                                // terrain_data is RGBA format (4 bytes per pixel), R channel = altitude
                                let terrain_idx = ((y * self.game_size + x) as usize) * 4;
                                if terrain_idx < self.terrain_data.len() {
                                    let altitude = self.terrain_data[terrain_idx]; // Read R channel
                                    let is_water = altitude < 135; // Ocean cutoff: 0.53 * 255 = 135
                                    
                                    if is_water {
                                        // Spawn boat at this ocean cell with very high strength
                                        // Boats need to be strong to overcome the 3:1 defender advantage
                                        // Reference implementation gave boats the cell's full strength or more
                                        // Using ~60% of max possible strength (40000/65535) for strong boats
                                        let boat = Boat {
                                            position: (x as f32, y as f32),
                                            empire_id,
                                            strength: 40000, // Strong boat (~61% of max) to overcome defender advantage
                                            direction,
                                            age: 0,
                                        };
                                        self.boats.push(boat);
                                        spawned_count += 1;
                                    }
                                }
                            }
                        }
                    }
                    
                    if spawned_count > 0 {
                        println!("GPU spawned {} boats this frame (total: {}/{})", 
                                spawned_count, self.boats.len(), MAX_BOAT_CAPACITY);
                    }
                    
                    drop(spawn_data);
                    self.boat_spawns_staging_buffer.unmap();
                } else {
                    eprintln!("WARNING: Boat spawns buffer map failed: {:?}", result.err());
                }
                
                // Clear pending state - next frame will start a new request
                self.boat_spawns_pending_map = None;
            }
        } else {
            // No pending map - start a new copy+request cycle
            let buffer_size = (self.game_size * self.game_size) as u64 * std::mem::size_of::<u32>() as u64;
            
            let mut encoder = graphics.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Boat Spawns Copy Encoder"),
            });
            encoder.copy_buffer_to_buffer(&self.boat_spawns_buffer, 0, &self.boat_spawns_staging_buffer, 0, buffer_size);
            graphics.queue.submit(std::iter::once(encoder.finish()));
            
            // Request async mapping
            let staging_slice = self.boat_spawns_staging_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            staging_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            self.boat_spawns_pending_map = Some(rx);
        }
        
        // Update boats (movement, landing detection)
        self.update_boats(graphics);
        
        // Async diplomacy processing pipeline - runs every 30 frames to reduce CPU overhead
        // OPTIMIZATION: Diplomacy runs on a separate thread to not block main simulation
        let should_process_diplomacy = self.frame_count % 30 == 0;
        
        // Check if diplomacy thread has completed work
        if self.diplomacy_processing {
            // Try to receive result without blocking
            if let Ok(response) = self.diplomacy_result_receiver.try_recv() {
                // Diplomacy processing complete! Upload results to GPU
                graphics.queue.write_buffer(
                    &self.diplomacy_relations_buffer,
                    0,
                    bytemuck::cast_slice(&response.relations_buffer),
                );
                self.diplomacy_processing = false;
            }
            // If not ready yet, just continue - don't block!
        }
        
        if let Some(rx) = &self.diplomacy_pending_map {
            // Try to receive without blocking
            if let Ok(result) = rx.try_recv() {
                // Mapping completed! Send data to worker thread
                if result.is_ok() {
                    let staging_slice = self.diplomacy_staging_buffer.slice(..);
                    let counter_data = staging_slice.get_mapped_range();
                    let counter_u32s: &[u32] = bytemuck::cast_slice(&counter_data);
                    
                    // Count non-zero events for debugging
                    let total_events: u32 = counter_u32s.iter().sum();
                    if total_events > 0 && self.frame_count % 300 == 0 {
                        println!("Diplomacy: Processing {} events at frame {} (async)", total_events, self.frame_count);
                    }
                    
                    // Send work to background thread (non-blocking)
                    let work_request = DiplomacyWorkRequest {
                        counters: counter_u32s.to_vec(),
                        personality_diffs: self.personality_diff_cache.clone(),
                        frame_count: self.frame_count,
                    };
                    
                    if self.diplomacy_work_sender.send(work_request).is_ok() {
                        self.diplomacy_processing = true;
                    }
                    
                    drop(counter_data);
                    self.diplomacy_staging_buffer.unmap();
                }
                
                // Clear pending state - ready for next cycle
                self.diplomacy_pending_map = None;
            }
        } else if should_process_diplomacy && !self.diplomacy_processing {
            // No pending map and it's time to process - start a new cycle
            // Copy counters to staging buffer
            let mut encoder = graphics.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Diplomacy Copy Encoder"),
            });
            encoder.copy_buffer_to_buffer(
                &self.diplomacy_counters_buffer,
                0,
                &self.diplomacy_staging_buffer,
                0,
                (2048 * 2048 * 5 * std::mem::size_of::<u32>()) as u64,
            );
            graphics.queue.submit(std::iter::once(encoder.finish()));
            
            // Zero out the counters immediately for next batch (reuse pre-allocated buffer)
            graphics.queue.write_buffer(
                &self.diplomacy_counters_buffer,
                0,
                bytemuck::cast_slice(&self.zero_counters_buffer),
            );
            
            // Request async mapping
            let staging_slice = self.diplomacy_staging_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            staging_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            
            // Store receiver for next frame
            self.diplomacy_pending_map = Some(rx);
        }
        
        // Empire ownership readback for boats - runs every 60 frames
        let should_update_ownership = self.frame_count % 60 == 0;
        
        if let Some(rx) = &self.empire_pending_readback {
            // Try to receive without blocking
            if let Ok(result) = rx.try_recv() {
                if result.is_ok() {
                    let empire_slice = self.empire_readback_buffer.slice(..);
                    let empire_data = empire_slice.get_mapped_range();
                    let empire_u16s: &[u16] = bytemuck::cast_slice(&empire_data);
                    
                    // Extract empire IDs from RGBA16Unorm texture (R channel = empire_id)
                    for i in 0..(self.game_size * self.game_size) as usize {
                        let pixel_offset = i * 4;  // 4 channels per pixel
                        if pixel_offset < empire_u16s.len() {
                            self.empire_ownership[i] = empire_u16s[pixel_offset];  // R channel
                        }
                    }
                    
                    drop(empire_data);
                    self.empire_readback_buffer.unmap();
                }
                
                self.empire_pending_readback = None;
            }
        } else if should_update_ownership {
            // Start a new readback cycle
            let current_texture = if self.current_is_a {
                &self.texture_b  // Just wrote to B
            } else {
                &self.texture_a  // Just wrote to A
            };
            
            let mut encoder = graphics.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Empire Readback Encoder"),
            });
            
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: current_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.empire_readback_buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(self.game_size * 8),  // 4 channels * 2 bytes
                        rows_per_image: Some(self.game_size),
                    },
                },
                wgpu::Extent3d {
                    width: self.game_size,
                    height: self.game_size,
                    depth_or_array_layers: 1,
                },
            );
            
            graphics.queue.submit(std::iter::once(encoder.finish()));
            
            // Request async mapping
            let empire_slice = self.empire_readback_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            empire_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            
            self.empire_pending_readback = Some(rx);
        }
        
        // Flip ping-pong state
        self.current_is_a = !self.current_is_a;
    }
    
    pub fn claim_cell_for_empire(&mut self, x: u32, y: u32, empire_id: u8, graphics: &GraphicsContext) {
        if x >= self.game_size || y >= self.game_size {
            return; // Out of bounds
        }
        
        println!("Claiming cell at ({}, {}) for Empire {} (Frame: {})", x, y, empire_id, self.frame_count);
        
        // Create empire cell data (Channel layout: R=Empire ID, G=Strength, B=Need, A=Action)
        // Now using 16-bit values (u16) - 8 bytes total per pixel
        let empire_id_u16 = empire_id as u16;
        let strength_u16: u16 = 10000; // Higher initial strength (16-bit range)
        let need_u16: u16 = 3000;      // Default need (16-bit range)
        let action_u16: u16 = 0;       // No initial action
        
        // Convert to bytes in little-endian format
        let mut cell_data = Vec::new();
        cell_data.extend_from_slice(&empire_id_u16.to_le_bytes());
        cell_data.extend_from_slice(&strength_u16.to_le_bytes());
        cell_data.extend_from_slice(&need_u16.to_le_bytes());
        cell_data.extend_from_slice(&action_u16.to_le_bytes());
        
        // Write to both textures to ensure the cell persists through ping-pong
        for texture in [&self.texture_a, &self.texture_b] {
            graphics.queue.write_texture(
                // Destination
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                // Data
                &cell_data,
                // Data layout
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(8), // 8 bytes per pixel (Rgba16Unorm)
                    rows_per_image: Some(1), // Single pixel
                },
                // Size
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }
        
        println!("  -> Successfully claimed cell for Empire {}", empire_id);
    }
    
    pub fn toggle_pause(&mut self) {
        self.is_paused = !self.is_paused;
        println!("Simulation {}", if self.is_paused { "paused" } else { "resumed" });
    }
    
    pub fn set_render_mode(&mut self, mode: RenderMode, graphics: &GraphicsContext) {
        if self.current_render_mode != mode {
            self.current_render_mode = mode;
            let mode_name = match mode {
                RenderMode::Empires => "Empires (unique colors)",
                RenderMode::Strength => "Strength Heatmap (blue=weak, red=strong)",
                RenderMode::Need => "Need Heatmap (green=low, yellow=med, red=high)",
                RenderMode::Action => "Action Visualization (blue=reinforce, red=attack)",
                RenderMode::Age => "Age Visualization (red=new, green=old)",
                RenderMode::BoatNeed => "Boat Need Heatmap (blue=low, cyan=med, yellow=high, red=coastal conquest)",
                RenderMode::Diplomacy => {
                    // Auto-select empire 1 when entering diplomacy mode if none selected
                    if self.perspective_empire == 0 {
                        self.set_perspective_empire(1, graphics);
                    }
                    "Diplomacy Perspective (blue=self, gold=allied, green=neutral, red=enemy)"
                },
            };
            println!("Switched to render mode: {}", mode_name);
        }
    }
    
    pub fn get_render_mode(&self) -> RenderMode {
        self.current_render_mode
    }

    pub fn set_perspective_empire(&mut self, empire_id: u16, graphics: &GraphicsContext) {
        self.perspective_empire = empire_id;
        // Update the GPU buffer with the new perspective empire
        graphics.queue.write_buffer(
            &self.perspective_empire_buffer,
            0,
            &(empire_id as u32).to_le_bytes(),
        );
        
        if empire_id == 0 {
            println!("Cleared diplomacy perspective");
        } else {
            println!("Set diplomacy perspective to Empire {}", empire_id);
        }
    }

    pub fn select_perspective_at_cell(&mut self, x: u32, y: u32, graphics: &GraphicsContext) {
        if x >= self.game_size || y >= self.game_size {
            return; // Out of bounds
        }
        
        println!("Reading empire at cell ({}, {})", x, y);
        
        // Instead of blocking, just read the current state synchronously with immediate device poll
        // Create a small staging buffer for one pixel
        let bytes_per_pixel = 8u64;
        let staging_buffer = graphics.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Perspective Selection Staging"),
            size: bytes_per_pixel,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = graphics.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Read Empire ID"),
        });
        
        let source_texture = if self.current_is_a { &self.texture_a } else { &self.texture_b };
        
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: source_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_pixel as u32),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        
        graphics.queue.submit(Some(encoder.finish()));
        
        // Request the buffer mapping
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        
        // Poll device until callback completes (wgpu 26 requires PollType argument)
        while receiver.try_recv().is_err() {
            let _ = graphics.device.poll(wgpu::PollType::Wait);
        }
        
        // At this point the buffer is mapped, read the data
        let data = buffer_slice.get_mapped_range();
        if data.len() >= 2 {
            let empire_id = u16::from_le_bytes([data[0], data[1]]);
            println!("Read empire ID: {}", empire_id);
            drop(data);
            staging_buffer.unmap();
            
            if empire_id > 0 {
                self.set_perspective_empire(empire_id, graphics);
            } else {
                println!("Clicked on unclaimed territory (no empire)");
            }
        } else {
            println!("WARNING: Buffer data too small");
        }
    }

    
    pub fn render(&self, render_pass: &mut wgpu::RenderPass, vertex_buffer: &wgpu::Buffer, camera_bind_group: &wgpu::BindGroup) {
        // First, render the terrain (opaque background)
        render_pass.set_pipeline(&self.terrain_pipeline);
        render_pass.set_bind_group(0, &self.terrain_bind_group, &[]);
        render_pass.set_bind_group(1, camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.draw(0..6, 0..1);
        
        // Then, render the empire simulation (translucent on top) with current render mode
        let current_pipeline = match self.current_render_mode {
            RenderMode::Empires => &self.render_pipeline_empires,
            RenderMode::Strength => &self.render_pipeline_strength,
            RenderMode::Need => &self.render_pipeline_need,
            RenderMode::Action => &self.render_pipeline_action,
            RenderMode::Age => &self.render_pipeline_age,
            RenderMode::Diplomacy => &self.render_pipeline_diplomacy,
            RenderMode::BoatNeed => &self.render_pipeline_boat_need,
        };
        render_pass.set_pipeline(current_pipeline);
        
        // Use the appropriate texture for display based on render mode
        let display_bind_group = match self.current_render_mode {
            RenderMode::Age | RenderMode::BoatNeed => {
                // Use auxiliary textures for age/boat_need visualization
                // After the flip, current_is_a indicates which texture was just WRITTEN to
                // If current_is_a is true, we just wrote to A (because we were reading from B)
                // If current_is_a is false, we just wrote to B (because we were reading from A)
                if self.current_is_a {
                    &self.aux_bind_group_a // Show A (just written)
                } else {
                    &self.aux_bind_group_b // Show B (just written)
                }
            }
            RenderMode::Diplomacy => {
                // Use diplomacy perspective bind groups
                if self.current_is_a {
                    &self.diplomacy_bind_group_a
                } else {
                    &self.diplomacy_bind_group_b
                }
            }
            _ => {
                // Use main game textures for all other modes
                if self.current_is_a {
                    &self.display_bind_group_a
                } else {
                    &self.display_bind_group_b
                }
            }
        };
        
        render_pass.set_bind_group(0, display_bind_group, &[]);
        render_pass.set_bind_group(1, camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.draw(0..6, 0..1);
        
        // Render boats on top of everything
        {
            let num_boats = self.boats.len() as u32;
            
            if num_boats > 0 {
                render_pass.set_pipeline(&self.boat_pipeline);
                render_pass.set_bind_group(0, camera_bind_group, &[]);    // Camera at group 0 for boats
                render_pass.set_vertex_buffer(0, self.boat_instance_buffer.slice(..));
                render_pass.draw(0..6, 0..num_boats);  // 6 vertices per quad, one instance per boat
            }
        }
    }
    
    /// Calculate personality difference between two empires (0.0 = identical, 1.0 = maximally different)
    /// Returns None if either empire doesn't exist
    pub fn get_personality_difference(&self, empire_a: u16, empire_b: u16) -> Option<f32> {
        let personality_a = self.empire_personalities.get(&empire_a)?;
        let personality_b = self.empire_personalities.get(&empire_b)?;
        
        // Calculate absolute differences for each trait (0-255 range)
        let aggression_diff = (personality_a.aggression as i16 - personality_b.aggression as i16).abs() as f32;
        let expansion_diff = (personality_a.expansion as i16 - personality_b.expansion as i16).abs() as f32;
        let cooperation_diff = (personality_a.cooperation as i16 - personality_b.cooperation as i16).abs() as f32;
        
        // Average the differences and normalize to 0.0-1.0 range
        // Maximum difference per trait is 255, so max total is 255
        let avg_diff = (aggression_diff + expansion_diff + cooperation_diff) / 3.0;
        let normalized = avg_diff / 255.0;
        
        Some(normalized)
    }
}
