use wgpu::util::DeviceExt;
use crate::graphics::{GraphicsContext, load_shader};
use noise::{NoiseFn, Simplex};
use std::collections::HashMap;

mod diplomacy;
use diplomacy::DiplomacyState;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RenderMode {
    Empires = 0,
    Strength = 1,
    Need = 2,
    Action = 3,
    Age = 4,
    Diplomacy = 5,
}

// Generate terrain data using simplex noise with cylindrical world wrapping
fn generate_terrain_data(size: u32) -> Vec<u8> {
    println!("Generating terrain data with cylindrical world wrapping...");
    
    // Multiple noise layers for varied terrain
    let noise = Simplex::new(12345);
    let noise2 = Simplex::new(67890);
    let noise3 = Simplex::new(54321);
    
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
    pub terrain_pipeline: wgpu::RenderPipeline,  // For terrain rendering
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
            (i, 54321u32).hash(&mut hasher); // Seed for shuffle consistency
            let j = (hasher.finish() as usize) % (i + 1);
            land_positions.swap(i, j);
        }
        
        // Spawn empires on shuffled land positions
        let empire_spawn_chance = 0.005; // 0.5% chance per land cell
        for (x, y) in &land_positions {
            let mut hasher = DefaultHasher::new();
            (*x, *y, 12345u32).hash(&mut hasher);
            let random_val = (hasher.finish() % 1000) as f32 / 1000.0;
            
            if random_val < empire_spawn_chance && empire_id_counter < 65535 {
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
            panic!("❌ CRITICAL: {} empires spawned, but diplomacy system only supports {}. Reduce empire count or increase MAX_EMPIRES.", 
                   num_spawned_empires, max_empires);
        }
        
        println!("Creating empire parameters texture ({}x{}) for {} empires...", max_empires, max_empires, num_spawned_empires);
        
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
                    (empire_id, target_empire_id, 54321u32).hash(&mut hasher); // Seed for consistency
                    let diplomacy_random = (hasher.finish() % 256) as u8;
                    
                    // Generate random aggression for this empire (same for all targets)
                    let mut hasher2 = DefaultHasher::new();
                    (empire_id, 98765u32).hash(&mut hasher2);
                    let aggression_random = (hasher2.finish() % 256) as u8;
                    
                    // Generate random expansion trait (0=defensive, 255=expansionist)
                    let mut hasher3 = DefaultHasher::new();
                    (empire_id, 13579u32).hash(&mut hasher3);
                    let expansion_random = (hasher3.finish() % 256) as u8;
                    
                    // Generate random cooperation trait (0=selfish, 255=cooperative)
                    let mut hasher4 = DefaultHasher::new();
                    (empire_id, 24680u32).hash(&mut hasher4);
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
            terrain_pipeline,
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
            diplomacy_state: DiplomacyState::new(),
            diplomacy_pending_map: None,
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
        
        // Async diplomacy processing pipeline - runs as fast as possible without blocking
        // State machine: None → Copy+Request → Poll → Process → None
        
        if let Some(rx) = &self.diplomacy_pending_map {
            // Try to receive without blocking
            if let Ok(result) = rx.try_recv() {
                // Mapping completed! Process the data
                if result.is_ok() {
                    let staging_slice = self.diplomacy_staging_buffer.slice(..);
                    let counter_data = staging_slice.get_mapped_range();
                    let counter_u32s: &[u32] = bytemuck::cast_slice(&counter_data);
                    
                    // Count non-zero events for debugging
                    let total_events: u32 = counter_u32s.iter().sum();
                    if total_events > 0 && self.frame_count % 300 == 0 {
                        println!("📊 Diplomacy: Processed {} events at frame {}", total_events, self.frame_count);
                    }
                    
                    // Process counters with personality-based modifiers (using pre-computed cache)
                    let personality_diffs = &self.personality_diff_cache;
                    let territory_sizes = std::collections::HashMap::new();
                    self.diplomacy_state.process_counters(
                        counter_u32s, 
                        &territory_sizes,
                        |a, b| {
                            let idx = (a as usize) * 2048 + (b as usize);
                            personality_diffs.get(idx).copied().unwrap_or(0.5)
                        },
                    );
                    
                    // Propagate attack reactions to third parties
                    // "You attacked my friend/enemy" effects
                    self.diplomacy_state.propagate_attack_reactions(counter_u32s, |a, b| {
                        let idx = (a as usize) * 2048 + (b as usize);
                        personality_diffs.get(idx).copied().unwrap_or(0.5)
                    });
                    
                    drop(counter_data);
                    self.diplomacy_staging_buffer.unmap();
                    
                    // Upload updated relations to GPU
                    let relations_data = self.diplomacy_state.to_buffer();
                    graphics.queue.write_buffer(
                        &self.diplomacy_relations_buffer,
                        0,
                        bytemuck::cast_slice(&relations_data),
                    );
                }
                
                // Clear pending state - ready for next cycle
                self.diplomacy_pending_map = None;
            }
        } else {
            // No pending map - start a new cycle
            // Copy counters to staging buffer
            let mut encoder = graphics.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Diplomacy Copy Encoder"),
            });
            encoder.copy_buffer_to_buffer(
                &self.diplomacy_counters_buffer,
                0,
                &self.diplomacy_staging_buffer,
                0,
                (256 * 256 * 4 * std::mem::size_of::<u32>()) as u64,
            );
            graphics.queue.submit(std::iter::once(encoder.finish()));
            
            // Zero out the counters immediately for next batch
            let zero_counters = vec![0u32; 256 * 256 * 4];
            graphics.queue.write_buffer(
                &self.diplomacy_counters_buffer,
                0,
                bytemuck::cast_slice(&zero_counters),
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
                RenderMode::Diplomacy => {
                    // Auto-select empire 1 when entering diplomacy mode if none selected
                    if self.perspective_empire == 0 {
                        self.set_perspective_empire(1, graphics);
                    }
                    "Diplomacy Perspective (blue=self, gold=allied, green=neutral, red=enemy)"
                },
            };
            println!("🎨 Switched to render mode: {}", mode_name);
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
            println!("🔭 Cleared diplomacy perspective");
        } else {
            println!("🔭 Set diplomacy perspective to Empire {}", empire_id);
        }
    }

    pub fn select_perspective_at_cell(&mut self, x: u32, y: u32, graphics: &GraphicsContext) {
        if x >= self.game_size || y >= self.game_size {
            return; // Out of bounds
        }
        
        println!("🔍 Reading empire at cell ({}, {})", x, y);
        
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
            println!("✅ Read empire ID: {}", empire_id);
            drop(data);
            staging_buffer.unmap();
            
            if empire_id > 0 {
                self.set_perspective_empire(empire_id, graphics);
            } else {
                println!("🔭 Clicked on unclaimed territory (no empire)");
            }
        } else {
            println!("⚠️  Buffer data too small");
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
        };
        render_pass.set_pipeline(current_pipeline);
        
        // Use the appropriate texture for display based on render mode
        let display_bind_group = match self.current_render_mode {
            RenderMode::Age => {
                // Use auxiliary textures for age visualization
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
