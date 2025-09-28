use wgpu::util::DeviceExt;
use crate::graphics::{GraphicsContext, load_shader};

pub struct EmpireSimulation {
    pub texture_a: wgpu::Texture,
    pub texture_b: wgpu::Texture,
    pub compute_bind_group_a: wgpu::BindGroup,
    pub compute_bind_group_b: wgpu::BindGroup,
    pub display_bind_group_a: wgpu::BindGroup,
    pub display_bind_group_b: wgpu::BindGroup,
    pub compute_pipeline: wgpu::ComputePipeline,
    pub render_pipeline: wgpu::RenderPipeline,
    pub current_is_a: bool,
    pub frame_count: u32,
    pub simulation_speed: u32,
    pub is_paused: bool,
    pub game_size: u32,
    pub frame_uniform_buffer: wgpu::Buffer,
}

impl EmpireSimulation {
    pub fn new(graphics: &GraphicsContext, camera_bind_group_layout: &wgpu::BindGroupLayout) -> Self {
        println!("Creating Empire simulation textures...");
        let game_size = 256u32;
        
        // Initialize with empty empire map (all cells unclaimed)
        let mut initial_data = Vec::new();
        for _y in 0..game_size {
            for _x in 0..game_size {
                // Channel layout: R=Empire ID, G=Strength, B=Need, A=Action
                // All cells start unclaimed (Empire ID = 0)
                initial_data.extend_from_slice(&[
                    0u8,   // R: Empire ID (0 = unclaimed)
                    0u8,   // G: Strength (0 for unclaimed cells)
                    0u8,   // B: Need (0 for unclaimed cells)
                    0u8,   // A: Action (0 = no action)
                ]);
            }
        }
        
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
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING 
                | wgpu::TextureUsages::STORAGE_BINDING 
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };
        
        let texture_a = graphics.device.create_texture_with_data(
            &graphics.queue,
            &texture_desc,
            wgpu::util::TextureDataOrder::LayerMajor,
            &initial_data,
        );
        
        let texture_b = graphics.device.create_texture(&texture_desc);
        
        // Create texture views
        let game_view_a = texture_a.create_view(&wgpu::TextureViewDescriptor::default());
        let game_view_b = texture_b.create_view(&wgpu::TextureViewDescriptor::default());
        
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
                        format: wgpu::TextureFormat::Rgba8Unorm,
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
        
        // Load render shader
        let render_shader = load_shader(&graphics.device, "shaders/empire_render.wgsl");
        
        // Create render pipeline
        let render_pipeline = graphics.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Empire Simulation Render Pipeline"),
            layout: Some(&graphics.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Empire Simulation Pipeline Layout"),
                bind_group_layouts: &[&display_bind_group_layout, camera_bind_group_layout],
                push_constant_ranges: &[],
            })),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[crate::graphics::Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
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
        
        println!("Empire simulation pipeline created! Ready to simulate.");
        
        Self {
            texture_a,
            texture_b,
            compute_bind_group_a,
            compute_bind_group_b,
            display_bind_group_a,
            display_bind_group_b,
            compute_pipeline,
            render_pipeline,
            current_is_a: true,
            frame_count: 0,
            simulation_speed: 1, // Update every frame for immediate feedback
            is_paused: false,
            game_size,
            frame_uniform_buffer,
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
            
            // Dispatch compute shader (256x256 texture with 8x8 workgroups)
            compute_pass.dispatch_workgroups(32, 32, 1);
        }

        graphics.queue.submit(std::iter::once(encoder.finish()));
        
        // Flip ping-pong state
        self.current_is_a = !self.current_is_a;
    }
    
    pub fn claim_cell_for_empire(&mut self, x: u32, y: u32, empire_id: u8, graphics: &GraphicsContext) {
        if x >= self.game_size || y >= self.game_size {
            return; // Out of bounds
        }
        
        println!("Claiming cell at ({}, {}) for Empire {} (Frame: {})", x, y, empire_id, self.frame_count);
        
        // Create empire cell data (Channel layout: R=Empire ID, G=Strength, B=Need, A=Action)
        let cell_data: [u8; 4] = [
            empire_id,  // R: Empire ID
            128,        // G: Strength (default strength for player-claimed cells)
            64,         // B: Need (default need)
            0,          // A: Action (no initial action, will be set by compute shader)
        ];
        
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
                    bytes_per_row: Some(4), // 4 bytes per pixel (Rgba8Unorm)
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
    

    
    pub fn render(&self, render_pass: &mut wgpu::RenderPass, vertex_buffer: &wgpu::Buffer, camera_bind_group: &wgpu::BindGroup) {
        render_pass.set_pipeline(&self.render_pipeline);
        
        // Use the current texture for display
        let display_bind_group = if self.current_is_a {
            &self.display_bind_group_a
        } else {
            &self.display_bind_group_b
        };
        
        render_pass.set_bind_group(0, display_bind_group, &[]);
        render_pass.set_bind_group(1, camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.draw(0..6, 0..1);
    }
}
