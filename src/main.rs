// Phase 5: Conway's Game of Life - GPU-based cellular automata
// Goal: Implement Conway's Game of Life using compute shaders and ping-pong textures

use std::sync::Arc;
use std::collections::HashSet;
use winit::{
    application::ApplicationHandler,
    event::{WindowEvent, ElementState, MouseButton},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};
use wgpu::util::DeviceExt;
use glyph_brush::ab_glyph::FontRef;
use glyph_brush::{Section, Text};

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    
    // Game of Life textures (ping-pong)
    game_texture_a: wgpu::Texture,
    game_texture_b: wgpu::Texture,
    game_bind_group_a: wgpu::BindGroup,
    game_bind_group_b: wgpu::BindGroup,
    display_bind_group_a: wgpu::BindGroup,
    display_bind_group_b: wgpu::BindGroup,
    
    // Compute pipeline for Game of Life simulation
    compute_pipeline: wgpu::ComputePipeline,
    
    // Ping-pong state (true = A is current, false = B is current)
    current_is_a: bool,
    
    // Simulation control
    frame_count: u32,
    simulation_speed: u32, // Skip frames between updates
    is_paused: bool,
    
    // Game world properties
    game_size: u32,
    
    // Camera system
    camera_x: f32,        // Camera position in world coordinates
    camera_y: f32,
    zoom_level: f32,      // Zoom factor (1.0 = normal, 2.0 = 2x zoom, 0.5 = zoomed out)
    
    // Input state
    keys_pressed: std::collections::HashSet<winit::keyboard::KeyCode>,
    
    // Camera system buffers
    camera_uniform_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    
    // FPS tracking and text overlay
    fps_visible: bool,
    frame_times: Vec<std::time::Instant>,
    last_fps_update: std::time::Instant,
    current_fps: f32,
    fps_1_percent_low: f32,
    text_brush: Option<wgpu_text::TextBrush<glyph_brush::ab_glyph::FontRef<'static>>>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        println!("Initializing GPU context...");
        
        let size = window.inner_size();
        
        // Create wgpu instance
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        // Create surface
        let surface = instance.create_surface(window.clone()).unwrap();
        
        // Find adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
            
        println!("GPU Adapter: {}", adapter.get_info().name);
        
        // Create device and queue
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
            
        println!("GPU Device initialized successfully!");
        
        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
            
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        
        // Create Game of Life textures (ping-pong buffers)
        println!("Creating Game of Life textures...");
        let game_size = 256u32;
        
        // Initialize with random pattern
        let mut initial_data = Vec::new();
        for y in 0..game_size {
            for x in 0..game_size {
                // Create some interesting initial patterns
                let alive = if x >= 120 && x <= 130 && y >= 120 && y <= 130 {
                    // Proper glider pattern (moving diagonally down-right)
                    match (x - 120, y - 120) {
                        (1, 0) | (2, 1) | (0, 2) | (1, 2) | (2, 2) => true,
                        _ => false,
                    }
                } else if x >= 100 && x <= 110 && y >= 100 && y <= 110 {
                    // Add a simple blinker (oscillator)
                    match (x - 100, y - 100) {
                        (5, 4) | (5, 5) | (5, 6) => true,
                        _ => false,
                    }
                } else {
                    false
                };
                
                if alive {
                    initial_data.extend_from_slice(&[255u8, 255u8, 255u8, 255u8]); // White = alive
                } else {
                    initial_data.extend_from_slice(&[0u8, 0u8, 0u8, 255u8]); // Black = dead
                }
            }
        }
        
        // Create two identical textures for ping-pong
        let texture_desc = wgpu::TextureDescriptor {
            label: Some("Game of Life Texture"),
            size: wgpu::Extent3d {
                width: game_size,
                height: game_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm, // Use Unorm instead of UnormSrgb for storage binding
            usage: wgpu::TextureUsages::TEXTURE_BINDING 
                | wgpu::TextureUsages::STORAGE_BINDING 
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };
        
        let game_texture_a = device.create_texture_with_data(
            &queue,
            &texture_desc,
            wgpu::util::TextureDataOrder::LayerMajor,
            &initial_data,
        );
        
        let game_texture_b = device.create_texture(&texture_desc);
        
        // Create texture views
        let game_view_a = game_texture_a.create_view(&wgpu::TextureViewDescriptor::default());
        let game_view_b = game_texture_b.create_view(&wgpu::TextureViewDescriptor::default());
        
        let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        // Create compute shader for Game of Life
        println!("Creating Game of Life compute pipeline...");
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Game of Life Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

fn get_cell(pos: vec2<i32>) -> u32 {
    let dims = textureDimensions(input_texture);
    let wrapped_pos = vec2<i32>(
        (pos.x + i32(dims.x)) % i32(dims.x),
        (pos.y + i32(dims.y)) % i32(dims.y)
    );
    let cell = textureLoad(input_texture, wrapped_pos, 0);
    return u32(cell.r > 0.5);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pos = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let dims = textureDimensions(input_texture);
    
    if (pos.x >= i32(dims.x) || pos.y >= i32(dims.y)) {
        return;
    }
    
    // Count living neighbors
    var neighbors = 0u;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            neighbors += get_cell(pos + vec2<i32>(dx, dy));
        }
    }
    
    let current_cell = get_cell(pos);
    var new_cell = 0u;
    
    // Conway's Game of Life rules
    if (current_cell == 1u) {
        // Live cell with 2 or 3 neighbors survives
        if (neighbors == 2u || neighbors == 3u) {
            new_cell = 1u;
        }
    } else {
        // Dead cell with exactly 3 neighbors becomes alive
        if (neighbors == 3u) {
            new_cell = 1u;
        }
    }
    
    let color = vec4<f32>(f32(new_cell), f32(new_cell), f32(new_cell), 1.0);
    textureStore(output_texture, pos, color);
}
                "#.into(),
            ),
        });
        
        // Create compute bind group layout
        let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Game of Life Compute Bind Group Layout"),
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
            ],
        });
        
        // Create compute pipeline
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Game of Life Compute Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Game of Life Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        // Create compute bind groups for ping-pong (A -> B and B -> A)
        let game_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            ],
        });
        
        let game_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            ],
        });
        
        // Create display bind group layout (for rendering to screen)
        let display_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let display_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        
        let display_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        
        // Create vertex buffer for the simulation quad in world coordinates
        let sim_size = game_size as f32;
        let vertices = [
            Vertex { position: [0.0, sim_size], tex_coords: [0.0, 1.0] }, // Bottom-left
            Vertex { position: [sim_size, sim_size], tex_coords: [1.0, 1.0] }, // Bottom-right
            Vertex { position: [sim_size, 0.0], tex_coords: [1.0, 0.0] }, // Top-right
            Vertex { position: [0.0, sim_size], tex_coords: [0.0, 1.0] }, // Bottom-left
            Vertex { position: [sim_size, 0.0], tex_coords: [1.0, 0.0] }, // Top-right
            Vertex { position: [0.0, 0.0], tex_coords: [0.0, 0.0] }, // Top-left
        ];
        
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        
        // Create camera uniform buffer
        let camera_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniform Buffer"),
            size: 64, // mat4x4<f32> = 16 floats * 4 bytes = 64 bytes
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create camera bind group layout
        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Create camera bind group
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_uniform_buffer.as_entire_binding(),
            }],
        });

        // Create shader for textured quad with camera support
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Texture Shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
// Camera uniform buffer
struct CameraUniform {
    view_proj: mat4x4<f32>,
}

@group(1) @binding(0)
var<uniform> camera: CameraUniform;

// Vertex shader
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Transform vertex position by camera matrix
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 0.0, 1.0);
    out.tex_coords = in.tex_coords;
    return out;
}

// Fragment shader
@group(0) @binding(0)
var t_texture: texture_2d<f32>;
@group(0) @binding(1)
var s_texture: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_texture, s_texture, in.tex_coords);
}
                "#.into(),
            ),
        });
        
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Game of Life Render Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Game of Life Pipeline Layout"),
                bind_group_layouts: &[&display_bind_group_layout, &camera_bind_group_layout],
                push_constant_ranges: &[],
            })),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
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
        
        println!("Game of Life pipeline created! Ready to simulate.");
        
        // Create text rendering system using wgpu_text
        println!("Creating text rendering system...");
        let font_data = include_bytes!("../assets/fonts/DejaVuSans.ttf");
        let text_brush = wgpu_text::BrushBuilder::using_font_bytes(font_data)
            .unwrap()
            .build(&device, config.width, config.height, config.format);
        println!("Text rendering system created successfully!");

        let now = std::time::Instant::now();
        
        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            game_texture_a,
            game_texture_b,
            game_bind_group_a,
            game_bind_group_b,
            display_bind_group_a,
            display_bind_group_b,
            compute_pipeline,
            current_is_a: true,
            frame_count: 0,
            simulation_speed: 5, // Update every 5 frames
            is_paused: false,
            game_size,
            // Camera system - start showing the full simulation
            camera_x: 0.0,
            camera_y: 0.0,
            zoom_level: 0.5, // Start zoomed out to see the full simulation
            keys_pressed: HashSet::new(),
            camera_uniform_buffer,
            camera_bind_group,
            // FPS tracking
            fps_visible: true,
            frame_times: Vec::with_capacity(120), // Store up to 120 frame times (2 seconds at 60fps)
            last_fps_update: now,
            current_fps: 0.0,
            fps_1_percent_low: 0.0,
            text_brush: Some(text_brush),
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            
            // Update text brush for new size
            if let Some(text_brush) = &mut self.text_brush {
                text_brush.resize_view(new_size.width as f32, new_size.height as f32, &self.queue);
            }
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // Update FPS tracking
        self.update_fps_tracking();
        
        // Handle camera input (using fixed timestep for now)
        let dt = 1.0 / 60.0; // Assume 60 FPS
        self.handle_camera_input(dt);
        
        // Update camera matrix
        self.update_camera();
        
        // Update simulation every few frames (only if not paused)
        if !self.is_paused {
            self.frame_count += 1;
            if self.frame_count % self.simulation_speed == 0 {
                self.update_simulation();
            }
        }
        
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,  // Dark gray background so we can see the simulation area
                            g: 0.1,
                            b: 0.1,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            
            // Use the current texture for display
            let display_bind_group = if self.current_is_a {
                &self.display_bind_group_a
            } else {
                &self.display_bind_group_b
            };
            
            render_pass.set_bind_group(0, display_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]); // Add camera bind group
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..6, 0..1);
            
            // Render FPS overlay if visible
            if self.fps_visible {
                if let Some(text_brush) = &mut self.text_brush {
                    let fps_text = format!("{:.1} / {:.1}", self.current_fps, self.fps_1_percent_low);
                    
                    let section = Section::default()
                        .add_text(Text::new(&fps_text).with_scale(20.0).with_color([1.0, 1.0, 1.0, 1.0]))
                        .with_screen_position((10.0, 10.0));
                    
                    match text_brush.queue(&self.device, &self.queue, [&section]) {
                        Ok(_) => {
                            text_brush.draw(&mut render_pass);
                        },
                        Err(e) => {
                            eprintln!("Text rendering error: {:?}", e);
                        }
                    }
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
    
    fn update_simulation(&mut self) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Game of Life Compute Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Game of Life Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.compute_pipeline);
            
            // Use the appropriate bind group for ping-pong
            let compute_bind_group = if self.current_is_a {
                &self.game_bind_group_a // A -> B
            } else {
                &self.game_bind_group_b // B -> A
            };
            
            compute_pass.set_bind_group(0, compute_bind_group, &[]);
            
            // Dispatch compute shader (256x256 texture with 8x8 workgroups)
            compute_pass.dispatch_workgroups(32, 32, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        
        // Flip ping-pong state
        self.current_is_a = !self.current_is_a;
    }
    
    // Generic function to modify cell values in the game world
    // Generic method to modify any cell data on any texture
    // This can be used for terrain editing, adding forces, spawning empires, etc.
    fn modify_cell(&mut self, x: u32, y: u32, alive: bool) {
        if x >= self.game_size || y >= self.game_size {
            return; // Out of bounds
        }
        
        println!("Modifying cell at ({}, {}) to {}", x, y, if alive { "alive" } else { "dead" });
        
        // Create the new cell data (Rgba8Unorm format: R=cell_state, G=B=A=0)
        let cell_data: [u8; 4] = if alive {
            [255, 0, 0, 255] // Alive: red pixel (R=255, G=0, B=0, A=255)
        } else {
            [0, 0, 0, 255]   // Dead: black pixel (R=0, G=0, B=0, A=255)
        };
        
        // Get the current texture (the one we're reading from in the simulation)
        let current_texture = if self.current_is_a {
            &self.game_texture_a
        } else {
            &self.game_texture_b
        };
        
        // Write directly to the current texture using queue.write_texture
        self.queue.write_texture(
            // Destination - use wgpu 26.0 TexelCopyTextureInfo
            wgpu::TexelCopyTextureInfo {
                texture: current_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x,
                    y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            // Data
            &cell_data,
            // Data layout - use wgpu 26.0 TexelCopyBufferLayout
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
        
        println!("  -> Successfully modified cell using wgpu 26.0 queue.write_texture API");
    }
    
    // ⚠️ WARNING: MOUSE COORDINATE SYSTEM IS WORKING PERFECTLY - DO NOT MODIFY! ⚠️
    // This coordinate conversion is correctly synchronized with the camera matrix
    // Convert screen coordinates to game world coordinates using camera transform
    fn screen_to_game_coords(&self, screen_x: f64, screen_y: f64) -> Option<(u32, u32)> {
        let window_width = self.size.width as f64;
        let window_height = self.size.height as f64;
        let aspect_ratio = window_width / window_height;
        
        // Convert screen coordinates to normalized device coordinates (-1 to 1)
        let ndc_x = (screen_x / window_width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen_y / window_height) * 2.0; // Flip Y for screen coordinates
        
        // Calculate the view bounds (same as in update_camera)
        let view_width = (self.game_size as f64) / (self.zoom_level as f64);
        let view_height = view_width / aspect_ratio;
        
        // Calculate the bounds of what we're seeing in world coordinates
        let left = (self.camera_x as f64) - view_width * 0.5;
        let right = (self.camera_x as f64) + view_width * 0.5;
        let bottom = (self.camera_y as f64) + view_height * 0.5;
        let top = (self.camera_y as f64) - view_height * 0.5;
        
        // Convert NDC to world coordinates using the same bounds as the camera
        let world_x = left + (ndc_x + 1.0) * 0.5 * (right - left);
        let world_y = bottom + (ndc_y + 1.0) * 0.5 * (top - bottom); // Fixed: use bottom + ... instead of top + ...
        
        // Check if coordinates are within the simulation bounds
        if world_x >= 0.0 && world_x < self.game_size as f64 && 
           world_y >= 0.0 && world_y < self.game_size as f64 {
            Some((world_x as u32, world_y as u32))
        } else {
            None
        }
    }
    
    fn toggle_pause(&mut self) {
        self.is_paused = !self.is_paused;
        println!("Simulation {}", if self.is_paused { "paused" } else { "resumed" });
    }
    
    // ⚠️ WARNING: CAMERA SYSTEM IS WORKING PERFECTLY - DO NOT MODIFY! ⚠️
    // This camera matrix calculation is correct and matches the mouse coordinate conversion
    fn update_camera(&mut self) {
        // Calculate proper orthographic projection matrix
        let aspect_ratio = self.size.width as f32 / self.size.height as f32;
        
        // Calculate the view bounds based on camera position and zoom
        let view_width = (self.game_size as f32) / self.zoom_level;
        let view_height = view_width / aspect_ratio;
        
        // Calculate the bounds of what we want to see in world coordinates
        let left = self.camera_x - view_width * 0.5;
        let right = self.camera_x + view_width * 0.5;
        let bottom = self.camera_y + view_height * 0.5; // Note: in our coordinate system, Y increases downward
        let top = self.camera_y - view_height * 0.5;
        
        // Create orthographic projection matrix that maps [left, right] x [bottom, top] to [-1, 1] x [-1, 1]
        let width = right - left;
        let height = top - bottom;
        
        // Orthographic projection matrix (column-major for WGSL)
        #[rustfmt::skip]
        let matrix = [
            2.0 / width, 0.0, 0.0, 0.0,                    // Column 0
            0.0, 2.0 / height, 0.0, 0.0,                   // Column 1  
            0.0, 0.0, 1.0, 0.0,                            // Column 2
            -(right + left) / width, -(top + bottom) / height, 0.0, 1.0,  // Column 3
        ];
        
        // Update the uniform buffer
        self.queue.write_buffer(
            &self.camera_uniform_buffer,
            0,
            bytemuck::cast_slice(&matrix),
        );
    }
    
    fn handle_camera_input(&mut self, dt: f32) {
        let camera_speed = 50.0 / self.zoom_level; // Move slower when zoomed in (reduced speed)
        let zoom_speed = 1.0; // Reduced zoom speed
        
        // WASD movement
        if self.keys_pressed.contains(&KeyCode::KeyW) {
            self.camera_y -= camera_speed * dt;
        }
        if self.keys_pressed.contains(&KeyCode::KeyS) {
            self.camera_y += camera_speed * dt;
        }
        if self.keys_pressed.contains(&KeyCode::KeyA) {
            self.camera_x -= camera_speed * dt;
        }
        if self.keys_pressed.contains(&KeyCode::KeyD) {
            self.camera_x += camera_speed * dt;
        }
        
        // Q/E zoom
        if self.keys_pressed.contains(&KeyCode::KeyQ) {
            self.zoom_level *= 1.0 + zoom_speed * dt; // Zoom out
            if self.zoom_level > 10.0 { self.zoom_level = 10.0; }
        }
        if self.keys_pressed.contains(&KeyCode::KeyE) {
            self.zoom_level *= 1.0 - zoom_speed * dt; // Zoom in
            if self.zoom_level < 0.1 { self.zoom_level = 0.1; }
        }
        
        // Keep camera within reasonable bounds
        let game_size = self.game_size as f32;
        let margin = game_size * 0.5; // Allow camera to go a bit outside the simulation
        self.camera_x = self.camera_x.clamp(-margin, game_size + margin);
        self.camera_y = self.camera_y.clamp(-margin, game_size + margin);
    }
    
    fn update_fps_tracking(&mut self) {
        let now = std::time::Instant::now();
        
        // Add current frame time
        self.frame_times.push(now);
        
        // Remove old frame times (keep only last 2 seconds worth)
        let cutoff_time = now - std::time::Duration::from_secs(2);
        self.frame_times.retain(|&time| time > cutoff_time);
        
        // Update FPS calculations every 0.25 seconds
        if now.duration_since(self.last_fps_update).as_secs_f32() > 0.25 {
            if self.frame_times.len() > 1 {
                // Calculate average FPS
                let total_duration = now.duration_since(self.frame_times[0]).as_secs_f32();
                self.current_fps = (self.frame_times.len() - 1) as f32 / total_duration;
                
                // Calculate 1% low (99th percentile of frame times)
                let mut frame_durations: Vec<f32> = Vec::new();
                for i in 1..self.frame_times.len() {
                    let duration = self.frame_times[i].duration_since(self.frame_times[i-1]).as_secs_f32();
                    frame_durations.push(duration);
                }
                
                if !frame_durations.is_empty() {
                    frame_durations.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Sort descending
                    let percentile_99_index = ((frame_durations.len() as f32 * 0.01).ceil() as usize).min(frame_durations.len() - 1);
                    let slowest_1_percent_duration = frame_durations[percentile_99_index];
                    self.fps_1_percent_low = 1.0 / slowest_1_percent_duration;
                }
            }
            self.last_fps_update = now;
        }
    }
    
    fn toggle_fps_overlay(&mut self) {
        self.fps_visible = !self.fps_visible;
        println!("FPS overlay {}", if self.fps_visible { "enabled" } else { "disabled" });
    }
    
    // TODO: Implement proper text rendering using wgpu_text crate
}

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
    cursor_position: Option<(f64, f64)>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        println!("Creating window...");
        
        let window_attributes = Window::default_attributes()
            .with_title("EmpiresGPU - Conway's Game of Life")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600));
        
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        self.window = Some(window.clone());
        
        println!("Window created successfully!");
        
        // Initialize GPU state asynchronously
        let state = pollster::block_on(State::new(window));
        self.state = Some(state);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("Close requested, exiting...");
                event_loop.exit();
            },
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key_code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            match key_code {
                                KeyCode::Escape => {
                                    println!("Escape pressed, exiting...");
                                    event_loop.exit();
                                },
                                KeyCode::Space => {
                                    if let Some(state) = &mut self.state {
                                        state.toggle_pause();
                                    }
                                },
                                KeyCode::F3 => {
                                    if let Some(state) = &mut self.state {
                                        state.toggle_fps_overlay();
                                    }
                                },
                                KeyCode::KeyW | KeyCode::KeyA | KeyCode::KeyS | KeyCode::KeyD |
                                KeyCode::KeyQ | KeyCode::KeyE => {
                                    // Track camera movement keys
                                    if let Some(state) = &mut self.state {
                                        state.keys_pressed.insert(key_code);
                                    }
                                },
                                _ => {}
                            }
                        },
                        ElementState::Released => {
                            // Remove key from pressed set
                            if let Some(state) = &mut self.state {
                                state.keys_pressed.remove(&key_code);
                            }
                        }
                    }
                }
            },
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_position = Some((position.x, position.y));
            },
            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                // Handle left mouse click to add living cells
                if let (Some(state_ref), Some((cursor_x, cursor_y))) = (&mut self.state, self.cursor_position) {
                    if let Some((game_x, game_y)) = state_ref.screen_to_game_coords(cursor_x, cursor_y) {
                        state_ref.modify_cell(game_x, game_y, true);
                    }
                }
            },
            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Right, .. } => {
                // Handle right mouse click to remove living cells
                if let (Some(state_ref), Some((cursor_x, cursor_y))) = (&mut self.state, self.cursor_position) {
                    if let Some((game_x, game_y)) = state_ref.screen_to_game_coords(cursor_x, cursor_y) {
                        state_ref.modify_cell(game_x, game_y, false);
                    }
                }
            },
            WindowEvent::Resized(physical_size) => {
                println!("Window resized to: {}x{}", physical_size.width, physical_size.height);
                if let Some(state) = &mut self.state {
                    state.resize(physical_size);
                }
            },
            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.state {
                    match state.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                        Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                        Err(e) => eprintln!("Render error: {:?}", e),
                    }
                }
            },
            _ => {}
        }
        
        // Request continuous redraws to keep the window responsive
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    println!("EmpiresGPU: Conway's Game of Life GPU Simulation with Camera System");
    println!("Controls:");
    println!("  SPACE - Pause/Resume simulation");
    println!("  WASD - Move camera around");
    println!("  Q/E - Zoom out/in");
    println!("  F3 - Toggle FPS overlay");
    println!("  Left Click - Make cell alive");
    println!("  Right Click - Make cell dead");
    println!("  ESC - Exit");
    
    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        window: None,
        state: None,
        cursor_position: None,
    };

    println!("Starting Game of Life simulation...");
    event_loop.run_app(&mut app).unwrap();
    println!("Simulation finished.");
}