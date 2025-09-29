use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 2],
    pub tex_coords: [f32; 2],
}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
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
        }
    }
}

pub struct GraphicsContext {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
}

impl GraphicsContext {
    pub async fn new(window: std::sync::Arc<winit::window::Window>) -> Self {
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
            
        // Query and display GPU limits
        let limits = device.limits();
        println!("GPU Device initialized successfully!");
        println!("GPU Limits:");
        println!("  Max texture dimension 2D: {}", limits.max_texture_dimension_2d);
        println!("  Max texture dimension 3D: {}", limits.max_texture_dimension_3d);
        println!("  Max compute workgroups per dimension: {}", limits.max_compute_workgroups_per_dimension);
        println!("  Max compute workgroup size X: {}", limits.max_compute_workgroup_size_x);
        println!("  Max compute workgroup size Y: {}", limits.max_compute_workgroup_size_y);
        
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
            present_mode: wgpu::PresentMode::Immediate, // Disable vsync for maximum FPS
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        
        Self {
            surface,
            device,
            queue,
            config,
            size,
        }
    }
    
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }
    
    pub fn create_quad_vertices(sim_size: f32) -> [Vertex; 6] {
        [
            Vertex { position: [0.0, sim_size], tex_coords: [0.0, 1.0] }, // Bottom-left
            Vertex { position: [sim_size, sim_size], tex_coords: [1.0, 1.0] }, // Bottom-right
            Vertex { position: [sim_size, 0.0], tex_coords: [1.0, 0.0] }, // Top-right
            Vertex { position: [0.0, sim_size], tex_coords: [0.0, 1.0] }, // Bottom-left
            Vertex { position: [sim_size, 0.0], tex_coords: [1.0, 0.0] }, // Top-right
            Vertex { position: [0.0, 0.0], tex_coords: [0.0, 0.0] }, // Top-left
        ]
    }
    
    pub fn create_vertex_buffer(&self, vertices: &[Vertex]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        })
    }
}

pub fn load_shader(device: &wgpu::Device, path: &str) -> wgpu::ShaderModule {
    let shader_source = match path {
        "shaders/empire_compute.wgsl" => include_str!("../../shaders/empire_compute.wgsl"),
        "shaders/empire_render.wgsl" => include_str!("../../shaders/empire_render.wgsl"),
        "shaders/empire_render_strength.wgsl" => include_str!("../../shaders/empire_render_strength.wgsl"),
        "shaders/empire_render_need.wgsl" => include_str!("../../shaders/empire_render_need.wgsl"),
        "shaders/empire_render_action.wgsl" => include_str!("../../shaders/empire_render_action.wgsl"),
        "shaders/terrain_render.wgsl" => include_str!("../../shaders/terrain_render.wgsl"),
        _ => panic!("Unknown shader path: {}", path),
    };
    
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(path),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    })
}
