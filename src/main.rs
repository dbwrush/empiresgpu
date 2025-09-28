// EmpiresGPU - Empire Simulation
// A GPU-based cellular automata empire simulation using compute shaders and hex grids

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{WindowEvent, ElementState, MouseButton},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

mod graphics;
mod camera;
mod simulation;
mod ui;

use graphics::GraphicsContext;
use camera::Camera;
use simulation::EmpireSimulation;
use ui::FpsTracker;

// Centralized simulation size configuration
// Change this value to easily adjust the simulation grid size (e.g., 128, 256, 512)
// The system will automatically handle camera centering, workgroup dispatch, and vertex buffer sizing
// Note: GPU texture size limits typically cap this at 8192x8192. Larger values will be constrained.
// For truly large simulations (>8192), see SCALING.md for storage buffer implementation approaches.
const SIMULATION_SIZE: u32 = 1024;

struct State {
    graphics: GraphicsContext,
    camera: Camera,
    simulation: EmpireSimulation,
    fps_tracker: FpsTracker,
    vertex_buffer: wgpu::Buffer,
    cursor_position: Option<(f64, f64)>,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        // Initialize graphics context
        let graphics = GraphicsContext::new(window).await;
        
        // Create camera with bind group layout for the simulation
        let camera_bind_group_layout = Camera::create_bind_group_layout(&graphics.device);
        
        // Create simulation (may constrain size based on GPU limits)
        let simulation = EmpireSimulation::new(&graphics, &camera_bind_group_layout, SIMULATION_SIZE);
        
        // Create camera with the actual simulation size (may be constrained)
        let camera = Camera::new(&graphics.device, simulation.game_size);
        
        // Create FPS tracker
        let fps_tracker = FpsTracker::new(&graphics.device, &graphics.config);
        
        // Create vertex buffer for rendering
        let sim_size = simulation.game_size as f32;
        let vertices = GraphicsContext::create_quad_vertices(sim_size);
        let vertex_buffer = graphics.create_vertex_buffer(&vertices);
        
        Self {
            graphics,
            camera,
            simulation,
            fps_tracker,
            vertex_buffer,
            cursor_position: None,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.graphics.resize(new_size);
        self.fps_tracker.resize(new_size, &self.graphics.queue);
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // Update FPS tracking
        self.fps_tracker.update();
        
        // Handle camera input (using fixed timestep for now)
        let dt = 1.0 / 60.0; // Assume 60 FPS
        self.camera.handle_input(dt, self.simulation.game_size as f32);
        
        // Update camera matrix
        self.camera.update_matrix(&self.graphics.queue, self.graphics.size, self.simulation.game_size as f32);
        
        // Update simulation
        self.simulation.update(&self.graphics);
        
        let output = self.graphics.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.graphics.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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

            // Render simulation (terrain + empires)
            self.simulation.render(&mut render_pass, &self.vertex_buffer, &self.camera.bind_group);
            
            // Render FPS overlay
            self.fps_tracker.render(&mut render_pass, &self.graphics.device, &self.graphics.queue);
        }

        self.graphics.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
    
    fn claim_cell_for_empire(&mut self, x: u32, y: u32, empire_id: u8) {
        self.simulation.claim_cell_for_empire(x, y, empire_id, &self.graphics);
    }
    
    fn toggle_pause(&mut self) {
        self.simulation.toggle_pause();
    }
    
    fn toggle_fps_overlay(&mut self) {
        self.fps_tracker.toggle_visibility();
    }
}

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        println!("Creating window...");
        
        let window_attributes = Window::default_attributes()
            .with_title("EmpiresGPU - Empire Simulation")
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
                                        state.camera.keys_pressed.insert(key_code);
                                    }
                                },
                                _ => {}
                            }
                        },
                        ElementState::Released => {
                            // Remove key from pressed set
                            if let Some(state) = &mut self.state {
                                state.camera.keys_pressed.remove(&key_code);
                            }
                        }
                    }
                }
            },
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(state) = &mut self.state {
                    state.cursor_position = Some((position.x, position.y));
                }
            },
            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                // Handle left mouse click to claim cell for Empire 1
                if let Some(state_ref) = &mut self.state {
                    if let Some((cursor_x, cursor_y)) = state_ref.cursor_position {
                        if let Some((game_x, game_y)) = state_ref.camera.screen_to_game_coords(cursor_x, cursor_y, state_ref.graphics.size, state_ref.simulation.game_size as f32) {
                            state_ref.claim_cell_for_empire(game_x, game_y, 1); // Empire 1
                        }
                    }
                }
            },
            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Right, .. } => {
                // Handle right mouse click to unclaim cell (set to Empire 0)
                if let Some(state_ref) = &mut self.state {
                    if let Some((cursor_x, cursor_y)) = state_ref.cursor_position {
                        if let Some((game_x, game_y)) = state_ref.camera.screen_to_game_coords(cursor_x, cursor_y, state_ref.graphics.size, state_ref.simulation.game_size as f32) {
                            state_ref.claim_cell_for_empire(game_x, game_y, 0); // Unclaim (Empire 0)
                        }
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
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.graphics.size),
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
    println!("EmpiresGPU: Empire Simulation with Camera System");
    println!("Controls:");
    println!("  SPACE - Pause/Resume simulation");
    println!("  WASD - Move camera around");
    println!("  Q/E - Zoom out/in");
    println!("  F3 - Toggle FPS overlay");
    println!("  Left Click - Claim territory for Empire 1 (Red)");
    println!("  Right Click - Unclaim territory");
    println!("  ESC - Exit");
    
    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        window: None,
        state: None,
    };

    println!("Starting Empire simulation...");
    event_loop.run_app(&mut app).unwrap();
    println!("Simulation finished.");
}