use std::collections::HashSet;
use winit::keyboard::KeyCode;

pub struct Camera {
    pub x: f32,
    pub y: f32,
    pub zoom_level: f32,
    pub keys_pressed: HashSet<KeyCode>,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl Camera {
    pub fn new(device: &wgpu::Device, _game_size: u32) -> Self {
        // Create camera uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniform Buffer"),
            size: 64, // mat4x4<f32> = 16 floats * 4 bytes = 64 bytes
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create camera bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        Self {
            x: _game_size as f32 / 2.0,  // Center horizontally on the map
            y: _game_size as f32 / 2.0,  // Center vertically on the map
            zoom_level: 0.5, // Start zoomed out to see the full simulation
            keys_pressed: HashSet::new(),
            uniform_buffer,
            bind_group,
        }
    }
    
    pub fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        })
    }
    
    pub fn handle_input(&mut self, dt: f32, game_size: f32) {
        let camera_speed = 50.0 / self.zoom_level; // Move slower when zoomed in
        let zoom_speed = 1.0;
        
        // WASD movement
        if self.keys_pressed.contains(&KeyCode::KeyW) {
            self.y -= camera_speed * dt;
        }
        if self.keys_pressed.contains(&KeyCode::KeyS) {
            self.y += camera_speed * dt;
        }
        if self.keys_pressed.contains(&KeyCode::KeyA) {
            self.x -= camera_speed * dt;
        }
        if self.keys_pressed.contains(&KeyCode::KeyD) {
            self.x += camera_speed * dt;
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
        let margin = game_size * 0.5; // Allow camera to go a bit outside the simulation
        self.x = self.x.clamp(-margin, game_size + margin);
        self.y = self.y.clamp(-margin, game_size + margin);
    }
    
    // ⚠️ WARNING: CAMERA SYSTEM IS WORKING PERFECTLY - DO NOT MODIFY! ⚠️
    // This camera matrix calculation is correct and matches the mouse coordinate conversion
    pub fn update_matrix(&self, queue: &wgpu::Queue, window_size: winit::dpi::PhysicalSize<u32>, game_size: f32) {
        // Calculate proper orthographic projection matrix
        let aspect_ratio = window_size.width as f32 / window_size.height as f32;
        
        // Calculate the view bounds based on camera position and zoom
        let view_width = game_size / self.zoom_level;
        let view_height = view_width / aspect_ratio;
        
        // Calculate the bounds of what we want to see in world coordinates
        let left = self.x - view_width * 0.5;
        let right = self.x + view_width * 0.5;
        let bottom = self.y + view_height * 0.5; // Note: in our coordinate system, Y increases downward
        let top = self.y - view_height * 0.5;
        
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
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&matrix),
        );
    }
    
    // ⚠️ WARNING: MOUSE COORDINATE SYSTEM IS WORKING PERFECTLY - DO NOT MODIFY! ⚠️
    // This coordinate conversion is correctly synchronized with the camera matrix
    pub fn screen_to_game_coords(&self, screen_x: f64, screen_y: f64, window_size: winit::dpi::PhysicalSize<u32>, game_size: f32) -> Option<(u32, u32)> {
        let window_width = window_size.width as f64;
        let window_height = window_size.height as f64;
        let aspect_ratio = window_width / window_height;
        
        // Convert screen coordinates to normalized device coordinates (-1 to 1)
        let ndc_x = (screen_x / window_width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen_y / window_height) * 2.0; // Flip Y for screen coordinates
        
        // Calculate the view bounds (same as in update_matrix)
        let view_width = (game_size as f64) / (self.zoom_level as f64);
        let view_height = view_width / aspect_ratio;
        
        // Calculate the bounds of what we're seeing in world coordinates
        let left = (self.x as f64) - view_width * 0.5;
        let right = (self.x as f64) + view_width * 0.5;
        let bottom = (self.y as f64) + view_height * 0.5;
        let top = (self.y as f64) - view_height * 0.5;
        
        // Convert NDC to world coordinates using the same bounds as the camera
        let world_x = left + (ndc_x + 1.0) * 0.5 * (right - left);
        let world_y = bottom + (ndc_y + 1.0) * 0.5 * (top - bottom);
        
        // Check if coordinates are within the simulation bounds
        if world_x >= 0.0 && world_x < game_size as f64 && 
           world_y >= 0.0 && world_y < game_size as f64 {
            Some((world_x as u32, world_y as u32))
        } else {
            None
        }
    }
}
