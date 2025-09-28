use glyph_brush::{Section, Text};
use std::time::Instant;

pub struct FpsTracker {
    pub visible: bool,
    pub frame_times: Vec<Instant>,
    pub last_fps_update: Instant,
    pub current_fps: f32,
    pub fps_1_percent_low: f32,
    pub text_brush: Option<wgpu_text::TextBrush<glyph_brush::ab_glyph::FontRef<'static>>>,
}

impl FpsTracker {
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        println!("Creating text rendering system...");
        let font_data = include_bytes!("../../assets/fonts/DejaVuSans.ttf");
        let text_brush = wgpu_text::BrushBuilder::using_font_bytes(font_data)
            .unwrap()
            .build(device, config.width, config.height, config.format);
        println!("Text rendering system created successfully!");

        let now = Instant::now();
        
        Self {
            visible: true,
            frame_times: Vec::with_capacity(120), // Store up to 120 frame times (2 seconds at 60fps)
            last_fps_update: now,
            current_fps: 0.0,
            fps_1_percent_low: 0.0,
            text_brush: Some(text_brush),
        }
    }
    
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>, queue: &wgpu::Queue) {
        if let Some(text_brush) = &mut self.text_brush {
            text_brush.resize_view(new_size.width as f32, new_size.height as f32, queue);
        }
    }
    
    pub fn update(&mut self) {
        let now = Instant::now();
        
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
    
    pub fn toggle_visibility(&mut self) {
        self.visible = !self.visible;
        println!("FPS overlay {}", if self.visible { "enabled" } else { "disabled" });
    }
    
    pub fn render<'a>(&'a mut self, render_pass: &mut wgpu::RenderPass<'a>, device: &wgpu::Device, queue: &wgpu::Queue) {
        if !self.visible {
            return;
        }
        
        if let Some(text_brush) = &mut self.text_brush {
            let fps_text = format!("{:.1} / {:.1}", self.current_fps, self.fps_1_percent_low);
            
            let section = Section::default()
                .add_text(Text::new(&fps_text).with_scale(20.0).with_color([1.0, 1.0, 1.0, 1.0]))
                .with_screen_position((10.0, 10.0));
            
            match text_brush.queue(device, queue, [&section]) {
                Ok(_) => {
                    text_brush.draw(render_pass);
                },
                Err(e) => {
                    eprintln!("Text rendering error: {:?}", e);
                }
            }
        }
    }
}
