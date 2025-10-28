use bevy::window::PrimaryWindow;
use bevy::{prelude::*, utils::HashMap};
use noise::{NoiseFn, Simplex};
use rand::Rng;
use rayon::prelude::*;
use std::time::Instant;
use std::env;
use std::sync::Mutex; // Import Mutex for thread-safe updates

const WIDTH: usize = 16 * 30;
const HEIGHT: usize = 9 * 30;
const VARIABLES: usize = 4; // Terrain, strength, empire
const OCEAN_CUTOFF: f32 = 0.5;
const EMPIRE_PROBABILITY: i32 = 1;
const TERRAIN_NEED: f32 = 0.99;
const TERRAIN_STRENGTH: f32 = 0.7;
const LOOP_DIST: usize = 10;
const BOAT_PROP: f32 = 0.01;
const TECH_GAIN: f32 = 0.001;
const START_TECH_RANGE: f32 = 0.01;
const MIN_BOAT_WAIT: u32 = 2;
const MAX_TECH: f32 = 0.2;
const TECH_DECAY: f32 = 0.000001;

fn main() {
    env::set_var("RUST_BACKTRACE", "full");
    let mut app = App::new();
    app.add_plugins(DefaultPlugins);
    app.add_systems(Startup, setup);
    app.add_systems(Update, (update_colors, draw_fps, update_render_mode_system, update_empires, update_camera_system));
    app.add_systems(PreUpdate, (update_boats_system.before(pull_system), pull_system.before(update_cell_map_system), update_cell_map_system));
    app.add_systems(PostUpdate, (push_system.before(update_cell_map_system), update_cell_map_system));
    app.insert_resource(RenderMode::AgeView);
    app.insert_resource(GameData { max_strength: 0.0 , max_age: 0, send_boats: false });
    app.insert_resource(MapData(HashMap::default(), Vec::new()));
    app.run();
}

fn setup(mut commands: Commands, mut windows: Query<&mut Window, With<PrimaryWindow>>, mut entity_map: ResMut<MapData>) {
    let window_width = windows.iter().next().unwrap().width();
    let window_height = windows.iter().next().unwrap().height();
    let scale_x = WIDTH as f32 / window_width;
    let scale_y = HEIGHT as f32 / window_height;
    let scale = scale_x.max(scale_y);
    if let Ok(mut window) = windows.get_single_mut() {
        window.title = "Empires!".to_string();
    }


    commands.spawn(Camera2dBundle {
        transform: Transform::from_xyz(WIDTH as f32 / 2.0, HEIGHT as f32 / 2.0, 100.0),
        projection: OrthographicProjection {
            scale: scale,
            ..Default::default()
        },
        ..Default::default()
    });

    let grid = Grid::new(WIDTH, HEIGHT, VARIABLES);

    commands.spawn(TextBundle {
        text: Text::from_section(
            "FPS: 0.00",
            TextStyle {
                font_size: 30.0,
                color: Color::WHITE,
                ..Default::default()
            }
        ).with_justify(JustifyText::Right),
        transform: Transform::from_xyz(window_width / 2.0 - 10.0, window_height / 2.0 - 10.0, 0.0),
        ..Default::default()
    });

    commands.insert_resource(LastDraw::default());

    let mut count = 0;

    // Initialize sprites
    let mut empire_count = 0;
    for x in 0..WIDTH {
        for y in 0..HEIGHT {
            let terrain = grid.data[x][y][0];
            let offset:f32 = (y % 2) as f32 / 2.0; //offset every other row by 0.5 for a hex grid.
            commands.spawn(SpriteBundle {
                sprite: Sprite {
                    color: Color::WHITE,
                    custom_size: Some(Vec2::new(1.0, 1.0)),
                    ..Default::default()
                },
                transform: Transform::from_xyz(x as f32 + offset, y as f32, 0.0),
                ..Default::default()
            }).insert(CellMarker);
            if terrain > OCEAN_CUTOFF {
                // chance to spawn an empire using cell.set_empire()
                let mut empire = -1;
                if rand::thread_rng().gen_range(0..EMPIRE_PROBABILITY) < 1 {
                    empire = empire_count;
                    empire_count += 1;
                    //println!("Empire {} has been created at ({}, {})", empire, x, y);
                    let starting_tech = rand::thread_rng().gen_range(0.0..START_TECH_RANGE);
                    entity_map.1.push((rand::thread_rng().gen_range(0..360) as f32, rand::thread_rng().gen_range(0..1000) as f32 / 1000.0, rand::thread_rng().gen_range(0..1000) as f32 / 1000.0, starting_tech));
                }
                count += 1;

                commands.spawn(Cell::new(x, y, terrain, empire));
                entity_map.0.insert((x, y), ((x, y), empire, 0.0, 0.0, (0, 0), 0.0, empire, 0, HashMap::new(), 0.0));
            }
        }
    }
    println!("{} cells created", count);
    commands.insert_resource(grid);
}

#[derive(Resource)]
struct MapData(HashMap<(usize, usize), ((usize, usize), i32, f32, f32, (usize, usize), f32, i32, u32, HashMap<(usize, usize), (i32, f32)>, f32)>, Vec<(f32, f32, f32, f32)>);
//vec is empire data including hue, saturation, aggression, and tech factor


#[derive(Resource)]
struct GameData {
    max_strength: f32,
    max_age: u32,
    send_boats: bool,
}

#[derive(Resource)]
struct Grid {
    data: Vec<Vec<Vec<f32>>>,
}

impl Grid {
    fn new(width: usize, height: usize, variables: usize) -> Self {
        let mut data = vec![vec![vec![0.0; variables]; height]; width];
        let mut rng = rand::thread_rng();
        let noise = Simplex::new(rng.gen::<u32>()); //Billow<_> = Billow::new(rng.gen::<u32>());
        let noise2 = Simplex::new(rng.gen::<u32>());
        let noise3 = Simplex::new(rng.gen::<u32>());
        
        data.par_iter_mut().enumerate().for_each(|(x, row)| {
            row.iter_mut().enumerate().for_each(|(y, cell)| {
                let mut elevation = get_elevation(noise, noise2, noise3, x, y);
                if x < LOOP_DIST || x > WIDTH - LOOP_DIST {
                    //at the edge, terrain will be the average of this and the opposite side.
                    let mut opp_prop = x as f32 / LOOP_DIST as f32 * -0.5 + 0.5;
                    let opp_x = WIDTH - x - 1;
                    if x > WIDTH - LOOP_DIST {
                        opp_prop = opp_x as f32 / LOOP_DIST as f32 * -0.5 + 0.5;
                    }
                    let opp_elevation = get_elevation(noise, noise2, noise3, opp_x, y);
                    elevation = elevation * (1.0 - opp_prop) + opp_elevation * opp_prop;
                }
                cell[0] = elevation;
            });
        });

        Grid { data}
    }
}

fn get_elevation(noise: Simplex, noise2: Simplex, noise3: Simplex, x: usize, y: usize) -> f32 {
    let mut x = x as f32;
    if y % 2 == 1 {
        //offset x by .5;
        x += 0.5;
    }
    let mut e = noise.get([x as f64 / 128.0, y as f64 / 128.0]) as f32 * 16.0 + 
    noise.get([x as f64 / 64.0, y as f64 / 64.0]) as f32 * 8.0 + 
    noise2.get([x as f64 / 32.0, y as f64 / 32.0]) as f32 * 4.0 + 
    noise3.get([x as f64 / 16.0, y as f64 / 16.0]) as f32 * 2.0 + 
    noise3.get([x as f64 / 8.0, y as f64 / 8.0]) as f32;
    e /= 64.0;
    e += 0.5;
    if e > 1.0 || e < 0.0 {
        println!("WTF? Exceed at ({}, {}) with {}", x, y, e);
    }
    e
}

#[derive(Component)]
struct CellMarker;

#[derive(Component)]
struct Boat {
    direction: u8,
    strength: f32,
    empire: i32,
    tech_bonus: f32, // New field for tech influence
}

impl Boat {
    fn new(direction: u8, strength: f32, empire: i32, tech: f32) -> Self {
        Boat {
            direction,
            strength: strength * (1.0 + tech), // Scale strength by tech
            empire,
            tech_bonus: tech,
        }
    }

    fn move_boat(&mut self, mut position: (i32, i32)) -> (i32, i32) {
        //use direction values to decide if boat moves on one axis or both randomly
        //if direction value is negative, that means the boat is moving in the negative direction
        let mut use_direction = self.direction;
        //small chance to add or remove 1 from use_direction;
        if rand::thread_rng().gen_range(0..10) < 1 {
            use_direction += 1;
            if use_direction > 5 {
                use_direction = 0;
            }
        } else if rand::thread_rng().gen_range(0..10) < 1 {
            use_direction -= 1;
            if use_direction > 5 {
                use_direction = 5;
            }
        }
        
                /*EVEN
                (-1, -1) => 0,
                (0, -1) => 1,
                (1, 0) => 3,
                (0, 1) => 5,
                (-1, 1) => 4,
                (-1, 0) => 2,
                _ => 0, */

                /*ODD
                (0, -1) => 0,
                (1, -1) => 1,
                (1, 0) => 3,
                (1, 1) => 5,
                (0, 1) => 4,
                (-1, 0) => 2,
                _ => 0, */

        //check if we're on an even or odd row (due to hexagonal grid)
        if position.1 % 2 == 0 {//even
            match use_direction {
                0 => { position.0 -= 1; position.1 -= 1; } // (x-1, y-1)
                1 => { position.1 -= 1; }                  // (x, y-1)
                2 => { position.0 += 1; }                  // (x-1, y)
                3 => { position.1 += 1; }                  // (x+1, y)
                4 => { position.0 -= 1; position.1 += 1; } // (x-1, y+1)
                5 => { position.0 -= 1; }                  // (x, y+1)
                _ => {}
            }
        } else {//odd
            match use_direction {
                0 => { position.1 -= 1; }                  // (x, y-1)
                1 => { position.0 += 1; position.1 -= 1; } // (x+1, y-1)
                2 => { position.0 -= 1; }                  // (x-1, y)
                3 => { position.0 += 1; position.1 += 1; }                  // (x+1, y)
                4 => { position.1 += 1; }                  // (x, y+1)
                5 => { position.0 -= 1;} // (x+1, y+1)
                _ => {}
            }
        }
        return (position.0, position.1);
    }
}

#[derive(Component)]
struct Cell {
    position: (usize, usize),
    empire: i32,
    strength: f32,
    need: f32,
    boat_need: f32,
    send_target: (usize, usize),
    send_amount: f32,
    send_empire: i32,
    terrain: f32,
    age: u32,
    ocean_need_prop: f32,
    boat_target: (usize, usize),
    boat_strength: f32,
    terrain_factor: f32,
    need_factor: f32,
    last_boat: u32,
}

impl Cell {
    fn new(x: usize, y: usize, terrain: f32, empire: i32) -> Self {
        let c = Cell {            
            position: (x, y),
            empire,
            strength: terrain,
            need: 0.0,
            boat_need: 0.0,
            send_target: (x, y),
            send_amount: 0.0,
            send_empire: empire,
            terrain,
            age: 0,
            ocean_need_prop: 0.0,
            boat_target: (0, 0),
            boat_strength: 0.0,
            terrain_factor: ((1.0 - ((terrain - OCEAN_CUTOFF) / (1.0 - OCEAN_CUTOFF))).powf(1.0 + 4.0 * TERRAIN_STRENGTH) * TERRAIN_STRENGTH + (1.0 - TERRAIN_STRENGTH)),
            need_factor: (((-terrain) / (1.0 - OCEAN_CUTOFF)) + (1.0 / (1.0 - OCEAN_CUTOFF))) * TERRAIN_NEED + (1.0 - TERRAIN_NEED),
            last_boat: 0,
        };
        c
    }

    fn get(& self) -> ((usize, usize), i32, f32, f32, (usize, usize), f32, i32, u32, HashMap<(usize, usize), (i32, f32)>, f32) {
        //0 = position, 1 = empire, 2 = strength, 3 = need, 4 = send_target, 5 = send_amount, 6 = send_empire
        (self.position, self.empire, self.strength, self.need, self.send_target, self.send_amount, self.send_empire, self.age, HashMap::new(), self.boat_need)
    }

    //neighbors are the 8 cells surrounding this cell, accessible through the hashmap.
    fn push(&mut self, data: Vec<((usize, usize), i32, f32, f32, (usize, usize), f32, i32)>, aggression: f32, coastlines: Vec<(usize, usize)>) {//I call this 'push' because the cell is reading data from neighbors and pushing a decision
        let mut max_enemy_strength = 0.0;
        let mut max_need = 0.0;
        let mut max_need_position = self.position;
        let mut min_enemy_strength = 0.0;
        let mut min_enemy_position = self.position;
        self.boat_need *= 0.99;
        self.need = (self.boat_need as f32).sqrt() / 100.0;
        self.send_target = self.position;
        self.send_amount = 0.0;
        self.send_empire = self.empire;
        self.boat_strength = 0.0;
        self.last_boat += 1;

        if self.empire == -1 {
            return;
        } else {
            self.boat_need += (coastlines.len() as f32)/self.age as f32 / 10.0;
        }
        let mut friendly_neighbors = 0;
        let mut enemy_neighbors = 0;

        for i in 0..data.len() {
            if let Some(neighbor_cell) = data.get(i) {
                if neighbor_cell.0 == self.position {
                    continue;
                }
                if self.empire == neighbor_cell.1 {
                    if neighbor_cell.3 > max_need {
                        max_need = neighbor_cell.3;
                        max_need_position = neighbor_cell.0;
                    }
                    friendly_neighbors += 1;
                } else {
                    enemy_neighbors += 1;
                    if neighbor_cell.2 > max_enemy_strength {
                        max_enemy_strength = neighbor_cell.2;
                    }
                    if neighbor_cell.2 < min_enemy_strength || min_enemy_position == self.position {
                        min_enemy_strength = neighbor_cell.2;
                        min_enemy_position = neighbor_cell.0;
                    }
                    if neighbor_cell.1 != self.empire {
                        self.need += 1.0 * neighbor_cell.2;
                        if neighbor_cell.1 == -1 {
                            self.need -= 0.9 * neighbor_cell.2;
                        }
                    }
                }
            }
        }

        if friendly_neighbors == 0 && rand::thread_rng().gen_range(0..10) < 1 {
            //destroy empire
            self.empire = -1;
            return;
        }

        let extra = self.strength - max_enemy_strength / 3.0;
        if extra > 0.0 {
            if extra > (3.0 * (1.0 - aggression)) * min_enemy_strength && min_enemy_position != self.position {
                self.send_target = min_enemy_position;
                self.send_amount = extra;
            } else if max_need > 0.0 && max_need_position != self.position{
                (self.send_target.0, self.send_target.1) = (max_need_position.0, max_need_position.1);
                self.send_amount = extra * 0.5;
            }
        }
        if enemy_neighbors > 0 {
            self.need /= enemy_neighbors as f32;
        }
        /*if self.need > self.strength {
            self.need -= self.strength;
        }*/
        self.need += max_need * 0.9;
        self.need *= self.need_factor;
        self.strength -= self.send_amount;
        if self.last_boat > MIN_BOAT_WAIT && coastlines.len() > 0 && self.boat_need > 1.0 && (self.strength > 1.0 / BOAT_PROP || rand::thread_rng().gen_range(0..1000) < 1) {
            self.boat_target = coastlines[rand::thread_rng().gen_range(0..coastlines.len())];
            self.boat_strength = self.strength * self.ocean_need_prop;
            self.boat_strength = self.boat_strength.max(self.strength);
            self.strength -= self.boat_strength;
            self.last_boat = 0;
            //println!("Attempting to launch boat from ({}, {}) with strength {}", self.position.0, self.position.1, self.boat_strength);
        }
        self.strength *= (coastlines.len() + friendly_neighbors) as f32 / 6.0;
    }

    fn pull(&mut self, data: Vec<((usize, usize), i32, f32, f32, (usize, usize), f32, i32)>, tech: f32, boat_attacks: f32) {//I call this 'pull' because the cell is pulling the decisions from other cells to update its own data
        // Check the send_ variables of all neighbors to see if they are sending strength to this cell
        //self.empire = grid_data.0;
        //self.strength = grid_data.1;

        for i in 0..data.len() {// First add reinforcements from friendly cells to this cell's strength
            if let Some(neighbor_cell) = data.get(i) {
                if neighbor_cell.6 == self.empire && neighbor_cell.4 == self.position {
                    self.strength += neighbor_cell.5;
                }
            }
        }
        // Then divide incoming strength from enemy cells by 3 and subtract it from this cell's strength. Handle attacks from weakest to strongest.
        // If an attack causes strength to go below 0, change this cell's owner to the attacking empire and multiply strength by -1, all further attacks will be considered reinforcements
        for i in 0..data.len() {
            if let Some(neighbor_cell) = data.get(i) {
                if neighbor_cell.6 != self.empire && neighbor_cell.4 == self.position && neighbor_cell.1 != -1 {
                    //println!("Empire {} is attacking cell ({}, {}) from ({}, {})", neighbor_cell.6, self.position.0, self.position.1, neighbor_cell.0.0, neighbor_cell.0.1);
                    if self.strength - neighbor_cell.5 / 3.0 < 0.0 {
                        self.age = 0;
                        //set boat need to be based on the number of coastline neighbors (i.e., 6 - data.len())
                        self.boat_need = 6.0 - data.len() as f32;
                        self.empire = neighbor_cell.6;
                        //println!("Empire {} has taken cell ({}, {})", self.empire, self.position.0, self.position.1);
                        self.strength = neighbor_cell.5 / 3.0 - self.strength;
                    } else {
                        self.strength -= neighbor_cell.5 / 3.0;
                    }
                }
            }
        }
        if self.empire != -1 {
            // Use terrain data from the grid to determine how much strength this cell should generate. The closer to ocean level, the more strength is made.
            self.strength += (self.terrain_factor + tech.powf(2.0)).min(1.0);
            // Multiply strength by 0.99 so it can't just go up forever.
            self.strength *= (self.terrain_factor + tech.powf(2.0)).min(1.0);
            self.boat_need += boat_attacks;
            self.age += 1;
        }
    }
}

fn push_system(mut query: Query<&mut Cell>, cell_map: Res<MapData>) {
    //println!("Pushing");

    //track start time of push
    //let start = Instant::now();

    query.par_iter_mut().for_each(|mut cell| {//iterate through all cells on many threads
        let position = cell.position;//get cell's position
        let mut data = Vec::new();//initialize data to be sent to cell.push
        let mut ocean = Vec::new();
        for i in 0..6 {//iterate through the 6 possible neighbor positions
            let (mut neighbor_x, mut neighbor_y): (i32, i32) = (position.0 as i32, position.1 as i32);
            if position.1 % 2 == 0 { // even row
                match i {
                    0 => { neighbor_x -= 1; neighbor_y -= 1; } // (x-1, y-1)
                    1 => { neighbor_y -= 1; }                  // (x, y-1)
                    2 => { neighbor_x -= 1; }                  // (x-1, y)
                    3 => { neighbor_x += 1; }                  // (x+1, y)
                    4 => { neighbor_x -= 1; neighbor_y += 1; } // (x-1, y+1)
                    5 => { neighbor_y += 1; }                  // (x, y+1)
                    _ => {}
                }
            } else { // odd row
                match i {
                    0 => { neighbor_y -= 1; }                  // (x, y-1)
                    1 => { neighbor_x += 1; neighbor_y -= 1; } // (x+1, y-1)
                    2 => { neighbor_x -= 1; }                  // (x-1, y)
                    3 => { neighbor_x += 1; }                  // (x+1, y)
                    4 => { neighbor_y += 1; }                  // (x, y+1)
                    5 => { neighbor_x += 1; neighbor_y += 1; } // (x+1, y+1)
                    _ => {}
                }
            }
            if neighbor_x < 0 {
                neighbor_x += WIDTH as i32;
            }
            if neighbor_x >= WIDTH as i32 {
                neighbor_x -= WIDTH as i32;
            }
            if (neighbor_y as usize) >= HEIGHT || neighbor_y < 0 {
                continue;
            }
            if cell.position == (neighbor_x as usize, neighbor_y as usize) {
                continue;
            }
            if let Some(neighbor) = cell_map.0.get(&(neighbor_x as usize, neighbor_y as usize)).clone() {
                data.push((neighbor.0, neighbor.1, neighbor.2, neighbor.3, neighbor.4, neighbor.5, neighbor.6));
            } else {
                //neighbor is in the map but isn't in the hashmap, so it's ocean
                ocean.push((neighbor_x as usize, neighbor_y as usize));
            }
        }
        let mut aggression = 0.0;
        if cell.empire != -1 {
            aggression = cell_map.1[cell.empire as usize].2;
        }
        //println!("Pushed {} neighbors to cell at ({}, {})", data.len(), position.0, position.1);
        cell.push(data, aggression, ocean);
    });

    //print time duration of push
    //println!("Push took {:?}", start.elapsed());

    //println!("Pushed");
}

//iterate through all cells, run the get() function on them to update CellMap
fn update_cell_map_system(mut commands: Commands, mut cell_map: ResMut<MapData>, mut game_data: ResMut<GameData>, query: Query<&Cell>) {
    let mut max_strength = 0.0;
    let mut max_age = 0;
    //track start time of update
    //let start = Instant::now();
    //println!("Updating");
    query.iter().for_each(|cell| {
        if game_data.send_boats && cell.empire != -1 && cell.boat_strength > 0.0 {
            let spawn_location = cell.boat_target;
            
            //due to hexagonal grid, it should be one of 6 directions (0 - 5 inclusive)
            //direction should be based on the position of the boat relative to the cell that spawned it.
            let mut dx = spawn_location.0 as i32 - cell.position.0 as i32;
            let dy = spawn_location.1 as i32 - cell.position.1 as i32;
            if dy != 0 && cell.position.1 % 2 == 1 {//make odd rows and even rows functionally equal.
                dx -= 1;
            }

                /*EVEN
                (-1, -1) => 0,
                (0, -1) => 1,
                (1, 0) => 3,
                (0, 1) => 5,
                (-1, 1) => 4,
                (-1, 0) => 2,
                _ => 0, */
            let direction = match (dx, dy) {
                (-1, -1) => 0,
                (0, -1) => 1,
                (1, 0) => 2,
                (0, 1) => 3,
                (-1, 1) => 4,
                (-1, 0) => 5,
                _ => 0,
            };
            let boat = Boat::new(
                direction,
                cell.boat_strength,
                cell.empire,
                cell_map.1[cell.empire as usize].3,
            );
            commands.spawn(SpriteBundle {
                sprite: Sprite {
                    color: Color::hsla(cell_map.1[cell.empire as usize].0, cell_map.1[cell.empire as usize].1, 0.5, 1.0),
                    custom_size: Some(Vec2::new(1.0, 1.0)),
                    ..Default::default()
                },
                transform: Transform::from_xyz(spawn_location.0 as f32, spawn_location.1 as f32, 1.0),
                ..Default::default()
            }).insert(boat);
        }
        cell_map.0.insert(cell.position, cell.get());
        if cell.strength > max_strength {
            max_strength = cell.strength;
        }
        if cell.age > max_age {
            max_age = cell.age;
        }
    });
    game_data.max_age = max_age;
    game_data.max_strength = max_strength;
    game_data.send_boats = !game_data.send_boats;

    //print time duration of update
    //println!("Update took {:?}", start.elapsed());
}

fn update_boats_system(mut commands: Commands, mut query: Query<(Entity, &mut Boat, &mut Transform)>, mut grid: ResMut<MapData>) {
    query.iter_mut().for_each(|(entity, mut boat, mut transform)| {
        let mut position:(i32, i32) = boat.move_boat((transform.translation.x as i32, transform.translation.y as i32));
        if position.1 >= HEIGHT as i32 || position.1 < 0 {
            //println!("Flipping direction! {}", position.1);
            match boat.direction {
                0 => { boat.direction = 4; }
                1 => { boat.direction = 3; }
                2 => { boat.direction = 2; }
                3 => { boat.direction = 1; }
                4 => { boat.direction = 0; }
                5 => { boat.direction = 5; }
                _ => {}
            }
            while position.1 >= HEIGHT as i32 || position.1 < 0 {
                position = boat.move_boat((transform.translation.x as i32, transform.translation.y as i32));
                //println!("New y: {}", position.1);
            }
        }
        //check if we've hit land
        if let Some(cell) = grid.0.get_mut(&(position.0 as usize, position.1 as usize)) {
            //add boat empire and strength to the vec at the end of the cell data
            cell.8.insert((position.0 as usize, position.1 as usize), (boat.empire, boat.strength * (boat.tech_bonus + 1.0)));
            //println!("Boat has arrived at ({}, {})", position.0, position.1);
            //remove the boat
            commands.entity(entity).despawn();
            //let count = grid.0.get(&(position.0 as usize, position.1 as usize)).unwrap().8.len();
            //println!("This cell has recieved {} boats", count);
        } else {
            let x_offset = if position.1 % 2 == 0 { 0.0 } else { 0.5 };
            transform.translation = Vec3::new(position.0 as f32 + x_offset, position.1 as f32, 0.0);
        }
        if transform.translation.x < 0.0 {//loop around the world
            transform.translation.x = transform.translation.x + WIDTH as f32 - 1.0;
        } else if transform.translation.x >= WIDTH as f32 {
            transform.translation.x = transform.translation.x - WIDTH as f32;
        }
    });
}

fn pull_system(mut query: Query<&mut Cell>, cell_map: Res<MapData>) {
    //println!("Pulling");

    //track start time of pull
    //let start = Instant::now();

    query.par_iter_mut().for_each(|mut cell| {//iterate through all cells on many threads
        let position = cell.position;//get cell's position
        let mut data = Vec::new();//initialize data to be sent to cell.push
        let boat_data = cell_map.0.get(&(position.0, position.1)).unwrap().8.clone();
        let mut boat_attacks = 0.0;
        for i in 0..6 {//iterate through the 6 possible neighbor positions
            let (mut neighbor_x, mut neighbor_y): (i32, i32) = (position.0 as i32, position.1 as i32);
            if position.1 % 2 == 0 { // even row
                match i {
                    0 => { neighbor_x -= 1; neighbor_y -= 1; } // (x-1, y-1)
                    1 => { neighbor_y -= 1; }                  // (x, y-1)
                    2 => { neighbor_x -= 1; }                  // (x-1, y)
                    3 => { neighbor_x += 1; }                  // (x+1, y)
                    4 => { neighbor_x -= 1; neighbor_y += 1; } // (x-1, y+1)
                    5 => { neighbor_y += 1; }                  // (x, y+1)
                    _ => {}
                }
            } else { // odd row
                match i {
                    0 => { neighbor_y -= 1; }                  // (x, y-1)
                    1 => { neighbor_x += 1; neighbor_y -= 1; } // (x+1, y-1)
                    2 => { neighbor_x -= 1; }                  // (x-1, y)
                    3 => { neighbor_x += 1; }                  // (x+1, y)
                    4 => { neighbor_y += 1; }                  // (x, y+1)
                    5 => { neighbor_x += 1; neighbor_y += 1; } // (x+1, y+1)
                    _ => {}
                }
            }
            if neighbor_x < 0 {
                neighbor_x += WIDTH as i32;
            }
            if neighbor_x >= WIDTH as i32 {
                neighbor_x -= WIDTH as i32;
            }
            if (neighbor_y as usize) >= HEIGHT || neighbor_y < 0 {
                continue;
            }
            if cell.position == (neighbor_x as usize, neighbor_y as usize) {
                continue;
            }
            if let Some(neighbor) = cell_map.0.get(&(neighbor_x as usize, neighbor_y as usize)) {
                data.push((neighbor.0, neighbor.1, neighbor.2, neighbor.3, neighbor.4, neighbor.5, neighbor.6));
            }
        }
        for (position, boat) in boat_data.iter() {
            data.push((*position, boat.0, boat.1, 0.0, *position, boat.1, boat.0));
            if boat.0 != cell.empire {
                boat_attacks += boat.1;
            }
            //println!("Added boat to data for cell at ({}, {})", position.0, position.1);
        }
        if boat_data.len() > 0 {
            //println!("This cell has recieved {} boats", boat_data.len());
        }
        let mut tech = 0.0;
        if cell.empire != -1 {
            tech = cell_map.1[cell.empire as usize].3;
        }
        cell.pull(data, tech, boat_attacks);
    });

    //print time duration of pull
    //println!("Pull took {:?}", start.elapsed());
}

fn update_empires(mut cell_map: ResMut<MapData>, query: Query<&Cell>) {
    // Use a thread-safe Mutex to collect tech updates
    let tech_updates = Mutex::new(Vec::new());

    // Iterate through all cells in parallel
    query.par_iter().for_each(|cell| {
        // Check if the cell belongs to an empire and if the random chance for tech growth is met
        if cell.empire != -1 && rand::thread_rng().gen_range(0..100) < 1 {
            // Calculate the probability of tech growth based on cell properties
            let mut tech_probability = (1.0 - (cell.age as f32 / 10000.0).min(1.0)) * TECH_GAIN;

            tech_probability = tech_probability.clamp(0.0, 1.0); // Ensure it's between 0 and 1

            // Roll for tech growth
            if tech_probability.is_finite() && rand::thread_rng().gen_bool(tech_probability as f64) {
                // Collect the empire and tech gain in the Mutex
                let mut updates = tech_updates.lock().unwrap();
                updates.push((cell.empire as usize, TECH_GAIN));
            }
        }
    });

    // Apply the collected updates to the cell_map
    for (empire_index, tech_gain) in tech_updates.into_inner().unwrap() {
        // Reduce the tech gain as the empire's tech level increases
        let current_tech = cell_map.1[empire_index].3;
        let adjusted_tech_gain = tech_gain * (MAX_TECH - current_tech).clamp(0.0, 1.0);

        // Apply the adjusted tech gain

        cell_map.1[empire_index].3 = (adjusted_tech_gain + current_tech).min(MAX_TECH);
        let percent = cell_map.1[empire_index].3 / MAX_TECH * 100.0;
        println!("Empire {}\t gained tech: {:.9},\t now at {:.9}, \t {:.2}%", empire_index, adjusted_tech_gain, cell_map.1[empire_index].3, percent);
    }

    for empire in &mut cell_map.1 {
        empire.3 = (empire.3 - TECH_DECAY).max(0.0); // Apply decay to tech level. 
        // Since amount is fixed and applied equally to all empires, this especially hurts stagnant empires.
    }
}

//function to take the camera and use keyboard input to move or zoom it.
fn update_camera_system(mut query: Query<&mut Transform, With<Camera2d>>, keyboard_input: Res<ButtonInput<KeyCode>>) {
    let mut camera_transform = query.iter_mut().next().unwrap();
    let mut translation = camera_transform.translation;
    let mut scale = camera_transform.scale.x;
    let mut move_speed = 1.0;
    let mut zoom_speed = 0.2;
    if keyboard_input.pressed(KeyCode::ShiftLeft) {
        move_speed = 2.0;
        zoom_speed = 0.4;
    }
    if keyboard_input.pressed(KeyCode::KeyW) {
        translation.y += move_speed;
    }
    if keyboard_input.pressed(KeyCode::KeyS) {
        translation.y -= move_speed;
    }
    if keyboard_input.pressed(KeyCode::KeyA) {
        translation.x -= move_speed;
    }
    if keyboard_input.pressed(KeyCode::KeyD) {
        translation.x += move_speed;
    }
    if keyboard_input.pressed(KeyCode::KeyQ) {
        scale -= zoom_speed;
    }
    if keyboard_input.pressed(KeyCode::KeyE) {
        scale += zoom_speed;
    }
    camera_transform.translation = translation;
    camera_transform.scale = Vec3::new(scale, scale, 1.0);
}

#[derive(Resource)]
enum RenderMode {
    StrengthView,
    EmpireView,
    TerrainView,
    NeedView,
    SendView,
    AgeView,
    BoatNeedView,
    TechView,
    // Add more render modes here
}

fn update_render_mode_system(keyboard_input: Res<ButtonInput<KeyCode>>, mut render_mode: ResMut<RenderMode>) {
    if keyboard_input.just_pressed(KeyCode::Digit1) {
        *render_mode = RenderMode::EmpireView;
    } else if keyboard_input.just_pressed(KeyCode::Digit2) {
        *render_mode = RenderMode::StrengthView;
    } else if keyboard_input.just_pressed(KeyCode::Digit3) {
        *render_mode = RenderMode::NeedView;
    } else if keyboard_input.just_pressed(KeyCode::Digit4) {
        *render_mode = RenderMode::TerrainView;
    } else if keyboard_input.just_pressed(KeyCode::Digit5) {
        *render_mode = RenderMode::SendView;
    } else if keyboard_input.just_pressed(KeyCode::Digit6) {
        *render_mode = RenderMode::AgeView;
    } else if keyboard_input.just_pressed(KeyCode::Digit7) {
        *render_mode = RenderMode::BoatNeedView;
    } else if keyboard_input.just_pressed(KeyCode::Digit8) {
        *render_mode = RenderMode::TechView;
    }
}

fn update_colors(
    grid: Res<Grid>,
    cell_map: Res<MapData>,
    render_mode: Res<RenderMode>,
    game_data: Res<GameData>,
    mut query: Query<(&Transform, &mut Sprite, Option<&CellMarker>)>,
) {
    // Collect query results into a vector
    //let start = Instant::now();
    let mut query_results: Vec<(&Transform, Mut<Sprite>, Option<&CellMarker>)> = query.iter_mut().collect();
    let max_strength: f32 = game_data.max_strength;

    // Use Rayon to iterate over the vector in parallel
    query_results.par_iter_mut().for_each(|(transform,ref mut sprite, cell_marker)| {
        let x = transform.translation.x as usize;
        let y = transform.translation.y as usize;
        if !(x >= WIDTH || y >= HEIGHT) && cell_marker.is_some() {
            //check if cell_marker is Some, which means it's not a boat.

            let terrain = &grid.data[x][y];
            let max_age = game_data.max_age as f32;

            //some grid spots don't have cells because they are ocean
            //check if a cell exists at this position before trying to access it
            //let cell = cell_map.0.get(&(x, y)).unwrap_or(&((0, 0), -1, 0.0, 0.0, (0, 0), 0.0, -1));

            let cell = match cell_map.0.get(&(x, y)) {
                Some(cell) => cell,
                None => &((0, 0), -1, 0.0, 0.0, (0, 0), 0.0, -1, 0, HashMap::new(), 0.0),
            };
            let color = if matches!(*render_mode, RenderMode::TerrainView) || cell.1 == -1 {
                if terrain[0] < OCEAN_CUTOFF {
                    //ocean
                    let brightness = terrain[0] / 1.5;//cell[0] + 0.01 / (cell[0].sqrt());
                    Color::hsla(240.0, 1.0, brightness, 1.0)
                } else {
                    //land
                    let brightness = terrain[0] / 1.6;
                    Color::hsla(110.0 + (terrain[0]) * 30.0 * (1.0 / OCEAN_CUTOFF), 1.0 - (terrain[0]-OCEAN_CUTOFF) * 2.5, brightness, 1.0)
                }
            } else {
                //println!("Empire {} has strength {} and need {} at ({}, {})", cell.1, cell.2, cell.3, x, y);
                let e_hue = cell_map.1[cell.1 as usize].0;
                let e_sat = cell_map.1[cell.1 as usize].1;
                let e_tech = cell_map.1[cell.1 as usize].3;
                match *render_mode {
                    RenderMode::StrengthView => {
                        let brightness = ((cell.2 as f32).ln() / max_strength.ln()).max(0.0);
                        Color::hsla(e_hue, e_sat, brightness, 1.0)
                    }
                    RenderMode::EmpireView => {
                        Color::hsla(e_hue, e_sat, terrain[0] * 0.8, 1.0)
                    }
                    RenderMode::NeedView => {
                        let mut brightness = cell.3.sqrt() / 32.0;
                        if brightness < 0.0 {
                            brightness = 100.0;
                        }
                        Color::hsla(e_hue, e_sat, brightness, 1.0)
                    }
                    RenderMode::SendView => {
                        //cell.4 is the send_target's coordinates. cell.0 is the cell's coordinates
                        //every other row of cells is offset by 0.5, so we need to account for that
                        //goal is to color the cell based on the direction from the cell to the send_target
                        //since the cells are arranged hexagonally, each direction lines up with either a primary or secondary color.

                        let (x, y) = cell.0;
                        let (tx, ty) = cell.4;
                        let mut dx = tx as i32 - x as i32;
                        let dy = ty as i32 - y as i32;
                        //account for every other row being offset by 0.5, so each cell has 2 options above and below.
                        //this means one of those 2 options will have the same x coordinate.
                        /* how it's done elsewhere in the code
                        EVEN ROW
                        0 = x-1, y-1
                        1 = x, y-1
                        2 = x+1, y
                        3 = x, y+1
                        4 = x-1, y+1
                        5 = x-1, y
                        
                        ODD ROW
                        0 = x, y-1
                        1 = x+1, y-1
                        2 = x+1, y
                        3 = x+1, y+1
                        4 = x, y+1
                        5 = x-1, y
                        */
                        let angle;
                        if dy == 0 {
                            if dx > 0 {
                                angle = 0.0;
                            } else {
                                angle = 180.0;
                            }
                        } else {
                            if y % 2 == 1 {
                                dx -= 1;
                            }
                            //from here on odd and even rows are treated the same
                            if dy > 0 {
                                if dx > 0 {
                                    angle = 60.0;
                                } else {
                                    angle = 120.0;
                                }
                            } else {
                                if dx > 0 {
                                    angle = 300.0;
                                } else {
                                    angle = 240.0;
                                }
                            }
                        }
                        let brightness = (cell.5 as f32 / max_strength.sqrt()) + 0.1;
                        Color::hsla(angle, 1.0, brightness, 1.0)
                    }
                    RenderMode::AgeView => {
                        let brightness = ((cell.7 as f32 / max_age) * 0.5).min(0.5);
                        Color::hsla(e_hue, e_sat, brightness, 1.0)
                    }
                    RenderMode::BoatNeedView => {
                        let brightness = cell.9 as f32 / 48.0;
                        Color::hsla(e_hue, e_sat, brightness, 1.0)
                    }
                    RenderMode::TechView => {
                        Color::hsla(e_hue, e_sat / 10.0, e_tech / MAX_TECH, 1.0)
                    }
                    _ => Color::WHITE,
                }
            };

            sprite.color = color;
        }
    });
    //println!("Render took {:?}", start.elapsed());
}

#[derive(Resource)]
struct LastDraw {
    time: Instant,
}

impl Default for LastDraw {
    fn default() -> Self {
        LastDraw {
            time: Instant::now(),
        }
    }
}

fn draw_fps(
    mut last_draw: ResMut<LastDraw>,
    mut query: Query<(&mut Text, &mut Transform)>,
) {
    let now = Instant::now();
    let duration = now.duration_since(last_draw.time);
    let fps = 1.0 / duration.as_secs_f32();

    // Update the last_draw time
    last_draw.time = now;

    // Update the FPS text
    for (mut text, mut transform) in query.iter_mut() {
        text.sections[0].value = format!("FPS: {:.2}", fps);
        transform.translation = Vec3::new(0.0, 0.0, 0.0); // Adjust the position as needed
    }
}