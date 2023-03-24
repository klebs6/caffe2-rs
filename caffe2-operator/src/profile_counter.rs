crate::ix!();

#[derive(Default)]
pub struct ProfileCounter {
    timer:      Timer,
    start_time: f32, // default = 0.0
    run_time:   f32, // default = 0.0
}
