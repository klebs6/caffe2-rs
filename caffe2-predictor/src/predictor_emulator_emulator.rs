crate::ix!();

/**
  | A net emulator. In short, it can run nets
  | with given @iterations.
  |
  */
pub trait Emulator {

    fn init(&mut self);

    fn run(&mut self, iterations: u64);
}
