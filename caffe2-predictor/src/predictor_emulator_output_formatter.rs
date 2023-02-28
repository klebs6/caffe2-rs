crate::ix!();

/**
  | An interface that formats the output
  | of the emulator runs.
  |
  */
pub trait OutputFormatter {

    fn format(
        &self,
        durations_ms: &Vec<f32>,
        threads:      u64,
        iterations:   u64) -> String;
}
