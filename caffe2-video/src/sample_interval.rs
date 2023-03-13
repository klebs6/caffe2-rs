crate::ix!();

/**
  | sampling interval for fps starting
  | at specified timestamp use enum SpecialFps
  | to set special fps decoding behavior
  | note sampled fps will not always accurately
  | follow the target fps, because sampled
  | frame has to snap to actual frame timestamp,
  | e.g. video fps = 25, sample fps = 4 will
  | sample every 0.28s, not 0.25 video fps
  | = 25, sample fps = 5 will sample every
  | 0.24s, not 0.2, because of floating-point
  | division accuracy (1 / 5.0 is not exactly
  | 0.2)
  |
  */
pub struct SampleInterval {
    timestamp: f64,
    fps:       f64,
}

impl Default for SampleInterval {

    fn default() -> Self {
        todo!();
        /*
           : timestamp(-1), fps(SpecialFps::SAMPLE_ALL_FRAMES
           */
    }
}

impl SampleInterval {
    
    pub fn new(ts: f64, f: f64) -> Self {
        todo!();
        /*
            : timestamp(ts), fps(f)
        */
    }
}

impl PartialEq for SampleInterval {

    fn eq(&self, other: &SampleInterval) -> bool {
        todo!();
    }
}

impl Eq for SampleInterval {}

impl Ord for SampleInterval {
    
    fn cmp(&self, other: &SampleInterval) -> Ordering {
        todo!();
        /*
            return (timestamp < itvl.timestamp);
        */
    }
}

impl PartialOrd for SampleInterval {
    fn partial_cmp(&self, other: &SampleInterval) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
