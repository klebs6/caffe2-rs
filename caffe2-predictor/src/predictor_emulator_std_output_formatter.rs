crate::ix!();

use crate::OutputFormatter;

pub const MS_IN_SECOND: u64 = 1000;

/**
  | Print the output of the emulator run
  | to stdout.
  |
  */
pub struct StdOutputFormatter {
    base: dyn OutputFormatter,
}

impl OutputFormatter for StdOutputFormatter {

    #[inline] fn format(&self, 
        durations_ms: &Vec<f32>,
        threads:      u64,
        iterations:   u64) -> String {
        
        todo!();
        /*
            auto mean = get_mean(durations_ms);
        auto throughput = iterations / (mean / MS_IN_SECOND);
        return std::string("\n\n====================================\n") +
            "Predictor benchmark finished with " + c10::to_string(threads) +
            " threads.\nThroughput:\t\t" + c10::to_string(throughput) +
            " iterations/s\nVariation:\t\t" +
            c10::to_string(get_stdev(durations_ms) * 100 / mean) +
            "%\n====================================";
        */
    }
}

impl StdOutputFormatter {
    
    #[inline] pub fn get_mean<T>(values: &Vec<T>) -> f32 {
    
        todo!();
        /*
            float sum = std::accumulate(values.begin(), values.end(), 0.0);
        return sum / values.size();
        */
    }
    
    #[inline] pub fn get_stdev<T>(values: &Vec<T>) -> f32 {
    
        todo!();
        /*
            auto mean = get_mean(values);
        double sq_sum =
            std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
        return std::sqrt(sq_sum / values.size() - mean * mean);
        */
    }
}
