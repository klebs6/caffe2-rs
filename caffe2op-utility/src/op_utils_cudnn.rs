crate::ix!();

/**
  | Earlier in the days Caffe sets the default
  | cudnn workspace to 8MB.
  | 
  | We bump it up to 64MB in Caffe2, as this
  | enables the use of Winograd in many cases,
  | something very beneficial to more recent
  | CNN models.
  |
  */
pub const kCONV_CUDNN_WORKSPACE_LIMIT_BYTES: usize = 64 * 1024 * 1024;

/**
  | Manually specified number of algorithms
  | implemented in Cudnn.
  | 
  | This does not have any performance implications,
  | as we will always find the fastest algorithm;
  | setting them to the right number of algorithms
  | will enable us to best report the statistics
  | when doing an exhaustive search, though.
  |
  */
macro_rules! manually_specified_number_of_algorithms {
    () => {
        /// Note: Double each of these due to potential
        /// tensorcore + non-tensorcore versions
        /// which are treated as separate returned algos
        pub const kNUM_CUDNN_FWD_ALGS:        usize = 2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
        pub const kNUM_CUDNN_BWD_FILTER_ALGS: usize = 2 * CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
        pub const kNUM_CUDNN_BWD_DATA_ALGS:   usize = 2 * CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
    };
    (old_cuda) => {
        pub const kNUM_CUDNN_FWD_ALGS:        usize = 7;
        pub const kNUM_CUDNN_BWD_FILTER_ALGS: usize = 4;
        pub const kNUM_CUDNN_BWD_DATA_ALGS:   usize = 5;
    }
}

#[cfg(cudnn_version_min = "7.0.0")]
manually_specified_number_of_algorithms![];

#[cfg(not(cudnn_version_min = "7.0.0"))]
manually_specified_number_of_algorithms![old_cuda];

///T is ArrayOfcudnnConvolutionAlgoPerf_t
#[inline] pub fn log_cu_dnnperf_stats<T>(
    perf_stat: &T,
    returned_algo_count: i32) 
{
    todo!();
    /*
        VLOG(1) << "Perf result: (algo: stat, time, memory)";
        for (int i = 0; i < returned_algo_count; ++i) {
            const auto& stat = perf_stat[i];
            VLOG(1) << stat.algo << ": " << stat.status << " " << stat.time << " "
                << stat.memory;
        }
    */
}

/**
  | Easier indexing into force_algo_ vector,
  | shared by CudnnConvTransposeOpBase
  | and CudnnConvOpBase to force usage
  | of a particular algorithm instead of
  | searching
  |
  */
pub enum ForceAlgoType { ALGO_FWD = 0, ALGO_WGRAD = 1, ALGO_DGRAD = 2 }
