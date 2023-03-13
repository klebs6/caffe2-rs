crate::ix!();


declare_int!{warmup}
declare_int!{iter}
declare_int!{threads}
declare_int!{runs}
declare_string!{run_net}
declare_string!{init_net}
declare_string!{data_net}
declare_string!{input_dims}
declare_string!{input_types}

pub struct BenchmarkParam {
    profiler:   Box<dyn Profiler>,
    emulator:   Box<dyn Emulator>,
    formatter:  Box<dyn OutputFormatter>,
}

/**
  | benchmark runner takes an @emulator
  | to run nets.
  | 
  | The runtime will be measured by @profiler.
  | 
  | The output will be formatted by @formatter
  |
  */
pub struct BenchmarkRunner {

}

impl BenchmarkRunner {
    
    #[inline] pub fn pre_benchmark_setup(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn post_benchmark_cleanup(&mut self)  {
        
        todo!();
        /*
        
        */
    }
}


// Basic benchmark params
define_int!{warmup,                10000, "The number of iterations to warm up."}
define_int!{iter,                  10000000, "The number of iterations to run."}
define_int!{threads,               32, "The number of threads to run."}
define_int!{runs,                  10, "The number of independent runs."}

// Benchmark setup params
define_int!{num_loading_threads,   56, "The number of threads to build predictors."}

// Benchmark model params
define_string!{run_net,            "", "The given net to benchmark."}
define_string!{init_net,           "", "The given net to initialize."}
define_string!{data_net,           "", "The given net to get input data."}
define_string!{input_dims,         "", "The path of the file that stores input dimensions of all the operators in the run net. Each element of the array is a mapping from operator index to its input dimension."}
define_string!{input_types,        "", "The path of the file that stores input types of all the operators in the run net. Each element of the array is a mapping from operator index to its input types."}

impl BenchmarkRunner {
    
    #[inline] pub fn benchmark(&mut self, param: &BenchmarkParam)  {
        
        todo!();
        /*
            param.emulator->init();
      std::vector<float> durations_ms;
      for (size_t run = 0; run < FLAGS_runs; ++run) {
        LOG(WARNING) << "Starting run " << run + 1;
        LOG(INFO) << "Warming up " << FLAGS_threads << " threads with "
                  << FLAGS_warmup << " iterations...";
        param.emulator->run(FLAGS_warmup);

        LOG(INFO) << "Starting benchmark with " << FLAGS_iter << " iterations...";
        pre_benchmark_setup();
        const auto duration_ms =
            param.profiler->profile([&]() { param.emulator->run(FLAGS_iter); });

        durations_ms.emplace_back(duration_ms);
        auto throughput = FLAGS_iter / (duration_ms / MS_IN_SECOND);
        LOG(INFO) << "Benchmark run finished in " << duration_ms / MS_IN_SECOND
                  << " seconds.\n"
                  << "Throughput:\t\t" << throughput << " iterations/s\n";

        post_benchmark_cleanup();
        LOG(INFO) << "Run " << run + 1 << " finished";
      }
      LOG(WARNING) << param.formatter->format(
          durations_ms, FLAGS_threads, FLAGS_iter);
        */
    }
}
