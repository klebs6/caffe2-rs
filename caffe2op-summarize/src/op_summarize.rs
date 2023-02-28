crate::ix!();

pub const kSummaryzeOpExtension: &'static str = ".summary";

/**
  | Summarize computes four statistics
  | of the input tensor (Tensor)- min, max,
  | mean and standard deviation.
  | 
  | The output will be written to a 1-D tensor
  | of size 4 if an output tensor is provided.
  | 
  | Else, if the argument 'to_file' is greater
  | than 0, the values are written to a log
  | file in the root folder.
  |
  */
pub struct SummarizeOp<T,Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage:   OperatorStorage,
    context:   Context,

    to_file:   bool,
    log_file:  Box<std::fs::File>,

    /**
      | Input: X;
      |
      | output: if set, a summarized Tensor of
      | shape 4, with the values being min, max,
      | mean and std respectively.
      */
    phantom:   PhantomData<T>,
}

register_cpu_operator!{Summarize, SummarizeOp<float, CPUContext>}

num_inputs!{Summarize, 1}

num_outputs!{Summarize, (0,1)}

inputs!{Summarize, 
    0 => ("data", "The input data as Tensor.")
}

outputs!{Summarize, 
    0 => ("output", "1-D tensor (Tensor) of size 4 containing min, max, mean and standard deviation")
}

args!{Summarize, 
    0 => ("to_file", "(int, default 0) flag to indicate if the summarized statistics have to be written to a log file.")
}

should_not_do_gradient!{Summarize}

impl<T,Context> Drop for SummarizeOp<T,Context> {
    fn drop(&mut self) {
        todo!();
        /* 
        if (to_file_)
          log_file_->close();
       */
    }
}

impl<T,Context> SummarizeOp<T,Context> {

    const MIN_IDX:   i32 = 0;
    const MAX_IDX:   i32 = 1;
    const MEAN_IDX:  i32 = 2;
    const STD_IDX:   i32 = 3;
    const NUM_STATS: i32 = 4;

    pub fn new(def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<Context>(def, ws),
            to_file_(this->template GetSingleArgument<int>("to_file", 0)) 
        if (to_file_) {
          // We will output to file instead of printing on screen.
          const string& target_folder = ws->RootFolder();
          // We will write each individual tensor to its individual file.
          // Also, since the namescope is currently represented by "/", we will
          // need to replace it with a symbol that does not conflict with the
          // folder separator in Linux.
          string proper_name = def.input(0);
          std::replace(proper_name.begin(), proper_name.end(), '/', '#');
          log_file_.reset(new std::ofstream(
              target_folder + "/" + proper_name + kSummaryzeOpExtension,
              std::ofstream::out | std::ofstream::trunc));
          CAFFE_ENFORCE(
              log_file_->good(),
              "Failed to open summarize file for tensor ",
              def.input(0),
              ". rdstate() = ",
              log_file_->rdstate());
        }
        */
    }
}

impl SummarizeOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
      const auto N = X.numel();
      CAFFE_ENFORCE_GT(N, 0);

      const float* Xdata = X.data<float>();
      double mean = 0;
      float max = Xdata[0];
      float min = Xdata[0];
      for (auto i = 0; i < N; ++i) {
        mean += static_cast<double>(Xdata[i]) / N;
        max = std::max(max, Xdata[i]);
        min = std::min(min, Xdata[i]);
      }
      // We will simply do a two-pass. More efficient solutions can be written but
      // I'll keep code simple for now.
      double standard_deviation = 0;
      for (auto i = 0; i < N; ++i) {
        double diff = Xdata[i] - mean;
        standard_deviation += diff * diff;
      }
      // Unbiased or biased? Let's do unbiased now.
      standard_deviation = N == 1 ? 0 : std::sqrt(standard_deviation / (N - 1));
      if (to_file_) {
        (*log_file_) << min << " " << max << " " << mean << " "
                     << standard_deviation << std::endl;
      }
      if (OutputSize()) {
        auto* Y = Output(0, {NUM_STATS}, at::dtype<float>());
        float* Ydata = Y->template mutable_data<float>();
        Ydata[MIN_IDX] = min;
        Ydata[MAX_IDX] = max;
        Ydata[MEAN_IDX] = static_cast<float>(mean);
        Ydata[STD_IDX] = static_cast<float>(standard_deviation);
      }
      return true;
        */
    }
}
