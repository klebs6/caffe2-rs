crate::ix!();

#[test] fn utility_op_gpu_test_reshape_with_scalar() {

    #[inline] pub fn add_const_input(
        shape: &Vec<i64>,
        value: f32,
        name:  &String,
        ws:    *mut Workspace)  
    {
        todo!();
        /*
            DeviceOption option;
          CPUContext context(option);
          Blob* blob = ws->CreateBlob(name);
          auto* tensor = BlobGetMutableTensor(blob, CPU);
          tensor->Resize(shape);
          math::Set<float, CPUContext>(
              tensor->numel(), value, tensor->template mutable_data<float>(), &context);
          return;
        */
    }
    todo!();
    /*
      if (!HasCudaGPU())
        return;
      Workspace ws;
      OperatorDef def;
      def.set_name("test_reshape");
      def.set_type("Reshape");
      def.add_input("X");
      def.add_output("XNew");
      def.add_output("OldShape");
      def.add_arg()->CopyFrom(MakeArgument("shape", vector<int64_t>{1}));
      def.mutable_device_option()->set_device_type(PROTO_CUDA);
      AddConstInput(vector<int64_t>(), 3.14, "X", &ws);
      // execute the op
      unique_ptr<OperatorStorage> op(CreateOperator(def, &ws));
      EXPECT_TRUE(op->Run());
      Blob* XNew = ws.GetBlob("XNew");
      const Tensor& XNewTensor = XNew->Get<Tensor>();
      EXPECT_EQ(1, XNewTensor.dim());
      EXPECT_EQ(1, XNewTensor.numel());
  */
}

#[test] fn utility_op_test_reshape_with_scalar() {

    #[inline] pub fn add_const_input(
        shape: &Vec<i64>,
        value: f32,
        name:  &String,
        ws:    *mut Workspace)  
    {
        todo!();
        /*
            DeviceOption option;
          option.set_device_type(PROTO_CUDA);
          CUDAContext context(option);
          Blob* blob = ws->CreateBlob(name);
          auto* tensor = BlobGetMutableTensor(blob, CUDA);
          tensor->Resize(shape);
          math::Set<float, CUDAContext>(
              tensor->numel(), value, tensor->template mutable_data<float>(), &context);
          return;
        */
    }

    todo!();

    /*
      Workspace ws;
      OperatorDef def;
      def.set_name("test_reshape");
      def.set_type("Reshape");
      def.add_input("X");
      def.add_output("XNew");
      def.add_output("OldShape");
      def.add_arg()->CopyFrom(MakeArgument("shape", vector<int64_t>{1}));
      AddConstInput(vector<int64_t>(), 3.14, "X", &ws);
      // execute the op
      unique_ptr<OperatorStorage> op(CreateOperator(def, &ws));
      EXPECT_TRUE(op->Run());
      Blob* XNew = ws.GetBlob("XNew");
      const TensorCPU& XNewTensor = XNew->Get<Tensor>();
      EXPECT_EQ(1, XNewTensor.dim());
      EXPECT_EQ(1, XNewTensor.numel());
  */
}

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

/// T is ArrayOfcudnnConvolutionAlgoPerf_t
///
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
pub enum ForceAlgoType { 
    ALGO_FWD   = 0, 
    ALGO_WGRAD = 1, 
    ALGO_DGRAD = 2 
}
