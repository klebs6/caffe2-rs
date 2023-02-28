crate::ix!();

/**
 | C2 wrapper for fp16 gemm
 |
 | Suppose your predict_net has an FC operator in
 | fp32 as follows:
 |
 | op {
 |   input: "x"
 |   input: "w"
 |   input: "b"
 |   output: "y"
 |   type: "FC"
 | }
 | ...
 | external_input: "w"
 |
 | To use FbFCPacked operator with fp16 fbgemm, in init_net
 | ... # an operator that generates w
 | op {
 |   input: "w"
 |   output: "w_packed"
 |   type: "FbGemmPack"
 | }
 | ...
 | external_output: "w_packed"
 |
 | in predict_net:
 | op {
 |   input: "x"
 |   input: "w_packed"
 |   input: "b"
 |   output: "y"
 |   type: "FbFCPacked"
 | }
 | ...
 | external_input: "w_packed"
 |
 |  class Engine = DefaultEngine,
 |  typename T_W = fbgemm::float16>
 */
pub struct FbFCPackedOperator<Context,Engine,T_W> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,

    axis:             usize, // default = 1
    axis_w:           usize, // default = 1

    /**
      | A local vector to cache the output shape
      | so we don't need to recreate a vector
      | object every time we run Run().
      |
      */
    y_shape_cache:    Vec<i64>,

    /// {Context::GetDeviceType()};
    bias_multiplier:  Tensor,

    #[cfg(use_fbgemm)]
    packed_w:         Box<fbgemm::PackedGemmMatrixFP16>, // default = nullptr

    phantomA: PhantomData<T_W>,
    phantomB: PhantomData<Engine>,
}

impl<Context,Engine,T_W> FbFCPackedOperator<Context,Engine,T_W> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            axis_(this->template GetSingleArgument<int32_t>("axis", 1)),
            axis_w_(this->template GetSingleArgument<int32_t>("axis_w", 1))
        */
    }

    /// template on X, B, and Y.
    #[inline] pub fn do_run_with_type<T_X, T_B, T_Y>(&mut self) -> bool {
    
        todo!();
        /*
            const auto& X = Input(0);
        const auto& b = Input(2);

        CAFFE_ENFORCE(b.dim() == 1, b.dim());
        // batch size
        const auto canonical_axis = X.canonical_axis_index(axis_);
        const int M = X.size_to_dim(canonical_axis);
        const int N = b.numel();

        // Load the packed matrix
        auto* W =
            OperatorStorage::Input<caffe2::unique_ptr<fbgemm::PackedGemmMatrixFP16>>(1)
                .get();
        const int K = W->numRows();
        if (!W->packed()) {
          if (!packed_w_) {
            std::vector<float> src_mat(W->matSize());
            for (int i = 0; i < W->matSize(); ++i) {
              src_mat[i] =
                fbgemm::cpu_half2float(W->pmat()[i]);
            }
            packed_w_ = std::make_unique<fbgemm::PackedGemmMatrixFP16>(
                fbgemm::matrix_op_t::Transpose,
                W->numRows(), W->numCols(),
                1.0,
                src_mat.data());
          }
          W = packed_w_.get();
        }

        auto dimErrorString = [&]() {
          return c10::str(
              "Dimension mismatch: ",
              "X: ",
              X.sizes(),
              ", W: ",
              std::vector<int>({K, W->numCols()}),
              ", b: ",
              b.sizes(),
              ", axis: ",
              axis_,
              ", M: ",
              M,
              ", N: ",
              N,
              ", K: ",
              K);
        };
        // Error checking
        CAFFE_ENFORCE(M == X.numel() / K, dimErrorString());
        CAFFE_ENFORCE(K == X.size_from_dim(canonical_axis), dimErrorString());
        CAFFE_ENFORCE(N == W->numCols(), dimErrorString());
        Y_shape_cache_ = X.sizes().vec();
        // This is an invariant of canonical_axis, so we can DCHECK.
        DCHECK_LE(canonical_axis + 1, Y_shape_cache_.size());
        Y_shape_cache_.resize(canonical_axis + 1);
        Y_shape_cache_[canonical_axis] = N;
        auto* Y = Output(0, Y_shape_cache_, at::dtype<T_Y>());

        if (X.numel() == 0) {
          // skip the rest of the computation if X is empty
          Y->template mutable_data<T_Y>();
          return true;
        }

        // Call the fp16 gemm interface
        fbgemm::cblas_gemm_compute(
            fbgemm::matrix_op_t::NoTranspose,
            M,
            X.template data<T_X>(),
            *W,
            0.f,
            Y->template mutable_data<T_Y>());

        // Add bias term, accumulation is still in fp32.
        TensorProto::DataType math_type = TensorProto_DataType_FLOAT;
        if (bias_multiplier_.numel() != M) {
          // If the helper bias multiplier is not M, reshape and fill it with one.
          bias_multiplier_.Resize(M);
          math::Set<T_B, Context>(
              M,
              convert::To<float, T_B>(1),
              bias_multiplier_.template mutable_data<T_B>(),
              &context_);
        }
        math::Gemm<T_B, Context, Engine>(
            CblasNoTrans,
            CblasNoTrans,
            M,
            N,
            1,
            1,
            bias_multiplier_.template data<T_B>(),
            b.template data<T_B>(),
            1,
            Y->template mutable_data<T_Y>(),
            &context_,
            math_type);

        return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DoRunWithType<
            float, // X
            float, // B
            float>(); // Y
        */
    }
}

///------------------
pub struct PackedGemmMatrixFP16ShapeFunctions { }

impl PackedGemmMatrixFP16ShapeFunctions {

    pub fn new() -> Self {
    
        todo!();
        /*
            : ExternalTensorFunctionsBase()
        */
    }
}

impl ExternalTensorFunctionsBase for PackedGemmMatrixFP16ShapeFunctions {

    #[inline] fn is_quantized(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    #[inline] fn load_info_of_blob(&self, 
        unused0: *const Blob,
        unused1: *mut Vec<f32>,
        unused2: *mut Vec<f32>,
        unused3: *mut u32)  {

        todo!();
        /*
        
        */
    }
    
    #[inline] fn is_same_meta_type(&self, id: TypeIdentifier) -> bool {
        
        todo!();
        /*
            return id == TypeMeta::Id<unique_ptr<fbgemm::PackedGemmMatrixFP16>>();
        */
    }
    
    #[inline] fn get_type_meta_id(&self) -> TypeIdentifier {
        
        todo!();
        /*
            return TypeMeta::Id<unique_ptr<fbgemm::PackedGemmMatrixFP16>>();
        */
    }
    
    #[inline] fn get_external_tensor_type(&self, unused: *const c_void) -> TypeMeta {
        
        todo!();
        /*
            return TypeMeta::Make<at::Half>();
        */
    }
    
    #[inline] fn get_external_tensor_info(&mut self, 
        c:        *const c_void,
        capacity: *mut usize,
        device:   *mut DeviceOption) -> Vec<i64> {

        todo!();
        /*
            return GetFbgemmTensorInfo(c, capacity, device);
        */
    }
    
    #[inline] fn setup_external_tensor_descriptor(&self, 
        blob:    *const Blob,
        shapes:  *mut Vec<Vec<u64>>,
        all_scales:  *mut Vec<Vec<f32>>,
        all_offsets: *mut Vec<Vec<i32>>,
        desc:    *mut ExternalTensorDescriptor)  {

        todo!();
        /*
            const auto* packed =
          blob->template Get<unique_ptr<fbgemm::PackedGemmMatrixFP16>>().get();

      // setup data and type
      desc->dataType = 10; // ONNXIFI_DATATYPE_FLOAT16
      desc->buffer = reinterpret_cast<uint64_t>(packed->pmat());

      // setup dim and shape
      std::vector<uint64_t> shape{static_cast<uint64_t>(packed->numCols()),
                                  static_cast<uint64_t>(packed->numRows())};
      shapes->emplace_back(std::move(shape));
      desc->dimensions = 2;
      desc->shape = shapes->back().data();

      // no quantization params as this is not quantization
      desc->quantizationParams = 0;

      // not an offline tensor
      desc->isOffline = 0;
        */
    }
}

register_cpu_operator!{
    FbFCPacked,
    FbFCPackedOperator<CPUContext, DefaultEngine, fbgemm::float16>}

#[inline] pub fn get_fbgemm_tensor_info(
    c:        *const c_void,
    capacity: *mut usize,
    device:   *mut DeviceOption) -> Vec<i64> {
    
    todo!();
    /*
        const unique_ptr<fbgemm::PackedGemmMatrixFP16>* tc =
          static_cast<const unique_ptr<fbgemm::PackedGemmMatrixFP16>*>(c);
      device->set_device_type(PROTO_CPU);
      *capacity = (*tc)->numRows() * (*tc)->numCols() * 2;
      return {(*tc)->numCols(), (*tc)->numRows()};
    */
}

pub fn caffe2_initialize_fbgemm(x: *mut i32, c: *mut *mut *mut u8) -> bool {

    todo!();

    /*
      RegisterTensorInfoFunction(
          TypeMeta::Id<unique_ptr<fbgemm::PackedGemmMatrixFP16>>(),
          GetFbgemmTensorInfo);
      return true;
    */
}

register_caffe2_init_function!{
    InitFbgemmContext,
    Caffe2InitializeFbgemm,
    "Register the tensor info function for the packed gemm matrix used in Fbgemm"
}

/**
  | Same as FC, but the weight is prepacked
  | as a fbgemm::PackedGemmMatrixFP16
  |
  */
register_external_tensor_functions!{
    (TypeMeta::Id::<Box::<fbgemm::PackedGemmMatrixFP16>>()),
    PackedGemmMatrixFP16ShapeFunctions
}

num_inputs!{FbFCPacked, 3}

num_outputs!{FbFCPacked, 1}

cost_inference_function!{
    FbFCPacked, 
    /* 

       (OpSchema::CostInferenceFunctionType(
       std::bind(CostInferenceForFC, _1, _2, false))) */ 
}

tensor_inference_function!{FbFCPacked, /* (std::bind(FCShapeInference, _1, _2, false)) */}
