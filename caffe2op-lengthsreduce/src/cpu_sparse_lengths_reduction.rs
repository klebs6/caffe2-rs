crate::ix!();

/**
 | A templated class that implements
 | SparseLengths[Sum,WeightedSum,Mean].
 |
 | typename T, // output type
 | class InputTypes, // supported input types, such as TensorTypes<float>
 | bool USE_WEIGHT = false, // Whether it is SparseLengthsWeightedSum
 | bool USE_MEAN = false, // Whether this is SparseLengthsMean
 | bool USE_POSITIONAL_WEIGHT = false
 | // USE_WEIGHT = true and USE_POSITIONAL_WEIGHT = true
 | // -> SparseLengthsPositionalWeightedSum
 */
pub struct CPUSparseLengthsReductionOp<
    T,
    InputTypes,
    const USE_WEIGHT: bool,
    const USE_MEAN: bool,
    const USE_POSITIONAL_WEIGHT: bool = false> 
{
    //USE_OPERATOR_FUNCTIONS(CPUContext);
    storage: OperatorStorage,
    context: CPUContext,

    #[cfg(use_fbgemm)]
    last_block_size:  i64, // default = -1

    #[cfg(use_fbgemm)]
    kernel_fp32_i32:  fbgemm::EmbeddingSpMDMKernelSignature<f32, i32>::Type,

    #[cfg(use_fbgemm)]
    kernel_fp32_i64:  fbgemm::EmbeddingSpMDMKernelSignature<f32, i64>::Type,

    #[cfg(use_fbgemm)]
    kernel_fp16_i32:  fbgemm::EmbeddingSpMDMKernelSignature<f16, i32>::Type,

    #[cfg(use_fbgemm)]
    kernel_fp16_i64:  fbgemm::EmbeddingSpMDMKernelSignature<f16, i64>::Type,

    phantom: PhantomData<T>,
    phantomIT: PhantomData<InputTypes>,
}

/**
  |note there is something weird here because we
  |may or may not have a Weights input therefore
  |check how we use Input(N) where N is integer for
  |correct usage (see c++ if this is confusing)
  */
pub enum CPUSparseLengthsReductionOpTags {

    // Data input.
    Data,

    /**
      | Weight input used in
      | SparseLengthsWeightedSum
      |
      */
    MaybeWeight,

    /**
      | 1 in SparseLengths[Sum,Mean] and 2
      | in SparseLengthsWeightedSum
      |
      */
    Indices,

    /**
      | 2 in SparseLengths[Sum, Mean], 3 in
      | SparseLengthsWeightedSum
      |
      */
    Lengths,
}

impl<T,InputTypes,const USE_WEIGHT: bool,const USE_MEAN: bool,const USE_POSITIONAL_WEIGHT: bool> 
CPUSparseLengthsReductionOp<T,InputTypes,USE_WEIGHT,USE_MEAN,USE_POSITIONAL_WEIGHT> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...) 

        static_assert(
            !(USE_WEIGHT & USE_MEAN), "Cannot both specify weight and mean.");
        */
    }

    /**
      | Currently, we support float and at::Half
      | inputs for input data type, and int32_t
      | and int64_t for the index type.
      |
      */
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<InputTypes>::call(this, Input(DATA));
        */
    }
    
    #[inline] pub fn do_run_with_type<InputType>(&mut self) -> bool {
    
        todo!();
        /*
            return DispatchHelper<TensorTypes2<int32_t, int64_t>, InputType>::call(
            this, Input(INDICES));
        */
    }
    
    #[inline] pub fn do_run_with_type2<InputType, IndexType>(&mut self) -> bool {
    
        todo!();
        /*
            auto& dataInput = Input(DATA);
        auto& indicesInput = Input(INDICES);
        auto& lengthsInput = Input(LENGTHS);

        const int64_t M = lengthsInput.size(0);
        const int64_t indices_size = indicesInput.numel();

        auto shape = dataInput.sizes().vec();
        shape[0] = M;
        auto* output = Output(0, shape, at::dtype<T>());
        T* out_data = output->template mutable_data<T>();

        if (indices_size == 0) {
          if (M > 0) {
            memset(out_data, 0, output->numel() * sizeof(T));
          }
          return true;
        }

        CAFFE_ENFORCE_EQ(1, indicesInput.dim(), "INDICES must be a vector");
        CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
        const int64_t N = dataInput.size(0);
        const int D = dataInput.size_from_dim(1);

        const InputType* in_data = dataInput.template data<InputType>();
        const IndexType* indices = indicesInput.template data<IndexType>();
        const int* lengths = lengthsInput.template data<int>();
        const T* in_weight = nullptr;

        if (USE_WEIGHT) {
          // static if
          auto& weightInput = Input(WEIGHT);
          CAFFE_ENFORCE_EQ(1, weightInput.dim(), "WEIGHT must be a vector");
          if (!USE_POSITIONAL_WEIGHT) {
            CAFFE_ENFORCE_EQ(
                weightInput.numel(),
                indices_size,
                "Weight should have the same length as indices.");
          }
          in_weight = weightInput.template data<T>();
        }

    #ifdef USE_FBGEMM
        // If this is the first call or block size has changed (should never
        // happen actually), generate a kernel.
        if (D != last_block_size) {
          last_block_size = D;
          if (std::is_same<InputType, float>::value) {
            if (std::is_same<IndexType, std::int32_t>::value) {
              kernel_fp32_i32_ =
                  fbgemm::GenerateEmbeddingSpMDM<float, std::int32_t>(
                      D,
                      USE_WEIGHT,
                      USE_MEAN,
                      /*prefetch distance*/ 16,
                      USE_POSITIONAL_WEIGHT,
                      /*use_offsets*/ false);
            } else {
              CAFFE_ENFORCE((std::is_same<IndexType, std::int64_t>::value));
              kernel_fp32_i64_ =
                  fbgemm::GenerateEmbeddingSpMDM<float, std::int64_t>(
                      D,
                      USE_WEIGHT,
                      USE_MEAN,
                      /*prefetch distance*/ 16,
                      USE_POSITIONAL_WEIGHT,
                      /*use_offsets*/ false);
            }
          } else {
            CAFFE_ENFORCE((std::is_same<InputType, at::Half>::value));
            if (std::is_same<IndexType, std::int32_t>::value) {
              kernel_fp16_i32_ =
                  fbgemm::GenerateEmbeddingSpMDM<fbgemm::float16, std::int32_t>(
                      D,
                      USE_WEIGHT,
                      USE_MEAN,
                      /*prefetch distance*/ 16,
                      USE_POSITIONAL_WEIGHT,
                      /*use_offsets*/ false);
            } else {
              CAFFE_ENFORCE((std::is_same<IndexType, std::int64_t>::value));
              kernel_fp16_i64_ =
                  fbgemm::GenerateEmbeddingSpMDM<fbgemm::float16, std::int64_t>(
                      D,
                      USE_WEIGHT,
                      USE_MEAN,
                      /*prefetch distance*/ 16,
                      USE_POSITIONAL_WEIGHT,
                      /*use_offsets*/ false);
            }
          }
        }

        bool success;
        if (std::is_same<InputType, float>::value) {
          if (std::is_same<IndexType, std::int32_t>::value) {
            success = kernel_fp32_i32_(
                M,
                indices_size,
                N,
                reinterpret_cast<const float*>(in_data),
                indicesInput.template data<std::int32_t>(),
                lengths,
                in_weight,
                out_data);
          } else {
            success = kernel_fp32_i64_(
                M,
                indices_size,
                N,
                reinterpret_cast<const float*>(in_data),
                indicesInput.template data<std::int64_t>(),
                lengths,
                in_weight,
                out_data);
          }
        } else {
          if (std::is_same<IndexType, std::int32_t>::value) {
            success = kernel_fp16_i32_(
                M,
                indices_size,
                N,
                reinterpret_cast<const fbgemm::float16*>(in_data),
                indicesInput.template data<std::int32_t>(),
                lengths,
                in_weight,
                out_data);
          } else {
            success = kernel_fp16_i64_(
                M,
                indices_size,
                N,
                reinterpret_cast<const fbgemm::float16*>(in_data),
                indicesInput.template data<std::int64_t>(),
                lengths,
                in_weight,
                out_data);
          }
        }

        if (success) {
          return true;
        }

        int64_t current = 0;
        for (int m = 0; m < M; ++m) {
          for (int i = 0; i < lengths[m]; ++i) {
            CAFFE_ENFORCE_LT(
                current,
                indices_size,
                "Your input seems to be incorrect: the sum of lengths values "
                "should be the size of the indices tensor, but it appears not.");
            IndexType idx = indices[current];
            CAFFE_ENFORCE(
                0 <= idx && idx < N,
                "Index ",
                current,
                " is out of bounds: ",
                idx,
                ", range 0 to ",
                N,
                ", actual batch length is ",
                M);
            ++current;
          }
        }
        CAFFE_ENFORCE_EQ(
            current,
            indices_size,
            "Your input seems to be incorrect: the sum of lengths values should be "
            "the size of the indices tensor, but it appears not.");

        return false;
    #endif

        // delegate work to perfkernel that branches based on architecture
        EmbeddingLookup<IndexType, InputType, T, USE_POSITIONAL_WEIGHT>(
            D,
            M,
            indices_size,
            N,
            in_data,
            indices,
            lengths,
            in_weight,
            nullptr, // scale_bias field is only used in SparseLengths8BitsRowwiseOp
            USE_MEAN,
            out_data);
        return true;
        */
    }
}
