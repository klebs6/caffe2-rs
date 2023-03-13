crate::ix!();

type RunnerType = fn() -> ();
type RunnerMap  = HashMap<(TensorProto_DataType, TensorProto_DataType), RunnerType>;

/**
  | @brief
  | 
  | Update slices of the tensor in-place
  | by overriding.
  | 
  | Input:
  | 
  | DATA - tensor to be updated
  | 
  | INDICES - 1-D list of indices on the first
  | dimension of X_0 that need to be updated
  | 
  | SLICES - update slices, has to have shape
  | of len(INDICES) + shape(X_0)[1:]
  | 
  | Output:
  | 
  | DATA - has to be exactly the same tensor
  | as the input 0
  | 
  | -----------
  | @note
  | 
  | The op pretty much ignores the exact
  | shapes of the input arguments and cares
  | only about sizes. It's done for performance
  | consideration to avoid unnecessary
  | reshapes. Only first dimension of X_0
  | is important, let's call it
  | 
  | N. If M is the total size of X_0 and K is
  | the size of INDICES then X_i is assumed
  | to be of shape K x (M / N) regardless of
  | the real shape.
  | ----------
  | @note
  | 
  | Each update in INDICES is applied independently
  | which means that if duplicated elements
  | are present in INDICES arbitrary one
  | will win.
  | 
  | For now really works only on CPU because
  | of INDICES access
  | 
  | Update slices of the tensor in-place
  | by overriding current value.
  | ----------
  | @note
  | 
  | The op pretty much ignores the exact
  | shapes of the input arguments and cares
  | only about sizes. It's done for performance
  | consideration to avoid unnecessary
  | reshapes. Only first dimension of X_0
  | is important, let's call it
  | 
  | N. If M is the total size of X_0 and K is
  | the size of INDICES then X_i is assumed
  | to be of shape K x (M / N) regardless of
  | the real shape.
  | ----------
  | @note
  | 
  | Each update in INDICES is applied independently
  | which means that if duplicated elements
  | are present in INDICES arbitrary one
  | will win.
  | 
  | Currently only works on CPU because
  | of access to INDICES.
  |
  */
  #[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ScatterAssignOp<Context> {
    
    storage: OperatorStorage,
    context: Context,
    runners: RunnerMap,
}

num_inputs!{ScatterAssign, 3}

num_outputs!{ScatterAssign, 1}

inputs!{ScatterAssign, 
    0 => ("DATA",    "Tensor to be updated."),
    1 => ("INDICES", "1-D list of indices on the first dimension of X_0 that need to be updated"),
    2 => ("SLICES",  "Update slices, with shape len(INDICES) + shape(X_0)[1:]")
}

outputs!{ScatterAssign, 
    0 => ("DATA", "Has to be exactly the same tensor as the input 0")
}

tensor_inference_function!{ScatterAssign, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          vector<TensorShape> out(1);
          out[0] = in[0];
          return out;
        */
    }
}

enforce_inplace!{ScatterAssign, vec![(0, 0)]}

input_tags!{
    ScatterAssignOp {
        Data,
        Indices,
        Slices
    }
}

should_not_do_gradient!{ScatterAssign}

impl<Context> ScatterAssignOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            runners_({{{TensorProto_DataType_INT32, TensorProto_DataType_FLOAT},
                       &ScatterAssignOp::DoRun<int32_t, float>},
                      {{TensorProto_DataType_INT32, TensorProto_DataType_FLOAT16},
                       &ScatterAssignOp::DoRun<int32_t, at::Half>},
                      {{TensorProto_DataType_INT32, TensorProto_DataType_UINT8},
                       &ScatterAssignOp::DoRun<int32_t, uint8_t>},
                      {{TensorProto_DataType_INT32, TensorProto_DataType_INT32},
                       &ScatterAssignOp::DoRun<int32_t, int32_t>},
                      {{TensorProto_DataType_INT32, TensorProto_DataType_INT64},
                       &ScatterAssignOp::DoRun<int32_t, int64_t>},
                      {{TensorProto_DataType_INT32, TensorProto_DataType_DOUBLE},
                       &ScatterAssignOp::DoRun<int32_t, double>},
                      {{TensorProto_DataType_INT64, TensorProto_DataType_FLOAT},
                       &ScatterAssignOp::DoRun<int64_t, float>},
                      {{TensorProto_DataType_INT64, TensorProto_DataType_FLOAT16},
                       &ScatterAssignOp::DoRun<int64_t, at::Half>},
                      {{TensorProto_DataType_INT64, TensorProto_DataType_UINT8},
                       &ScatterAssignOp::DoRun<int64_t, uint8_t>},
                      {{TensorProto_DataType_INT64, TensorProto_DataType_INT32},
                       &ScatterAssignOp::DoRun<int64_t, int32_t>},
                      {{TensorProto_DataType_INT64, TensorProto_DataType_INT64},
                       &ScatterAssignOp::DoRun<int64_t, int64_t>},
                      {{TensorProto_DataType_INT64, TensorProto_DataType_DOUBLE},
                       &ScatterAssignOp::DoRun<int64_t, double>}})
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& data = Input(DATA);
        const auto& slices = Input(SLICES);
        auto& indices = Input(INDICES);

        const auto dataType = TypeMetaToDataType(data.dtype());
        const auto slicesType = TypeMetaToDataType(slices.dtype());
        const auto indicesType = TypeMetaToDataType(indices.dtype());
        auto* output = Output(0);

        auto runner = GetRunner(dataType, slicesType, indicesType);
        (this->*runner)();
        return true;
        */
    }
    
    #[inline] pub fn get_runner(
        &mut self, 
        data_type:    TensorProto_DataType,
        slices_type:  TensorProto_DataType,
        indices_type: TensorProto_DataType) -> RunnerType 
    {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(dataType, slicesType, "Data and slice types must match");
        auto it = runners_.find({indicesType, dataType});
        CAFFE_ENFORCE(
            it != runners_.end(),
            "Could not find the runner corresponding to indicesType, dataType = ",
            indicesType,
            " ",
            dataType);
        return it->second;
        */
    }
    
    #[inline] pub fn do_run<Index, T>(&mut self) {
        todo!();
        /*
            auto& input = Input(DATA);
        auto& indices = Input(INDICES);
        auto& slices = Input(SLICES);
        auto* output = Output(0);
        CAFFE_ENFORCE_EQ(&input, output, "In place operation is required");

        CAFFE_ENFORCE_GT(input.dim(), 0, "X0 has to be at least the vector");
        int64_t M = input.numel();
        int64_t N = input.size(0);
        int64_t K = indices.numel();
        int64_t block_size = M / N;
        CAFFE_ENFORCE_EQ(slices.numel(), block_size * K);
        // TODO(dzhulgakov): it can be made to work with arbitrary data type by
        // using raw_mutable_data
        T* data = output->template mutable_data<T>();
        const Index* idxs = indices.template data<Index>();
        const T* slicesData = slices.template data<T>();
        DoScatterAssign(data, idxs, slicesData, N, K, block_size);
        */
    }
    
    #[inline] pub fn do_scatter_assign<Index, T>(
        &mut self, 
        data:          *mut T,
        idxs:          *const Index,
        slices_data:   *const T,
        n:             i64,
        k:             i64,
        block_size:    i64) 
    {
        todo!();
        /*
            for (int i = 0; i < K; ++i) {
          Index idx = idxs[i];
          // double-checking the indices, but it's fine as it's DCHECK only
          DCHECK(0 <= idx && idx < N)
              << "Index out of bounds: " << idx << ", range 0 to " << N;
          context_.template CopySameDevice<T>(
              block_size, slicesData + block_size * i, data + block_size * idx);
        }
        */
    }
}

