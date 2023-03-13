crate::ix!();

should_not_do_gradient!{Scatter}

/**
  | Update values of the tensor by overriding
  | current value specified by indices.
  | 
  | Writes all values from the tensor UPDATES
  | into DATA at the indices specified in
  | the INDICES tensor.
  | 
  | For each value in DATA, its output index
  | is specified by its index in UPDATES
  | and by the corresponding value in INDICES
  | for the specified axis.
  | 
  | For a 3-D tensor, DATA is updated as:
  | 
  | DATA[INDICES[i][j][k]][j][k] = UPDATES[i][j][k]
  | # if axis == 0
  | 
  | DATA[i][INDICES[i][j][k]][k] = UPDATES[i][j][k]
  | # if axis == 1
  | 
  | DATA[i][j][INDICES[i][j][k]] = UPDATES[i][j][k]
  | # if axis == 2
  | 
  | Currently only works on CPU because
  | of access to INDICES.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ScatterOp<Context> {
    storage: OperatorStorage,
    context: CPUContext,
    axis:    i32,
    phantom: PhantomData<Context>,
}

num_inputs!{Scatter, 3}

num_outputs!{Scatter, 1}

inputs!{Scatter, 
    0 => ("DATA", "Tensor to be updated."),
    1 => ("INDICES", "1-D list of indices on the first dimension of X_0 that need to be updated"),
    2 => ("UPDATES", "Update slices, with shape len(INDICES) + shape(X_0)[1:]")
}

outputs!{Scatter, 
    0 => ("OUTPUT", "The updated output.")
}

args!{Scatter, 
    0 => ("axis", "*(type: int; default: 1)* Which dimension to scatter on.")
}

allow_inplace!{Scatter, vec![(0, 0)]}

input_tags!{
    ScatterOp
    {
        Data,
        Indices,
        Updates
    }
}

impl<Context> ScatterOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "axis", axis_, 1)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            TORCH_CHECK(
            Context::GetDeviceType() == kCPU,
            "ScatterOp currently only supports CPU.")

        return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, this->template Input<Tensor>(INDICES, CPU));
        */
    }
    
    #[inline] pub fn do_run_with_type<IndexType>(&mut self) -> bool {
        todo!();
        /*
            const Tensor& data = Input(DATA);
        const Tensor& indices = Input(INDICES);
        const Tensor& updates = Input(UPDATES);
        const TypeMeta dataType = data.dtype();
        size_t item_bytesize = dataType.itemsize();

        // ONNX allows negative axis to index from the back, valid range: [-r, r].
        axis_ = data.canonical_axis_index(axis_);

        CAFFE_ENFORCE_GE(
            data.dim(), axis_ + 1, "DATA should be at least [axis+1]-D");
        CAFFE_ENFORCE_GE(axis_, 0, "Axis should be non-negative");
        CAFFE_ENFORCE_LT(axis_, data.dim(), "Axis out of range");

        Tensor* output = Output(0, data.sizes().vec(), at::dtype(dataType));
        output->CopyFrom(data);
        char* out = static_cast<char*>(output->raw_mutable_data(dataType));

        // Succeed if size of output is zero, which can happen for empty batch which
        // would have data dimension size of 0.
        // This *must* be done AFTER output->raw_mutable_data() above as that has
        // important allocation side effect that we must see.
        if (output->numel() == 0) {
          return true;
        }

        const IndexType* idxs = indices.template data<IndexType>();
        const char* src_base = static_cast<const char*>(updates.raw_data());

        const int64_t outer_dims_product = indices.size_to_dim(axis_);

        const int64_t dst_indexing_axis_dim = data.size(axis_);

        const int64_t idxs_block_size = indices.size_from_dim(axis_ + 1);
        const int64_t src_block_size = updates.size_from_dim(axis_ + 1);
        const int64_t dst_block_size = data.size_from_dim(axis_ + 1);

        const int64_t idxs_batch_size = indices.size_from_dim(axis_);
        const int64_t src_batch_size = updates.size_from_dim(axis_);
        const int64_t dst_batch_size = data.size_from_dim(axis_);

        const int64_t N = indices.size(axis_);

        check_indexarray_range<IndexType>(idxs, N, dst_indexing_axis_dim);

        // For a 3-D tensor, dst is updated as:
        //    dst[i][idxs[i][j][k]][k] = src[i][j][k]  # if dim == 1
        // where i, j, k are iterating over their corresponding axis I, J, K.
        // For a given i, j, k tuple.
        // idxs offset can be computed as i * J_src * K + j * K + k.
        // src offset can be computed as i * J_src * K + j * K + k.
        // dst offset can be computed as i * J_dst * K + idxs[idxs_offset] * K + K
        // Note that idxs and src should have the same rank and shape.
        // dst should have the same rank as idxs and src, but the dimension of dim
        // axis can be different. That is why in the above equation, there is the
        // difference of J_src and J_dst.
        for (int64_t outer_batch = 0; outer_batch < outer_dims_product;
             ++outer_batch) {
          for (int64_t i = 0; i < N; ++i) {
            for (int64_t inner_batch = 0; inner_batch < idxs_block_size;
                 ++inner_batch) {
              auto idxs_elem_idx =
                  outer_batch * idxs_batch_size + i * idxs_block_size + inner_batch;
              auto src_elem_idx =
                  outer_batch * src_batch_size + i * src_block_size + inner_batch;
              auto dst_elem_idx = outer_batch * dst_batch_size +
                  idxs[idxs_elem_idx] * dst_block_size + inner_batch;

              auto src = src_base + src_elem_idx * item_bytesize;
              auto dst = out + dst_elem_idx * item_bytesize;
              context_.CopyItemsSameDevice(dataType, 1, src, dst);
            }
          }
        }
        return true;
        */
    }

    /**
      | Check that indices fall within dimension
      | array size with CAFFE_ENFORCE.
      |
      */
    #[inline] pub fn check_indexarray_range<IndexType>(
        &mut self, 
        indices:           *const IndexType,
        n:                 i64,
        indexing_axis_dim: IndexType) 
    {
        todo!();
        /*
            for (auto i = 0; i < n; ++i) {
          auto idx = indices[i];
          CAFFE_ENFORCE(
              0 <= idx && idx < indexing_axis_dim,
              "INDICES element is out of DATA bounds, id=",
              idx,
              " axis_dim=",
              indexing_axis_dim);
        }
        */
    }
}
