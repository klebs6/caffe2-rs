crate::ix!();

/**
  | Similar to WeightedSum, computes the
  | weighted sum of several tensors, with
  | the difference that inputs are sliced
  | tensors.
  | 
  | The first tensor has to be in-place and
  | only slices of it on the first dimension
  | as indexed by
  | 
  | INDICES will be updated.
  | 
  | -----------
  | @brief
  | 
  | Update slices of the tensor in-place
  | with weighted sum.
  | 
  | ScatterWeightedSumOp is similar to
  | WeightedSum and computes the weighted
  | sum of several tensors.
  | 
  | The first tensor has to be in-place and
  | only slices of it on the first dimension
  | as indexed by
  | 
  | INDICES will be updated.
  | 
  | Input:
  | 
  | X_0 - tensor to be updated
  | 
  | weight_0 - scalar weight for X_0, applied
  | only to slices affected,
  | 
  | INDICES - 1-D list of indices on the
  | first dimension of X_0 that need to be
  | updated
  | 
  | X_1 - update slices, has to have shape
  | of len(INDICES) + shape(X_0)[1:]
  | 
  | weight_1 - scalar weight for X_1 update
  | X_2, weight_2, ...
  | 
  | Output:
  | 
  | X_0 - has to be exactly the same tensor
  | as the input 0
  | 
  | -----------
  | @note
  | 
  | The op pretty much ignores the exact
  | shapes of the input arguments and cares
  | only about sizes.
  | 
  | It's done for performance consideration
  | to avoid unnecessary reshapes.
  | 
  | Only first dimension of X_0 is important,
  | let's call it N.
  | 
  | If M is the total size of X_0 and K is the
  | size of
  | 
  | INDICES then X_i is assumed to be of shape
  | K x (M / N) regardless of the real shape.
  | ----------
  | @note
  | 
  | Each update in INDICES is applied independently
  | which means that if duplicated elements
  | are present in INDICES the corresponding
  | slice of X_0 will be scaled multiple
  | times.
  | 
  | Manual collapsing of INDICES is required
  | beforehand if necessary.
  | ----------
  | @note
  | 
  | Updates are applied sequentially by
  | inputs which might have undesired consequences
  | if the input tensor is accessed concurrently
  | by different op (e.g. when doing Hogwild).
  | 
  | Other threads might see intermediate
  | results even on individual slice level,
  | e.g. X_0 scaled by weight_0 but without
  | any updates applied.
  | 
  | Currently only works on CPU because
  | of access to
  | 
  | INDICES.
  | ----------
  | @note
  | 
  | The op pretty much ignores the exact
  | shapes of the input arguments and cares
  | only about sizes.
  | 
  | It's done for performance consideration
  | to avoid unnecessary reshapes.
  | 
  | Only first dimension of X_0 is important,
  | let's call it N.
  | 
  | If M is the total size of X_0 and K is the
  | size of INDICES then X_i is assumed to
  | be of shape K x (M / N) regardless of the
  | real shape.
  | ----------
  | @note
  | 
  | Each update in INDICES is applied independently
  | which means that if duplicated elements
  | are present in INDICES the corresponding
  | slice of X_0 will be scaled multiple
  | times.
  | 
  | Manual collapsing of INDICES is required
  | beforehand if necessary.
  | ----------
  | @note
  | 
  | Updates are applied sequentially by
  | inputs which might have undesired consequences
  | if the input tensor is accessed concurrently
  | by different op (e.g. when doing Hogwild).
  | 
  | Other threads might see intermediate
  | results even on individual slice level,
  | 
  | e.g. X_0 scaled by weight_0 but without
  | any updates applied.
  | 
  | For now really works only on CPU because
  | of INDICES access
  |
  */
#[USE_DISPATCH_HELPER]
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ScatterWeightedSumOp<T,Context> {
    storage:        OperatorStorage,
    context:        Context,
    x_data_host:    Tensor,
    weights_host:   Tensor,
    x_data_device:  Tensor,
    weights_device: Tensor,
    phantom:        PhantomData<T>,
}

num_outputs!{ScatterWeightedSum, 1}

inputs!{ScatterWeightedSum, 
    0 => ("X_0",      "Tensor to be updated."),
    1 => ("Weight_0", "Scalar weight for X_0, applied only to slices affected."),
    2 => ("INDICES",  "1-D list of indices on the first dimension of X_0 that need to be updated"),
    3 => ("X_1",      "Update slices, with shape len(INDICES) + shape(X_0)[1:]"),
    4 => ("Weight_1", "Scalar weight for X_1 update")
}

outputs!{ScatterWeightedSum, 
    0 => ("X_0", "Has to be exactly the same tensor as the input 0")
}

enforce_inplace!{ScatterWeightedSum, vec![(0, 0)]}

num_inputs!{ScatterWeightedSum, 
    |n: i32| {
        n > 3 && (n - 3) % 2 == 0
    }
}

should_not_do_gradient!{ScatterWeightedSum}

impl<T,Context> ScatterWeightedSumOp<T, Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(2));
        */
    }
    
    #[inline] pub fn do_run_with_type<Index>(&mut self) -> bool {
        todo!();
        /*
            int64_t block_size = Input(0).size_from_dim(1);
        return DispatchHelper<FixedValues<1>, Index>::call(this, block_size);
        */
    }
    
    #[inline] pub fn do_run_with_value<Index, const FixedSize: i32>(&mut self) -> bool {
        todo!();
        /*
            CAFFE_ENFORCE_EQ(InputSize() % 2, 1);
        auto& X0 = Input(0);
        auto& weight0 = Input(1);
        auto& indices = Input(2);
        auto* output = Output(0);
        CAFFE_ENFORCE_EQ(&X0, output, "In place operation is required");

        if (X0.numel() == 0) {
          return true;
        }
        CAFFE_ENFORCE_GT(X0.numel(), 0);
        CAFFE_ENFORCE_GT(X0.dim(), 0, "X0 has to be at least the vector");
        CAFFE_ENFORCE_EQ(weight0.numel(), 1);
        int64_t M = X0.numel();
        int64_t N = X0.size(0);
        int64_t K = indices.numel();
        int64_t block_size = M / N;
        T* data = output->template mutable_data<T>();
        const Index* idxs = indices.template data<Index>();
        T w0 = *weight0.template data<T>();
        // It's most likely a constant so exact comparison is fine
        if (w0 != 1.0) {
          for (int i = 0; i < K; ++i) {
            Index idx = idxs[i];
            CAFFE_ENFORCE(
                0 <= idx && idx < N,
                "Index out of bounds: ",
                idx,
                ", range 0 to ",
                N);
            math::ScaleFixedSize<T, Context, FixedSize>(
                block_size,
                w0,
                data + block_size * idx,
                data + block_size * idx,
                &context_);
          }
        }
        for (int inp = 3; inp < InputSize(); inp += 2) {
          auto& X = Input(inp);
          auto& weight = Input(inp + 1);
          CAFFE_ENFORCE_EQ(X.numel(), block_size * K);
          CAFFE_ENFORCE_EQ(weight.numel(), 1);
          const T* x_data = X.template data<T>();
          T w = *weight.template data<T>();
          for (int i = 0; i < K; ++i) {
            Index idx = idxs[i];
            // double-checking the indices, but it's fine as it's DCHECK only
            DCHECK(0 <= idx && idx < N)
                << "Index out of bounds: " << idx << ", range 0 to " << N;
            math::AxpyFixedSize<T, Context, FixedSize>(
                block_size,
                w,
                x_data + block_size * i,
                data + block_size * idx,
                &context_);
          }
        }
        return true;
        */
    }
}
