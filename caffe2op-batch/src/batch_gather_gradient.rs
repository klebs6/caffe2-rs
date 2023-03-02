crate::ix!();

pub struct BatchGatherGradientOp<Context> {
    storage: OperatorStorage,
    context: Context,
    axis:        i32,
    match_outer: bool,
}

input_tags!{
    BatchGatherGradientOp {
        Data,
        Indices,
        Grad
    }
}

num_inputs!{BatchGatherGradient,  3}

num_outputs!{BatchGatherGradient, 1}

impl<Context> BatchGatherGradientOp<Context> {

    /**
      | Constructor to receive axis in case
      | it was passed for GatherOp gradient,
      | use default of 1 for batch gather otherwise.
      |
      */
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
        Self {
            context:     Operator<Context>::new(args),
            axis:        1,
            match_outer: false,
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, this->template Input<Tensor>(INDICES, CPU));
        */
    }

    #[inline] pub fn do_run_with_type<TInd>() -> bool {
        todo!();
        /*
            return DispatchHelper<
                TensorTypes2<float, GenericTensorImplementation>,
                TInd>::call(this, Input(DATA));
        */
    }

    #[inline] pub fn do_run_with_type2<TInd, TData>() -> bool {
        todo!();
        /*
            auto& data = Input(DATA);
            auto& indices = Input(INDICES);
            auto& grad = Input(GRAD);

            // ONNX allows negative axis to index from the back, valid range: [-r, r].
            int axis = axis_;
            bool match_outer = match_outer_;
            if (axis < 0) {
              axis = data.dim() + axis;
            }

            CAFFE_ENFORCE_GE(data.dim(), 2, "DATA should be at least 2-D");
            // Outer dimensions of input data and gradient should be the same
            // because they are preserved for gathers with axis > 0.
            for (int acheck = 0; acheck < axis; acheck++) {
              CAFFE_ENFORCE_EQ(
                  data.size(acheck),
                  grad.size(acheck),
                  "batch gather outer dimensions should match");
            }

            auto* output = Output(0, data.sizes(), at::dtype<TData>());
            TData* out_data = output->template mutable_data<TData>();
            if (data.numel() <= 0) {
              return true;
            }
            memset(out_data, 0, output->nbytes());

            const TData* grad_data = grad.template data<TData>();
            const TInd* idxs = indices.template data<TInd>();

            auto outer_dims_product = data.size_to_dim(axis);
            auto batch_size = data.size_from_dim(axis);
            auto block_size = data.size_from_dim(axis + 1);
            auto N = indices.numel();

            auto idx_inner_dims_product = indices.size_from_dim(axis);
            if (match_outer) {
              CAFFE_ENFORCE_GE(axis, 1, "Axis should be at least 1");
              for (auto i = 0; i < axis; i++) {
                CAFFE_ENFORCE_EQ(
                    data.size(i),
                    indices.size(i),
                    "INDICES must have the same outer dims as DATA (before dim AXIS)");
              }
              N = idx_inner_dims_product;
            }

            auto gathered_grad_batch_size = N * block_size;
            // Check indexing bounds.
            auto src_indexing_axis_dim = data.dim(axis);
            gather_helper::check_indexarray_range<TInd>(
                idxs, N, src_indexing_axis_dim, false);

            for (auto batch = 0; batch < outer_dims_product; ++batch) {
              auto grad_batch_base = grad_data + batch * gathered_grad_batch_size;
              auto out_batch_base = out_data + batch * batch_size;

              for (auto i = 0; i < N; ++i) {
                auto idx = idxs[i];
                if (match_outer) {
                  idx = idxs[batch * idx_inner_dims_product + i];
                }
                if (idx < 0) {
                  idx = idx + src_indexing_axis_dim;
                }
                if (block_size == 1) {
                  out_batch_base[idx] += grad_batch_base[i];
                } else {
                  math::Add(
                      block_size,
                      out_batch_base + idx * block_size,
                      grad_batch_base + i * block_size,
                      out_batch_base + idx * block_size,
                      &context_);
                }
              }
            }
            return true;
        */
    }

    #[inline] pub fn do_run_with_other_type2<TInd>() -> bool {
        todo!();
        /*
            CAFFE_THROW(
                "BatchGatherGradient is not implemented on tensor of type ",
                Input(DATA).meta().name(),
                "consider adding it as a type in the DispatchHelper list or "
                "implementing a generic version (which won't work for "
                "duplicated indices though)");
        */
    }
}

pub struct GetBatchGatherGradient;

impl GetGradientDefs for GetBatchGatherGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            using Op = BatchGatherOp<CPUContext>;
        return SingleGradientDef(
            "BatchGatherGradient",
            "",
            vector<string>{I(Op::DATA), I(Op::INDICES), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_cpu_operator!{
    BatchGatherGradient, 
    BatchGatherGradientOp<CPUContext>
}

register_gradient!{
    BatchGather, 
    GetBatchGatherGradient
}
