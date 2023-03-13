crate::ix!();



#[inline] pub fn weight_scale_update<T, Context>(
    n:                  i32,
    w:                  *const T,
    scale:              T,
    iter:               i64,
    stepsize:           i64,
    update_upper_bound: i64,
    nw:                 *mut T,
    context:            *mut Context) 
{
    todo!();
    /*
        const auto w_size = N * sizeof(float);
      if (iter % stepsize != 0 || iter >= update_upper_bound) {
        memcpy(nw, w, w_size);
        return;
      }
      // perform the weight scaling
      caffe2::math::Scale<T, T, Context>(N, scale, w, nw, context);
    */
}


/**
Every `stepsize` iterations, multiply the weights by a constant `scale`:
    nw = w * scale
*/
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct WeightScaleOp<Context> {
    storage: OperatorStorage,
    context: Context,
    stepsize_:           i64,
    update_upper_bound_: i64,
    scale_:              f32,
}

impl<Context> RunOnDevice for WeightScaleOp<Context> {

    #[inline] fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            Output(OUTPUT_WEIGHTS)->ResizeLike(Input(WEIGHTS));
        return DispatchHelper<TensorTypes<float>>::call(this, Input(WEIGHTS));
        */
    }
}

impl<Context> WeightScaleOp<Context> {

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto iter =
                OperatorStorage::Input<Tensor>(ITER, CPU).template data<int64_t>()[0] + 1;

            weight_scale_update<T, Context>(
                Input(WEIGHTS).size(),
                Input(WEIGHTS).template data<T>(),
                scale_,
                iter,
                stepsize_,
                update_upper_bound_,
                Output(OUTPUT_WEIGHTS)->template mutable_data<T>(),
                &context_);
            return true;
        */
    }

    #[inline] pub fn do_run_with_type_cuda_context<T>(&mut self) -> bool {
        todo!();
        /*
            const auto iter =
              OperatorStorage::Input<Tensor>(ITER, CPU).template data<int64_t>()[0] + 1;
          weight_scale_update_kernel<T>(
              Input(WEIGHTS).size(),
              Input(WEIGHTS).template data<T>(),
              scale_,
              iter,
              stepsize_,
              update_upper_bound_,
              Output(OUTPUT_WEIGHTS)->template mutable_data<T>(),
              &context_);
          return true;
        */
    }
    
    pub fn new(
        operator_def: &OperatorDef,
        ws: *mut Workspace) -> Self 
    {
        todo!();
        /*
            : Operator(operator_def, ws),
            stepsize_(OperatorStorage::GetSingleArgument<int64_t>(
                "stepsize",
                int64_t::max)),
            update_upper_bound_(OperatorStorage::GetSingleArgument<int64_t>(
                "upper_bound_iter",
                int64_t::max)),
            scale_(this->template GetSingleArgument<float>("scale", 1.0f))
        */
    }
}

input_tags!{
    WeightScaleOp {
        Weights,
        Iter
    }
}

output_tags!{
    WeightScaleOp {
        OutputWeights
    }
}

impl<Context> Operator for WeightScaleOp<Context> {
}

register_cpu_operator![
    WeightScale, 
    WeightScaleOp<CPUContext>
];

num_inputs!{WeightScale,  2}

num_outputs!{WeightScale, 1}

allow_inplace!{WeightScale, vec![(0, 0), (1, 1)]}

device_inference_function!{
    /*
    WeightScale,

     [](const OperatorDef& def) {
      auto op_device =
          def.has_device_option() ? def.device_option() : DeviceOption();
      vector<DeviceOption> in_dev(def.input_size(), op_device);
      vector<DeviceOption> out_dev(def.output_size(), op_device);
      // ITER input lives on CPU
      in_dev[1] = DeviceOption();
      return std::make_pair(in_dev, out_dev);
    }
    */
}

inputs!{WeightScale,
    0 => ("w",                "Current weights"),
    1 => ("iter",             "Training Iteration")
}

outputs!{WeightScale,
    0 => ("nw",               "Updated weights")
}

args!{WeightScale,
    0 => ("stepsize",         "Every iteration number to do weight scaling"),
    1 => ("upper_bound_iter", "After iter passes this bound, do not perform the weight rescaling"),
    2 => ("scale",            "The multiplicative factor applied to weights.")
}

should_not_do_gradient!{WeightScale}

register_cuda_operator!{
    WeightScale, 
    WeightScaleOp<CUDAContext>
}

#[inline] pub fn weight_scale_update_kernel<T>(
    n:                  i32,
    w:                  *const T,
    scale:              &T,
    iter:               i64,
    stepsize:           i64,
    update_upper_bound: i64,
    nw:                 *mut T,
    context:            *mut CUDAContext)
{
    todo!();
    /*
        const auto w_size = N * sizeof(float);
      if (iter % stepsize != 0 || iter >= update_upper_bound) {
        (void)cudaMemcpy(nw, w, w_size, cudaMemcpyDefault);
      } else {
        // perform the weight scaling
        caffe2::math::Scale<T, T, CUDAContext>(N, scale, w, nw, context);
      }
    */
}
