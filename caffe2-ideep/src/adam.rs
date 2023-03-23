crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_OPERATOR_FUNCTIONS]
pub struct IDEEPAdamOp<T: Float> {
    beta1:   T, //0.9
    beta2:   T, //0.999
    epsilon: T, //1e-8
}

input_tags!{
    IDEEPAdamOp {
        Param,
        Moment1,
        Moment2,
        Grad,
        LR,
        Iter
    }
}

output_tags!{
    IDEEPAdamOp {
        OutputParam,
        OutputMoment1,
        OutputMoment2,
        OutputGrad
    }
}

impl<T: Float> IDEEPAdamOp<T> {

    fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();

        /*
      IDEEPOperator(operator_def, ws),
        beta1_(OperatorStorage::GetSingleArgument<float>("beta1", 0.9f)),
        beta2_(OperatorStorage::GetSingleArgument<float>("beta2", 0.999f)),
        epsilon_(OperatorStorage::GetSingleArgument<float>("epsilon", 1e-5f)) 
        */
    }
}

impl<T: Float> RunOnDevice for IDEEPAdamOp<T> {

    #[inline] fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Iter live on the CPU
            CAFFE_ENFORCE(OperatorStorage::InputIsTensorType(ITER, CPU));
            const auto& params = Input(PARAM);
            const auto& moment_1 = Input(MOMENT_1);
            const auto& moment_2 = Input(MOMENT_2);
            const auto& grad = Input(GRAD);
            // TODO: Use itensor after 0-dim is supported. Now use CPU tensor.
            const auto& lr = OperatorStorage::Input<TensorCPU>(LR, CPU);
            auto* out_params = Output(OUTPUT_PARAM);
            auto* out_moment1 = Output(OUTPUT_MOMENT_1);
            auto* out_moment2 = Output(OUTPUT_MOMENT_2);

            CAFFE_ENFORCE(lr.size() == 1);
            CAFFE_ENFORCE(grad.get_nelems() == params.get_nelems());
            CAFFE_ENFORCE(grad.get_nelems() == moment_1.get_nelems());
            CAFFE_ENFORCE(grad.get_nelems() == moment_2.get_nelems());
            if (params != *out_params)
                out_params->init(params.get_descriptor());
            if (moment_1 != *out_moment1)
                out_moment1->init(moment_1.get_descriptor());
            if (moment_2 != *out_moment2)
                out_moment2->init(moment_2.get_descriptor());
            const auto w = static_cast<float *>(params.get_data_handle());
            const auto g = static_cast<float *>(grad.get_data_handle());
            const auto m = static_cast<float *>(moment_1.get_data_handle());
            const auto v = static_cast<float *>(moment_2.get_data_handle());
            auto nw = static_cast<float *>(out_params->get_data_handle());
            auto nm = static_cast<float *>(out_moment1->get_data_handle());
            auto nv = static_cast<float *>(out_moment2->get_data_handle());
            const auto nlr = lr.template data<T>();
            const auto iter =
                OperatorStorage::Input<TensorCPU>(ITER, CPU).template data<int64_t>()[0];
            const auto t = iter + 1;
            const auto correction =
                std::sqrt(T(1.) - std::pow(beta2_, t)) / (T(1.) - std::pow(beta1_, t));
            if (OutputSize() == 3) {
              adam_ideep_compute(
                  grad.get_nelems(),
                  w,
                  g,
                  m,
                  v,
                  nw,
                  nm,
                  nv,
                  beta1_,
                  beta2_,
                  epsilon_,
                  correction,
                  nlr);
            } else {
              auto* out_grad = Output(OUTPUT_GRAD);
              if (grad != *out_grad)
                out_grad->init(grad.get_descriptor());
              auto ng = static_cast<float *>(out_grad->get_data_handle());
              adam_ideep_compute_output_grad(
                  grad.get_nelems(),
                  w,
                  g,
                  m,
                  v,
                  nw,
                  nm,
                  nv,
                  ng,
                  beta1_,
                  beta2_,
                  epsilon_,
                  correction,
                  nlr);
            }

            return true;
        */
    }
}
