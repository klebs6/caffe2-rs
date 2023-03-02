crate::ix!();

/**
  | Raise if there is NaN or Inf values in
  | the input tensor.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct EnforceFiniteOp<Context> {
    storage: OperatorStorage,
    context: Context,

    ws:      *mut Workspace,
    buffer:  Tensor, //{CPU};
}

num_inputs!{EnforceFinite, 1}

num_outputs!{EnforceFinite, 0}

inputs!{EnforceFinite, 
    0 => ("input", "Input tensor")
}

impl<Context> EnforceFiniteOp<Context> {
    
    pub fn new<Args>(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws), ws_(ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
        */
    }

    #[inline] pub fn enforce_on_cpu<T>(&mut self, input: &Tensor) {
        todo!();
        /*
            const T* input_data = input.template data<T>();
            auto size = input.numel();

            for (auto i = 0; i < size; i++) {
              auto isfinite = std::isfinite(input_data[i]);
              if (!isfinite) {
                LogBlobFiniteness();
              }
              CAFFE_ENFORCE_FINITE(
                isfinite,
                  "Index ",
                  i,
                  " is not finite (e.g., NaN, Inf): ",
                  input_data[i]);
            }
        */
    }

    /**
      | LogBlobFiniteness sums every tensor
      | in the workspace and logs whether it's
      | finite or not.
      |
      */
    #[inline] pub fn log_blob_finiteness(&mut self)  {
        
        todo!();
        /*
            // This uses the aten interfaces to compute the sum and finiteness of the
        // tensors which are not present by default on xplat and mobile builds.
    #if defined(EXPOSE_C2_OPS) || \
        !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
        for (const std::string& blob_name : ws_->Blobs()) {
          try {
            const auto& blob = ws_->GetBlob(blob_name);
            if (blob != nullptr && blob->IsType<Tensor>()) {
              Tensor* c2Tensor = blob->GetMutable<Tensor>();
              const at::Tensor& tensor = static_cast<at::Tensor>(*c2Tensor);
              bool blob_finite = tensor.sum().isfinite().cpu().data_ptr<bool>()[0];
              LOG(INFO) << "blob " << blob_name << " isfinite=" << (blob_finite ? "true" : "false");
            }
          } catch (const std::exception& ex) {
            LOG(ERROR) << "failed to check finiteness for " << blob_name << ": " << ex.what();
          }
        }
    #endif
        */
    }
}

impl EnforceFiniteOp<CPUContext> {

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            EnforceOnCPU<T>(Input(0));
          return true;
        */
    }
}

register_cpu_operator!{EnforceFinite, EnforceFiniteOp<CPUContext>}

should_not_do_gradient!{EnforceFinite}
