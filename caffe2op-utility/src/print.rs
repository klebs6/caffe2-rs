crate::ix!();

/**
  | Logs shape and contents of input tensor
  | to stderr or to a file.
  |
  */
#[USE_DISPATCH_HELPER]
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct PrintOp<Context, W: Write> {
    storage:           OperatorStorage,
    context:           Context,
    tensor_printer:    TensorPrinter<W>,
    every_n:           i32,
    occurrences_mod_n: i32, // default = 0
}

should_not_do_gradient!{Print}

num_inputs!{Print, 1}

num_outputs!{Print, 0}

inputs!{Print, 
    0 => ("tensor", "The tensor to print.")
}

args!{Print, 
    0 => ("to_file", "(bool) if 1, saves contents to the root folder of the current workspace, appending the tensor contents to a file named after the blob name. Otherwise, logs to stderr."),
    1 => ("limit",   "(int, default 0) If set, prints the first `limit` elements of tensor. If 0, prints the first `k_limit_default`(1000) elements of tensor"),
    2 => ("every_n", "(int, default 1) Print tensor every `every_n` runs")
}

impl<Context, W: Write> PrintOp<Context,W> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            tensor_printer_(
                operator_def.input(0),
                this->template GetSingleArgument<int>("to_file", 0)
                    ? ws->RootFolder() + "/" + operator_def.input(0) +
                        kPrintFileExtension
                    : "",
                this->template GetSingleArgument<int>("limit", 0)),
            every_n_(this->template GetSingleArgument<int>("every_n", 1)) 

        CAFFE_ENFORCE_GE(every_n_, 1);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (++occurrences_mod_n_ > every_n_) {
          occurrences_mod_n_ -= every_n_;
        }
        if (occurrences_mod_n_ != 1) {
          return true;
        }

        if (!this->InputIsTensorType(0, Context::GetDeviceType()) &&
            !this->InputIsTensorType(0, CPU)) {
          LOG(INFO) << "Blob of type: "
                    << OperatorStorage::Inputs().at(0)->meta().name();
          return true;
        }
        // special-case empty tensors since they may have no meta()
        if (Input(0).numel() == 0) {
          tensor_printer_.PrintMeta(Input(0));
          return true;
        }

        using Types = TensorTypes<
            float,
            double,
            int,
            long,
            bool,
            char,
            unsigned char,
            std::string>;

        if (this->InputIsTensorType(0, CPU)) {
          return DispatchHelper<Types>::call(
              this, this->template Input<Tensor>(0, CPU));
        } else {
          return DispatchHelper<Types>::call(this, Input(0));
        }
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            // A simple strategy to copy tensor if needed, and have the tensor pointer
        // pointing to the right instantiation. Note that tensor_copy_if_needed
        // will handle memory deallocation itself so no smart pointer is needed.
        const TensorCPU* tensor;
        Tensor tensor_copy_if_needed(CPU);
        if (this->InputIsTensorType(0, CPU)) {
          tensor = &this->template Input<Tensor>(0, CPU);
        } else {
          // sync copy
          tensor_copy_if_needed.CopyFrom(Input(0));
          tensor = &tensor_copy_if_needed;
        }
        tensor_printer_.Print<T>(*tensor);
        return true;
        */
    }
}
