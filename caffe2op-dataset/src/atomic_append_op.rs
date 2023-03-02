crate::ix!();

///----------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AtomicAppendOp<Context> {

    storage: OperatorStorage,
    context: Context,
}

num_inputs!{AtomicAppend, (3,INT_MAX)}

num_outputs!{AtomicAppend, (1,INT_MAX)}

allow_inplace!{AtomicAppend, 
    |input: i32, output: i32| {
        input == out + 1
    }
}

impl<Context> AtomicAppendOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& mutex = OperatorStorage::Input<std::unique_ptr<std::mutex>>(0);
        const auto numFields = (InputSize() - 1) / 2;
        CAFFE_ENFORCE(OutputSize() == numFields);

        std::lock_guard<std::mutex> guard(*mutex);

        // 1: checks
        for (int i = 0; i < numFields; ++i) {
          auto& a = Input(1 + i);
          auto& b = Input(1 + i + numFields);
          auto* c = Output(i);
          CAFFE_ENFORCE(b.dim() >= 1);
          if (a.numel() == 0) {
            continue;
          }
          CAFFE_ENFORCE(
              (void*)&a == (void*)c, "Appended-to arguments must be in-place.");
          CAFFE_ENFORCE(c->dim() == b.dim());
          CAFFE_ENFORCE(b.dim() == c->dim());
          CAFFE_ENFORCE(a.dtype() == b.dtype());
          for (int j = 1; j < a.dim(); ++j) {
            CAFFE_ENFORCE(a.sizes()[j] == b.sizes()[j]);
          }
        }

        // 2: copies
        for (int i = 0; i < numFields; ++i) {
          auto& a = Input(1 + i);
          auto& b = Input(1 + i + numFields);
          auto* c = Output(i);
          if (a.numel() == 0 && a.size(0) == 0) {
            c->CopyFrom(b);
            continue;
          }
          auto oldSize = c->numel();
          c->Extend(b.sizes()[0], kDatasetGrowthPct);
          auto* dst = (char*)c->raw_mutable_data() + oldSize * b.dtype().itemsize();
          context_.CopyItemsSameDevice(b.dtype(), b.numel(), b.raw_data(), dst);
        }
        return true;
        */
    }
}

