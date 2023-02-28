crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct KeySplitOp<T, Context> {
    storage:           OperatorStorage,
    context:           Context,
    categorical_limit: i32,
    phantom:           PhantomData<T>,
}

impl<T, Context> KeySplitOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
                 categorical_limit_( this->template GetSingleArgument<int>("categorical_limit", 0)) 

                     CAFFE_ENFORCE_GT(categorical_limit_, 0);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& keys = Input(0);
        int N = keys.numel();
        const T* keys_data = keys.template data<T>();
        std::vector<int> counts(categorical_limit_);
        std::vector<int*> eids(categorical_limit_);
        for (int k = 0; k < categorical_limit_; k++) {
          counts[k] = 0;
        }
        for (int i = 0; i < N; i++) {
          int k = keys_data[i];
          CAFFE_ENFORCE_GT(categorical_limit_, k);
          CAFFE_ENFORCE_GE(k, 0);
          counts[k]++;
        }
        for (int k = 0; k < categorical_limit_; k++) {
          auto* eid = Output(k, {counts[k]}, at::dtype<int>());
          eids[k] = eid->template mutable_data<int>();
          counts[k] = 0;
        }
        for (int i = 0; i < N; i++) {
          int k = keys_data[i];
          eids[k][counts[k]++] = i;
        }
        return true;
        */
    }
}

register_cpu_operator!{KeySplit, KeySplitOp<i64, CPUContext>}

no_gradient!{KeySplitOp}

num_inputs!{KeySplit, 1}

num_outputs!{KeySplit, (1,INT_MAX)}
