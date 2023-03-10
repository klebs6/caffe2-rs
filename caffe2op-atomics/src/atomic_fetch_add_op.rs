crate::ix!();

/**
  | Given a mutex and two int32 scalar tensors,
  | performs an atomic fetch add by mutating
  | the first argument and adding it to the
  | second input argument.
  | 
  | Returns the updated integer and the
  | value prior to the update.
  |
  */
pub struct AtomicFetchAddOp<IntType> {
    storage: OperatorStorage,
    context: CPUContext,
    phantom: PhantomData<IntType>,
}

num_inputs!{AtomicFetchAdd, 3}

num_outputs!{AtomicFetchAdd, 2}

inputs!{AtomicFetchAdd, 
    0 => ("mutex_ptr", "Blob containing to a unique_ptr<mutex>"),
    1 => ("mut_value", "Value to be mutated after the sum."),
    2 => ("increment", "Value to add to the first operand.")
}

outputs!{AtomicFetchAdd, 
    0 => ("mut_value", "Mutated value after sum. Usually same as input 1."),
    1 => ("fetched_value", "Value of the first operand before sum.")
}

allow_inplace!{AtomicFetchAdd, vec![(1, 0)]}

should_not_do_gradient!{AtomicFetchAdd}

register_cpu_operator!{AtomicFetchAdd,   AtomicFetchAddOp<i32>}

register_cpu_operator!{AtomicFetchAdd64, AtomicFetchAddOp<i64>}

impl<T> AtomicFetchAddOp<T> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& mutex = OperatorStorage::Input<std::unique_ptr<std::mutex>>(0);
        std::lock_guard<std::mutex> lg(*mutex);
        auto& a = Input(1);
        auto& b = Input(2);
        auto* c = Output(0);
        auto* d = Output(1);
        c->Resize();
        d->Resize();
        auto* aPtr = a.template data<IntType>();
        auto* bPtr = b.template data<IntType>();
        auto* cPtr = c->template mutable_data<IntType>();
        auto* dPtr = d->template mutable_data<IntType>();
        *dPtr = *aPtr;
        *cPtr = *aPtr + *bPtr;
        return true;
        */
    }
}
