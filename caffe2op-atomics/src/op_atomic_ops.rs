crate::ix!();

/**
  | Creates an unlocked mutex and returns
  | it in a unique_ptr blob.
  |
  */
pub struct CreateMutexOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{CreateMutex, 0}

num_outputs!{CreateMutex, 1}

outputs!{CreateMutex, 
    0 => ("mutex_ptr", "Blob containing a std::unique_ptr<mutex>.")
}

scalar_type!{
    CreateMutex, 
    TensorProto_DataType_UNDEFINED
}

should_not_do_gradient!{CreateMutex}

register_cpu_operator!{
    CreateMutex, 
    CreateMutexOp
}

register_ideep_operator!{
    CreateMutex, 
    IDEEPFallbackOp::<CreateMutexOp, SkipIndices<0>>
}

impl<Context> CreateMutexOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            *OperatorStorage::Output<std::unique_ptr<std::mutex>>(0) =
            std::unique_ptr<std::mutex>(new std::mutex);
        return true;
        */
    }
}

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

/**
  | Create an unique_ptr blob to hold an
  | atomic<bool>
  |
  */
pub struct CreateAtomicBoolOp {
    storage: OperatorStorage,
    context: CPUContext,
}

num_inputs!{CreateAtomicBool, 0}

num_outputs!{CreateAtomicBool, 1}

outputs!{CreateAtomicBool, 
    0 => ("atomic_bool", "Blob containing a unique_ptr<atomic<bool>>")
}

should_not_do_gradient!{CreateAtomicBool}

register_cpu_operator!{CreateAtomicBool, CreateAtomicBoolOp}

impl CreateAtomicBoolOp {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            *OperatorStorage::Output<std::unique_ptr<std::atomic<bool>>>(0) =
            std::unique_ptr<std::atomic<bool>>(new std::atomic<bool>(false));
        return true;
        */
    }
}

/**
  | Set an atomic<bool> to true if the given
  | condition bool variable is true
  |
  */
pub struct ConditionalSetAtomicBoolOp {
    storage: OperatorStorage,
    context: CPUContext,
}

num_inputs!{ConditionalSetAtomicBool, 2}

num_outputs!{ConditionalSetAtomicBool, 0}

inputs!{ConditionalSetAtomicBool, 
    0 => ("atomic_bool", "Blob containing a unique_ptr<atomic<bool>>"),
    1 => ("condition", "Blob containing a bool")
}

should_not_do_gradient!{ConditionalSetAtomicBool}

register_cpu_operator!{ConditionalSetAtomicBool, ConditionalSetAtomicBoolOp}

impl ConditionalSetAtomicBoolOp {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& ptr =
            OperatorStorage::Input<std::unique_ptr<std::atomic<bool>>>(ATOMIC_BOOL);
        if (Input(CONDITION).data<bool>()[0]) {
          ptr->store(true);
        }
        return true;
        */
    }
}

input_tags!{
    ConditionalSetAtomicBoolOp {
        AtomicBool,
        Condition
    }
}

/**
  | Copy the value of an atomic<bool> to
  | a bool
  |
  */
pub struct CheckAtomicBoolOp {
    storage: OperatorStorage,
    context: CPUContext,
}

num_inputs!{CheckAtomicBool, 1}

num_outputs!{CheckAtomicBool, 1}

inputs!{CheckAtomicBool, 
    0 => ("atomic_bool", "Blob containing a unique_ptr<atomic<bool>>")
}

outputs!{CheckAtomicBool, 
    0 => ("value", "Copy of the value for the atomic<bool>")
}

should_not_do_gradient!{CheckAtomicBool}

register_cpu_operator!{CheckAtomicBool, CheckAtomicBoolOp}

impl CheckAtomicBoolOp {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& ptr = OperatorStorage::Input<std::unique_ptr<std::atomic<bool>>>(0);
        Output(0)->Resize(1);
        *Output(0)->template mutable_data<bool>() = ptr->load();
        return true;
        */
    }
}
