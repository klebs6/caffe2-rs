crate::ix!();

pub type NumericTypes = TensorTypes<(i32, i64, f32, f64)>;
pub type IntTypes     = TensorTypes<(i32, i64)>;
pub type BoolTypes    = TensorTypes<(bool)>;
pub type IntBoolTypes = TensorTypes<(i32, i64, bool)>; // discrete types

///-------------------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct UnaryElementwiseWithArgsOp<InputTypes, Context, Functor, OutputTypeMap=SameTypeAsInput> {

    storage:    OperatorStorage,
    context:    Context,

    functor:    Functor,
    phantomIT:  PhantomData<InputTypes>,
    phantomOTM: PhantomData<OutputTypeMap>,
}

impl<InputTypes,Context,Functor> UnaryElementwiseWithArgsOp<InputTypes,Context,Functor> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...), functor_(*this)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<InputTypes>::call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<T>() -> bool {
        todo!();
        /*
            const auto& X = Input(0);

            auto* Y = Output(
                0, X.sizes(), at::dtype<typename OutputTypeMap::template type<T>>());
            return functor_(
                X.numel(),
                X.template data<T>(),
                Y->template mutable_data<typename OutputTypeMap::template type<T>>(),
                &context_);
        */
    }
}
