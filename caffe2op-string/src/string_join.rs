crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct StringJoinOp<Context> {
    storage:    OperatorStorage,
    context:    Context,
    delimiter:  String,
    axis:       i32,
}

pub type StringElementwiseOp<
ScalarFunctor, 
    TypeMap = FixedType<String>>
    = UnaryElementwiseWithArgsOp<
    TensorTypes<String>,
    CPUContext,
    ForEach<ScalarFunctor>,
    TypeMap>;

impl<Context> StringJoinOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            delimiter_( this->template GetSingleArgument<std::string>("delimiter", ",")),
            axis_(this->template GetSingleArgument<int>("axis", 0)) 

        CAFFE_ENFORCE(axis_ == 0 || axis_ == 1);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<
            float,
            double,
            int8_t,
            uint8_t,
            int16_t,
            uint16_t,
            int32_t,
            int64_t,
            std::string,
            bool>>::call(this, Input(0));
        */
    }
}
