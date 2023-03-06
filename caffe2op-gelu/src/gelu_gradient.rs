crate::ix!();

pub struct GeluGradientFunctor<Context> {
    fast_gelu: bool,

    phantom: PhantomData<Context>,
}

impl<Context> GeluGradientFunctor<Context> {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
        todo!();
        /*
            : fast_gelu(op.GetSingleArgument<bool>("fast_gelu", false))
        */
    }
}

pub type GeluGradientOp<Context> = BinaryElementwiseWithArgsOp<
    TensorTypes<f32>,
    Context,
    GeluGradientFunctor<Context>,
    SameTypeAsInput>;

num_inputs!{GeluGradient, 2}

num_outputs!{GeluGradient, 1}

identical_type_and_shape_of_input!{GeluGradient, 1}

register_cpu_operator!{GeluGradient, GeluGradientOp<CPUContext>}
