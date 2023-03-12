crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct FullyConnectedGradientOp<Context, Engine, const TransposeWeight: bool> {
    storage:         OperatorStorage,
    context:         Context,
    axis:            usize, //{1};
    axis_w:          usize, //{1};
    bias_multiplier: Option<Tensor>,
    float16_compute: bool,
    phantomE:        PhantomData<Engine>,
}

num_inputs!{FCGradient, 3}

num_outputs!{FCGradient, (2,3)}

tensor_inference_function!{FCGradient, /* std::bind(FCGradientShapeInference, _1, _2, false) */}

cost_inference_function!{FCGradient, /* std::bind(CostInferenceForFCGradient, _1, _2, false) */}

register_cpu_gradient_operator!{
    FCGradient,
    FullyConnectedGradientOp<CPUContext>
}
