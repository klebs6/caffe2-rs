crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SliceGradientOp<Context> {
    storage:            OperatorStorage,
    context:            Context,
    starts:             Vec<i64>,
    ends:               Vec<i64>,
    statically_inited:  bool,
    starts_host:        Tensor,
    ends_host:          Tensor,
}

register_cpu_gradient_operator!{SliceGradient, SliceGradientOp<CPUContext>}

tensor_inference_function!{SliceGradient, 
    /* ([](const OperatorDef& /*def*/,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      out.at(0) = in.at(0);
      return out;
    }) */
}
