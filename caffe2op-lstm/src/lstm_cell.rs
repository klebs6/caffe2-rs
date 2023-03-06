crate::ix!();

pub struct LSTMCell {
    context: *mut CPUContext,
}

impl LSTMCell {
    
    pub fn new(context: *mut CPUContext) -> Self {
        todo!();
        /*
            : context_(context)
        */
    }

    pub fn invoke(&self, input: &Tensor, hidden: &TensorTuple, params: &CellParams) -> TensorTuple {
        todo!();
        /*
        const auto& hx = std::get<0>(hidden);
        const auto& cx = std::get<1>(hidden);
        auto linear_ih = params.linear_ih(input);
        auto linear_hh = params.linear_hh(hx);
        auto gates = add(linear_ih, linear_hh, context_);
        auto chunked_gates = chunk(gates, 4, 1, context_);
        auto ingate = sigmoid(chunked_gates[0]);
        auto forgetgate = sigmoid(chunked_gates[1]);
        auto cellgate = tanh(chunked_gates[2], context_);
        auto outgate = sigmoid(chunked_gates[3]);

        auto cy =
            add(mul(forgetgate, cx, context_),
                mul(ingate, cellgate, context_),
                context_);
        auto hy = mul(outgate, tanh(cy, context_), context_);
        return std::make_tuple(std::move(hy), std::move(cy));
        */
    }
}
