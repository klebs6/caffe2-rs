crate::ix!();

caffe_known_type!{detail::ScratchWorkspaces}

register_cpu_operator!{RecurrentNetwork, RecurrentNetworkOp<CPUContext>}

register_cpu_operator!{
    RecurrentNetworkGradient,
    RecurrentNetworkGradientOp<CPUContext>
}

/**
  | Internal RNN operator.
  |
  */
register_cpu_operator!{
    rnn_internal_accumulate_gradient_input,
    AccumulateInputGradientOp<CPUContext>
}

num_inputs!{rnn_internal_accumulate_gradient_input, 3}

num_outputs!{rnn_internal_accumulate_gradient_input, (1,INT_MAX)}

enforce_inplace!{rnn_internal_accumulate_gradient_input, 
    vec![(2, 0)]}

private_operator!{rnn_internal_accumulate_gradient_input}

/**
  | Internal RNN operator.
  |
  */
register_cpu_operator!{
    rnn_internal_apply_link,
    RNNApplyLinkOp<CPUContext>
}

num_inputs!{rnn_internal_apply_link, 2}

num_outputs!{rnn_internal_apply_link, 2}

enforce_inplace!{rnn_internal_apply_link, vec![(1, 1)]}

private_operator!{rnn_internal_apply_link}

