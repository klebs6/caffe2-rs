crate::ix!();

/**
 | Mask op designed for use in attention mechanisms
 | for sequence modeling tasks.
 |
 | Supports batching: given batch_dim, collapses dims
 | 0 through batch_dim into a single dimension,
 | e.g. if tensor dims are [4,2,1,3,4] and
 | batch_dim=2, first collapse tensor to [4*2*1,3,4],
 | then mask each batch [i,:,:].
 |
 | Two current operating modes:
 |
 | 1) Given a 2D input tensor and 1D tensor of
 | sequence lengths, for each row i in the input
 | tensor, set elements in that row to -inf if their
 | column index j >= sequence_lengths[i]. This mode
 | takes two inputs and argument mode = 'sequence'
 |
 | 2) Triangular mask. Given row index i and column
 | index j, set elements to -inf given the following
 | conditions:
 |
 |       mode='upper', x_ij = -inf if j < i
 |       mode='lower', x_ij = -inf if j > i
 |       mode='upperdiag', x_ij = -inf if j <= i
 |       mode='lowerdiag', x_ij = -inf if j >= i
 |
 | This mode takes one input.
 |
 | 3) Window Mask. Given a 2D input tensor and 1D
 | tensor of window centers, for each row i in the
 | input tensor, set elements in that row to -inf if
 | their column index j outside [center - radius,
 | center + radius].
 |
 | This mode takes two inputs and argument mode
 | = 'sequence'.
 |
 | Argument 'radius' should be provided.
 */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SequenceMaskOp<Context> {
    storage:      OperatorStorage,
    context:      Context,
    axis:         i32,
    radius:       i32,
    mode:         String,
    grad:         bool,
    fill_val:     f32,
    batch:        i32,
    repeat_from:  i32,
}

register_cpu_operator!{
    SequenceMask, 
    SequenceMaskOp<CPUContext>
}

num_inputs!{SequenceMask, (1,2)}

num_outputs!{SequenceMask, 1}

inputs!{SequenceMask, 
    0 => ("input",             "Tensor to apply masking to"),
    1 => ("sequence_lengths",  "1D Tensor of sequence lengths for mode #1")
}

outputs!{SequenceMask, 
    0 => ("masked_tensor", "Input tensor with masking applied")
}

args!{SequenceMask, 
    0 => ("mode",             "(string) Mode selection. Possible values: 'sequence', 'upper', 'lower', 'upperdiag', 'lowerdiag'"),
    1 => ("axis",             "(int) Beginning axis of row elements. All dimensions to the left will be treated as row indices and those to the right (inclusive) will be treated as column indices in the 2D mask"),
    2 => ("grad",             "(bool) operate in gradient mode"),
    3 => ("radius",           "(int) radius of windows in window mode"),
    4 => ("batch",            "(int) batch dimension of tensor (optional)"),
    5 => ("repeat_from_axis", "(int) used when mask should be repeated for one or more data dimensions (beginning at this axis).  (currently only supported for sequence mode without batch argument)")
}

impl<Context> SequenceMaskOp<Context> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            axis_(this->template GetSingleArgument<int>("axis", 1)),
            radius_(this->template GetSingleArgument<int>("radius", 10)),
            grad_(this->template GetSingleArgument<bool>("grad", false)),
            fill_val_(this->template GetSingleArgument<float>(
                    "fill_val",
                    -1.0f * std::numeric_limits<float>::infinity())) 

                // Mode argument is required
                mode_ = GetArgument(operator_def, "mode").s();
            // batch argument is optional, but if not given, we don't want a default val
            if (HasArgument("batch")) {
                batch_ = GetArgument(operator_def, "batch").i();
            }

            if (HasArgument("repeat_from_axis")) {
                CAFFE_ENFORCE(
                    mode_ == "sequence",
                    "repeat_from_axis currently only supported in sequence mode.");
                CAFFE_ENFORCE(
                    !HasArgument("batch"),
                    "repeat_from_axis and batch not currently supported together.");
                repeat_from_ =
                    this->template GetSingleArgument<int>("repeat_from_axis", -1);
            }
        */
    }
}
