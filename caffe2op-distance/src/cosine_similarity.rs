crate::ix!();

/**
 | This op takes two input float tensors of the same
 | size, $X$ and $Y$, and produces one output float
 | tensor , $Z$, calculated as the cosine similarity
 | between $X$ and $Y$. 
 |
 | Recall, the cosine similarity between two tensors
 | $X$ and $Y$ is defined as:
 |
 | $$\mathbf{Z}=CosineSimilarity(\mathbf{X},\mathbf{Y}) =
 | \frac{\mathbf{X}\cdot\mathbf{Y}}{\|\mathbf{X}\|\|\mathbf{Y}\|} =
 | \frac{\sum_n^{i=1}X_iY_i}{\sqrt{\sum_n^{i=1}X_i^2}\sqrt{\sum_n^{i=1}Y_i^2}}$$
 |
 | Github Links:
 | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.h
 | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.cc
 |
 | <details>
 |
 | <summary> <b>Example</b> </summary>
 |
 */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct CosineSimilarityOp<T, Context> {

    storage: OperatorStorage,
    context: Context,
    aux:     Tensor,
    phantom: PhantomData<T>,
}

register_cpu_operator!{
    CosineSimilarity, 
    CosineSimilarityOp<f32, CPUContext>
}

num_inputs!{CosineSimilarity, 2}

num_outputs!{CosineSimilarity, 1}

inputs!{CosineSimilarity, 
    0 => ("X", "1D or 2D input tensor"),
    1 => ("Y", "1D or 2D input tensor (must have the same shape as X)")
}

outputs!{CosineSimilarity, 
    0 => ("Z", "1D output tensor")
}

identical_type_and_shape_of_input_dim!{CosineSimilarity, (0, 0)}

impl<T,Context> CosineSimilarityOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}

input_tags!{
    CosineSimilarityOp {
        XIn,
        YIn
    }
}

output_tags!{
    CosineSimilarityOp {
        CosOut
    }
}
