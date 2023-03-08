crate::ix!();

use crate::{
    CPUContext,
    Operator,
    OperatorDef,
    Workspace
};

pub trait RmsPropUpdate {

    fn rmsprop_update(
        &mut self,
        n:        i32,
        g:        *const f32,
        ms:       *const f32,
        mom:      *const f32,
        ng:       *mut f32,
        nms:      *mut f32,
        nmom:     *mut f32,
        decay:    f32,
        momentum: f32,
        epsilon:  f32,
        lr:       *const f32);
}

impl RmsPropUpdate for CPUContext {

    fn rmsprop_update(
        &mut self,
        n:        i32,
        g:        *const f32,
        ms:       *const f32,
        mom:      *const f32,
        ng:       *mut f32,
        nms:      *mut f32,
        nmom:     *mut f32,
        decay:    f32,
        momentum: f32,
        epsilon:  f32,
        lr:       *const f32) 
    {
        todo!();
        /*
          ConstEigenVectorArrayMap<float> gVec(g, N);
          ConstEigenVectorArrayMap<float> msVec(ms, N);
          ConstEigenVectorArrayMap<float> momVec(mom, N);
          // Update new mean square estimate
          EigenVectorArrayMap<float> nmsVec(nms, N);
          nmsVec = msVec + (1.0f - decay) * (gVec * gVec - msVec);
          // Update momentum estimate
          EigenVectorArrayMap<float> nmomVec(nmom, N);
          nmomVec = momVec * momentum + lr[0] * gVec / (epsilon + nmsVec).sqrt();
          // New gradient is the momentum
          EigenVectorArrayMap<float>(ng, N) = nmomVec;
        */
    }
}

/**
 | Computes the RMSProp update
 |
 | (http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).
 |
 | Concretely, given inputs (grad, mean_squares, mom,
 | lr), computes:
 |
 |     mean_squares_o = mean_squares + (1 - decay) * (square(grad) - mean_squares)
 |     mom_o = momentum * mom + lr * grad / sqrt(epsilon + mean_squares_o)
 |     grad_o = mom_o
 |
 | Returns (grad_o, mean_squares_o, mom_o).
 */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct RmsPropOp<T,Context> {
    context:  Context,
    decay:    T,// {0.9};
    momentum: T,// {0.0};
    epsilon:  T,// {1e-8};
}

impl<T,Context> Operator for RmsPropOp<T,Context> {

}

register_cpu_operator!{RmsProp, RmsPropOp<f32, CPUContext>}

num_inputs!{RmsProp, 4}

num_outputs!{RmsProp, 3}

allow_inplace!{RmsProp, vec![(0, 0), (1, 1), (2, 2)]}

should_not_do_gradient!{RmsProp}

input_tags!{
    RmsPropOp
    {
        Grad,
        MeanSquares,
        Momentum,
        Lr
    }
}

output_tags!{
    RmsPropOp
    {
        OutputGrad,
        OutputMeanSquares,
        OutputMomentum
    }
}

impl<T,Context> RmsPropOp<T,Context> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator(operator_def, ws),
            decay_(this->template GetSingleArgument<float>("decay", 0.9f)),
            momentum_(this->template GetSingleArgument<float>("momentum", 0.0f)),
            epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE(Input(LR).numel() == 1);
        CAFFE_ENFORCE(Input(GRAD).numel() == Input(MEAN_SQUARES).numel());
        CAFFE_ENFORCE(Input(GRAD).numel() == Input(OUTPUT_MOMENTUM).numel());
        Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
        Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
        Output(OUTPUT_MEAN_SQUARES)->ResizeLike(Input(MEAN_SQUARES));
        Output(OUTPUT_MOMENTUM)->ResizeLike(Input(MOMENTUM));
        rmsprop_update<Context>(
            Input(GRAD).numel(),
            Input(GRAD).template data<T>(),
            Input(MEAN_SQUARES).template data<T>(),
            Input(MOMENTUM).template data<T>(),
            Output(OUTPUT_GRAD)->template mutable_data<T>(),
            Output(OUTPUT_MEAN_SQUARES)->template mutable_data<T>(),
            Output(OUTPUT_MOMENTUM)->template mutable_data<T>(),
            decay_,
            momentum_,
            epsilon_,
            Input(LR).template data<T>(),
            &context_);
        return true;
        */
    }
}
