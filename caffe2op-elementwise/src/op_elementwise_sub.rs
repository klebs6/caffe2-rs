crate::ix!();

use crate::{
    OperatorDef,
    GradientMakerBase,
};

#[test] fn sub_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Sub",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", np.array([[10,12],[4,14]]))
    workspace.FeedBlob("B", np.array([[5,16],[1,19]]))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("C"))

    A:
    [[10 12]
     [ 4 14]]
    B:
    [[ 5 16]
     [ 1 19]]
    C:
    [[ 5 -4]
     [ 3 -5]]

    */
}

struct SubFunctor<Context> { 
    phantom: PhantomData<Context>,
}

num_inputs!{Sub, 2}

num_outputs!{Sub, 1}

allow_inplace!{Sub, vec![(0, 0), (1, 0)]}

inherit_onnx_schema!{Sub}

tensor_inference_function!{Sub, ElementwiseOpShapeInference}

cost_inference_function!{Sub, /* PointwiseCostInference<1> */ }

impl<Context> SubFunctor<Context> {

    #[inline] pub fn forward<TIn, TOut>(
        &self, 
        a_dims:  &Vec<i32>,
        b_dims:  &Vec<i32>,
        a:       *const TIn,
        b:       *const TIn,
        c:       *mut TOut,
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Sub(
                A_dims.size(),
                A_dims.data(),
                B_dims.size(),
                B_dims.data(),
                A,
                B,
                C,
                context);
            return true;
        */
    }


    #[inline] pub fn backward<TGrad, TIn, TOut>(
        &self, 
        a_dims:   &Vec<i32>,
        b_dims:   &Vec<i32>,
        dC:       *const TGrad,
        a:        *const TIn,
        b:        *const TIn,
        c:        *const TOut,
        dA:       *mut TGrad,
        dB:       *mut TGrad,
        context:  *mut Context) -> bool 
    {
        todo!();
        /*
            const std::vector<int> C_dims =
                elementwise_ops_utils::ComputeBinaryBroadcastForwardDims(
                    A_dims, B_dims);
            std::vector<int> A_back_dims;
            std::vector<int> B_back_dims;
            elementwise_ops_utils::ComputeBinaryBroadcastBackwardDims(
                A_dims, B_dims, &A_back_dims, &B_back_dims);
            math::ReduceSum(
                C_dims.size(),
                C_dims.data(),
                A_back_dims.data(),
                TGrad(1),
                dC,
                dA,
                context);
            math::ReduceSum(
                C_dims.size(),
                C_dims.data(),
                B_back_dims.data(),
                TGrad(-1),
                dC,
                dB,
                context);
            return true;
        */
    }
}

register_cpu_operator!{
    Sub,
    BinaryElementwiseOp<NumericTypes, CPUContext, SubFunctor<CPUContext>>
}

///-----------------------------------------
register_cpu_operator!{
    SubGradient,
    BinaryElementwiseGradientOp<
        NumericTypes,
        CPUContext,
        SubFunctor<CPUContext>>}

num_inputs!{SubGradient, 3}

num_outputs!{SubGradient, 2}

tensor_inference_function!{SubGradient, ElementwiseGradientOpShapeInference}

allow_inplace!{SubGradient, vec![(0, 0), (0, 1)]}

pub struct GetSubGradient { }

impl GetGradientDefs for GetSubGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "SubGradient",
            "",
            std::vector<std::string>{GO(0), I(0), I(1)},
            std::vector<std::string>{GI(0), GI(1)});
        */
    }
}

register_gradient!{Sub, GetSubGradient}

register_cuda_operator!{
    Sub,
    BinaryElementwiseOp<NumericTypes, CUDAContext, SubFunctor<CUDAContext>>
}

register_cuda_operator!{
    SubGradient,
    BinaryElementwiseGradientOp<
        NumericTypes,
        CUDAContext,
        SubFunctor<CUDAContext>>
}

