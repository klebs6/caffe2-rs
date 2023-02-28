crate::ix!();

use crate::{
    OperatorDef,
    GradientMakerBase,
};

#[test] fn add_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Add",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", np.array([[1,2],[3,4]]))
    workspace.FeedBlob("B", np.array([[5,6],[7,8]]))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("C"))

    A:
    [[1 2]
     [3 4]]
    B:
    [[5 6]
     [7 8]]
    C:
    [[ 6  8]
     [10 12]]

    */
}

///-----------------------------
pub struct AddFunctor<Context> {

    //.FillUsing(MathDocGenerator("addition", kAddExample))
    phantom: PhantomData<Context>,
}

num_inputs!{Add, 2}

num_outputs!{Add, 1}

cost_inference_function!{Add, /* (PointwiseCostInference<1>) */ }

tensor_inference_function!{Add, /* (ElementwiseOpShapeInference) */}

allow_inplace!{Add, vec![(0, 0), (1, 0)]}

inherit_onnx_schema!{Add}

impl<Context> AddFunctor<Context> {

    #[inline] pub fn forward<TIn, TOut>(
        &mut self,
        a_dims:   &Vec<i32>,
        b_dims:   &Vec<i32>,
        a:        *const TIn,
        b:        *const TIn,
        c:        *mut TOut,
        context:  *mut Context) -> bool 
    {
        todo!();
        /*
            math::Add(
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
        &mut self,
        a_dims:     &Vec<i32>,
        b_dims:     &Vec<i32>,
        dC:         *const TGrad,
        a:          *const TIn,
        b:          *const TIn,
        c:          *const TOut,
        dA:         *mut TGrad,
        dB:         *mut TGrad,
        context:    *mut Context) -> bool 
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
                TGrad(1),
                dC,
                dB,
                context);
            return true;
        */
    }
}

register_cpu_operator!{
    Add,
    BinaryElementwiseOp<NumericTypes, CPUContext, AddFunctor<CPUContext>>
}

///-------------------------------
register_cpu_operator!{
    AddGradient,
    BinaryElementwiseGradientOp<
        NumericTypes,
        CPUContext,
        AddFunctor<CPUContext>>
}

num_inputs!{AddGradient, 3}

num_outputs!{AddGradient, 2}

tensor_inference_function!{AddGradient, /* (ElementwiseGradientOpShapeInference) */}

allow_inplace!{AddGradient, vec![(0, 0), (0, 1)]}

///-------------------------------
pub struct GetAddGradient;

impl GetGradientDefs for GetAddGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "AddGradient",
            "",
            std::vector<std::string>{GO(0), I(0), I(1)},
            std::vector<std::string>{GI(0), GI(1)});
        */
    }
}

register_gradient!{Add, GetAddGradient}

register_cuda_operator!{
    Add,
    BinaryElementwiseOp<NumericTypes, CUDAContext, AddFunctor<CUDAContext>>
}

register_cuda_operator!{
    AddGradient,
    BinaryElementwiseGradientOp<
    NumericTypes,
    CUDAContext,
    AddFunctor<CUDAContext>>
}
