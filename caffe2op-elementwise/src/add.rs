crate::ix!();

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
