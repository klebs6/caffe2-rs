crate::ix!();

#[test] fn div_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Div",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", np.array([[18,8],[2,9]]))
    workspace.FeedBlob("B", np.array([[9,2],[3,2]]))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("C"))

    A:
    [[18  8]
     [ 2  9]]
    B:
    [[9 2]
     [3 2]]
    C:
    [[2 4]
     [0 4]]

    */
}

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct DivFunctor<Context> {
    storage: OperatorStorage,
    context: Context,
    //.FillUsing(MathDocGenerator("division", kDivExample))
}

num_inputs!{Div, 2}

num_outputs!{Div, 1}

cost_inference_function!{Div, /* (PointwiseCostInference<1>) */ }

tensor_inference_function!{Div, /* (ElementwiseOpShapeInference) */}

allow_inplace!{Div, vec![(0, 0)]}

inherit_onnx_schema!{Div}

///--------------------------------
num_inputs!{DivGradient, (3,4)}

num_outputs!{DivGradient, 2}

tensor_inference_function!{DivGradient, /* (ElementwiseGradientOpShapeInference) */}

allow_inplace!{DivGradient, vec![(0, 0)]}

///--------------------------------

impl<Context> DivFunctor<Context> {

    #[inline] pub fn forward<TIn, TOut>(
        &mut self, 
        a_dims:    &Vec<i32>,
        b_dims:    &Vec<i32>,
        a:         *const TIn,
        b:         *const TIn,
        c:         *mut TOut,
        context:   *mut Context) -> bool 
    {
        todo!();
        /*
            math::Div(
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
}

impl DivFunctor<CPUContext> {

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
        context:    *mut CPUContext) -> bool 
    {
        todo!();
        /*
            if (A_dims == B_dims) {
            const int size = std::accumulate(
                A_dims.cbegin(), A_dims.cend(), 1, std::multiplies<int>());
            EigenVectorMap<TGrad>(dB, size) =
                -ConstEigenVectorArrayMap<TGrad>(dC, size) *
                ConstEigenVectorArrayMap<TOut>(C, size) /
                ConstEigenVectorArrayMap<TIn>(B, size);
            math::Div(size, dC, B, dA, context);
            return true;
          }
          const int ndim = std::max(A_dims.size(), B_dims.size());
          std::vector<int> A_broadcast_dims(ndim);
          std::vector<int> B_broadcast_dims(ndim);
          std::vector<int> C_broadcast_dims(ndim);
          math::utils::ComputeBroadcastBinaryOpDims(
              A_dims.size(),
              A_dims.data(),
              B_dims.size(),
              B_dims.data(),
              A_broadcast_dims.data(),
              B_broadcast_dims.data(),
              C_broadcast_dims.data());
          if (dA == dC) {
            ComputeDivGradient<TGrad, TIn, TOut>(
                ndim,
                A_broadcast_dims.data(),
                B_broadcast_dims.data(),
                C_broadcast_dims.data(),
                dC,
                B,
                C,
                nullptr,
                dB,
                context);
            math::Div(
                A_dims.size(),
                A_dims.data(),
                B_dims.size(),
                B_dims.data(),
                dC,
                B,
                dA,
                context);
          } else {
            ComputeDivGradient<TGrad, TIn, TOut>(
                ndim,
                A_broadcast_dims.data(),
                B_broadcast_dims.data(),
                C_broadcast_dims.data(),
                dC,
                B,
                C,
                dA,
                dB,
                context);
          }
          return true;
        */
    }
}

register_cpu_operator!{
    Div,
    BinaryElementwiseOp<NumericTypes, CPUContext, DivFunctor<CPUContext>>
}

#[inline] pub fn compute_div_gradient<TGrad, TIn, TOut>(
    ndim:      i32,
    a_dims:    *const i32,
    b_dims:    *const i32,
    c_dims:    *const i32,
    dC:        *const TGrad,
    b:         *const TIn,
    c:         *const TOut,
    dA:        *mut TGrad,
    dB:        *mut TGrad,
    context:   *mut CPUContext) 
{
    todo!();
    /*
        const int A_size =
          std::accumulate(A_dims, A_dims + ndim, 1, std::multiplies<int>());
      const int B_size =
          std::accumulate(B_dims, B_dims + ndim, 1, std::multiplies<int>());
      const int C_size =
          std::accumulate(C_dims, C_dims + ndim, 1, std::multiplies<int>());
      if (dA != nullptr) {
        math::Set<TGrad, CPUContext>(A_size, TGrad(0), dA, context);
      }
      math::Set<TGrad, CPUContext>(B_size, TGrad(0), dB, context);
      std::vector<int> index(ndim, 0);
      for (int C_index = 0; C_index < C_size; ++C_index) {
        const int B_index =
            math::utils::GetIndexFromDims(ndim, B_dims, index.data());
        dB[B_index] += -dC[C_index] * C[C_index] / B[B_index];
        if (dA != nullptr) {
          const int A_index =
              math::utils::GetIndexFromDims(ndim, A_dims, index.data());
          dA[A_index] += dC[C_index] / B[B_index];
        }
        math::utils::IncreaseIndexInDims(ndim, C_dims, index.data());
      }
    */
}
