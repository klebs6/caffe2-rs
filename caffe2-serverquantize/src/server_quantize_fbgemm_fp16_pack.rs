
crate::ix!();


/**
    class Engine = DefaultEngine,
    bool TransposeWeight = true,
    typename TPacked = fbgemm::float16>
 */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct FbGemmPackOp<Context,Engine,const TransposeWeight: bool,TPacked> {
    storage: OperatorStorage,
    context: Context,

    axis:        usize, // default = 1

    /// Do not pack the layout, for testing only
    no_packing:  bool,
    phantomA: PhantomData<TPacked>,
    phantomB: PhantomData<Engine>,
}

impl<Context, Engine, const TransposeWeight: bool, TPacked> 
FbGemmPackOp<Context, Engine, TransposeWeight, TPacked> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            axis_(this->template GetSingleArgument<int32_t>("axis_w", 1)),
            no_packing_( this->template GetSingleArgument<int32_t>("no_packing", 0))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
        const auto canonical_axis = X.canonical_axis_index(axis_);
        const auto N = X.size_to_dim(canonical_axis);
        const auto K = X.size_from_dim(canonical_axis);

        fbgemm::PackedGemmMatrixFP16* resultPtr;
        if (TransposeWeight) {
          resultPtr = new fbgemm::PackedGemmMatrixFP16(
              fbgemm::matrix_op_t::Transpose,
              K,
              N,
              1.0f, /*alpha*/
              X.template data<float>());
        } else {
          resultPtr = new fbgemm::PackedGemmMatrixFP16(
              fbgemm::matrix_op_t::NoTranspose,
              N,
              K,
              1.0f, /*alpha*/
              X.template data<float>());
        }

        if (no_packing_) {
          C10_LOG_FIRST_N(WARNING, 10) << "no_packing will be deprecated soon";

          vector<fbgemm::float16> src_mat(resultPtr->matSize());
          fbgemm::float16* pmat = resultPtr->pmat();
          memcpy(
              src_mat.data(), pmat, resultPtr->matSize() * sizeof(fbgemm::float16));
          resultPtr->unpackFromSrc(fbgemm::matrix_op_t::Transpose, src_mat.data());
        }

        auto* Y =
            this->template Output<unique_ptr<fbgemm::PackedGemmMatrixFP16>>(0);
        Y->reset(resultPtr);
        return true;
        */
    }
}

/// Expilictly register TypeMeta
caffe_known_type!{unique_ptr<fbgemm::PackedGemmMatrixFP16>}

///Prepack weight for fbgemm
register_cpu_operator!{
    FbGemmPack,
    FbGemmPackOp<CPUContext, DefaultEngine, true, fbgemm::float16>}

num_inputs!{FbGemmPack, 1}

num_outputs!{FbGemmPack, 1}

inputs!{FbGemmPack, 
    0 => ("X", "row major format weight matrix")
}

outputs!{FbGemmPack, 
    0 => ("Y", "Block row major packed format weight matrix")
}

tensor_inference_function!{FbGemmPack, /* ([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_FLOAT16);
      return out;
    }) */
}

allow_inplace!{FbGemmPack, vec![(0, 0)]}

///-----------------
///Prepack weight for fbgemm
register_cpu_operator!{
    FbGemmPackTranspose,
    FbGemmPackOp<CPUContext, DefaultEngine, false, fbgemm::float16>}

num_inputs!{FbGemmPackTranspose, 1}

num_outputs!{FbGemmPackTranspose, 1}

inputs!{FbGemmPackTranspose, 
    0 => ("X", "col major format weight matrix")
}

outputs!{FbGemmPackTranspose, 
    0 => ("Y", "Block col major packed format weight matrix")
}

tensor_inference_function!{FbGemmPackTranspose, /* ([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];

      X.set_dims(1, in[0].dims(0));
      X.set_dims(0, in[0].dims(1));
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_FLOAT16);
      return out;
    }) */}

allow_inplace!{FbGemmPackTranspose, vec![(0, 0)]}

