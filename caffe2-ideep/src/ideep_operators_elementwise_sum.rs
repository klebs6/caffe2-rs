crate::ix!();

use crate::{
    SameTypeAsInput,
    IDEEPOperator,
    SumOp,
    BinaryElementwiseOp,
    CPUContext,
    IDEEPFallbackOp,
    AddFunctor,
    SkipIndices,
    Workspace,
    OperatorDef,
};

pub type FALLBACK_SUM = IDEEPFallbackOp<SumOp<CPUContext>, dyn SkipIndices<0>>;
pub type FALLBACK_ADD<N: Num> = 
IDEEPFallbackOp<BinaryElementwiseOp<N, CPUContext, AddFunctor<CPUContext>, SameTypeAsInput>, dyn SkipIndices<0>>;

pub struct IDEEPSumOp<N: Num> {
    //USE_IDEEP_DEF_ALIASES();
    //USE_IDEEP_OPERATOR_FUNCTIONS();
    base: IDEEPOperator,

    fallback_sum: FALLBACK_SUM,
    fallback_add: FALLBACK_ADD<N>,
    phantom: PhantomData<N>,
}

input_tags!{
    IDEEPSumOp {
        Input0
    }
}

output_tags!{
    IDEEPSumOp {
        Output
    }
}

impl<N: Num> IDEEPSumOp<N> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws),
            fallback_sum_(operator_def, ws),
            fallback_add_(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            itensor::dims input_dims;
        bool fallback_to_cpu = false;
        vector<itensor> inputs_itensor;

        // We only support element-wise sum for ideep tensors here.
        // If a CPU tensor is detected in input list, we have to fallback
        // to corresponding CPU operator.
        for (int i = 0; i < InputSize(); ++i) {
          if (OperatorStorage::InputBlob(i).template IsType<itensor>()) {
            auto& tensor_ideep = Input(i);
            if (input_dims.empty()) {
              input_dims = tensor_ideep.get_dims();
            } else if (input_dims != tensor_ideep.get_dims()) {
              fallback_to_cpu = true;
              break;
            }
            inputs_itensor.emplace_back(tensor_ideep);
          } else {
            CAFFE_ENFORCE(
                BlobIsTensorType(OperatorStorage::InputBlob(i), CPU),
                "Expect cpu tensor if not itensor");
            fallback_to_cpu = true;
            break;
          }
        }

        if (!fallback_to_cpu) {
          auto* Y = Output(OUTPUT);
          if (InputSize() == 1) {
            const auto& X = Input(INPUT0);
            ideep::direct_copy::compute(X, *Y);
          } else {
            const vector<float> scales(InputSize(), 1.0);
            ideep::sum::compute(scales, inputs_itensor, *Y);
          }
          return true;
        }

        if (InputSize() == 2) {
          return fallback_add_.Run(0);
        }

        return fallback_sum_.Run(0);
        */
    }
}

register_ideep_operator!{Sum, IDEEPSumOp}

register_ideep_operator!{Add, IDEEPSumOp}
