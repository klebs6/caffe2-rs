crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_OPERATOR_FUNCTIONS]
pub struct IDEEPWeightedSumOp {
    base: IDEEPOperator,
} 

impl IDEEPWeightedSumOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(InputSize() % 2, 0);
        auto ndims = Input(0).ndims();
        auto nelems = Input(0).get_nelems();
        auto w_nelems = Input(1).get_nelems();
        CAFFE_ENFORCE_GT(nelems, 0);
        CAFFE_ENFORCE_EQ(w_nelems, 1);
        auto* output = Output(0);
        std::vector<float> scales;
        scales.reserve(InputSize() / 2);
        std::vector<itensor> inputs;
        inputs.reserve(InputSize() / 2);
        for (int i = 0; i < InputSize(); i += 2) {
          auto& X = Input(i);
          CAFFE_ENFORCE(X.ndims() == ndims);
          CAFFE_ENFORCE(X.get_nelems() == nelems);
          CAFFE_ENFORCE(Input(i + 1).get_nelems() == w_nelems);
          inputs.push_back(X);
          auto scale = static_cast<float *>(Input(i + 1).get_data_handle());
          scales.push_back(scale[0]);
        }

        ideep::sum::compute(scales, inputs, *output);

        return true;
        */
    }
}
