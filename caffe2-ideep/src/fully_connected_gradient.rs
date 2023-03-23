crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_OPERATOR_FUNCTIONS]
pub struct IDEEPFullyConnectedGradientOp {
    base:   IDEEPOperator,
    axis:   usize, //{1};
    axis_w: usize, //{1};
}

input_tags!{
    IDEEPFullyConnectedGradientOp {
        Input,
        Filter,
        OutputGrad
    }
}

output_tags!{
    IDEEPFullyConnectedGradientOp {
        FilterGrad,
        BiasGrad,
        InputGrad
    }
}

impl IDEEPFullyConnectedGradientOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws),
            axis_(OperatorStorage::GetSingleArgument<int32_t>("axis", 1)),
            axis_w_(OperatorStorage::GetSingleArgument<int32_t>("axis_w", 1))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(INPUT);
        const auto& filter = Input(FILTER);
        const auto& dY = Input(OUTPUT_GRAD);
        auto* dfilter = Output(FILTER_GRAD);
        auto* dbias = Output(BIAS_GRAD);

        itensor X_in = X;
        auto X_dims = CanonicalDims(X_in.get_dims(), axis_);
        if (X_in.get_dims() != X_dims) {
          X_in.reshape(X_dims);
        }

        itensor filter_in = filter;
        auto filter_dims = CanonicalDims(filter_in.get_dims(), axis_w_);
        if (filter_in.get_dims() != filter_dims) {
          filter_in.reshape(filter_dims);
        }

        ideep::inner_product_backward_weights::compute(X_in, dY, *dfilter, *dbias);
        dfilter->to_default_format();

        /**
         * In mkl-dnn,weight gradient shape is determined by X_in,
         * so we should ensure that weight gradient shape is consistent with weight shape.
         */
        if (dfilter->get_dims() != filter.get_dims()) {
          dfilter->reshape(filter.get_dims());
        }

        if (OutputSize() > INPUT_GRAD) {
          ideep::inner_product_backward_data::compute(
              dY, filter_in, X.get_dims(), *Output(INPUT_GRAD));
        }

        return true;
        */
    }
}
