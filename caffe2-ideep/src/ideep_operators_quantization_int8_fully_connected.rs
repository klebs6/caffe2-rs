crate::ix!();

use crate::{
    IDType,
    ITensorDescriptor,
    IScale,
    ITensor,
    Workspace,
    OperatorDef,
    IDEEPOperator
};

pub struct IDEEPInt8FullyConnectedOp {
    //USE_IDEEP_DEF_ALIASES();
    //USE_IDEEP_OPERATOR_FUNCTIONS();
    base: IDEEPOperator,

    axis:                        usize, //{1};
    axis_w:                      usize, //{1};
    scale:                       f32,
    zero_point:                  i32,

    y_data_type:                 IDType,
    filter:                      ITensor,
    bias:                        ITensor,
    y:                           ITensor,
    y_scales:                    IScale,
    cached_X_descriptor:         ITensorDescriptor,
    cached_weights_descriptor:   ITensorDescriptor,
}

register_ideep_operator_with_engine!{
    Int8FC, 
    DNNLOWP, 
    IDEEPInt8FullyConnectedOp
}

input_tags!{
    IDEEPInt8FullyConnectedOp {
        Input,
        Filter,
        Bias
    }
}

output_tags!{
    IDEEPInt8FullyConnectedOp {
        Output
    }
}

impl IDEEPInt8FullyConnectedOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws),
            axis_(OperatorStorage::GetSingleArgument<int32_t>("axis", 1)),
            axis_w_(OperatorStorage::GetSingleArgument<int32_t>("axis_w", 1)),
            scale_(this->template GetSingleArgument<float>("Y_scale", 1.0)),
            zero_point_( this->template GetSingleArgument<int32_t>("Y_zero_point", 0)) 

        CAFFE_ENFORCE(zero_point_ == 128 || zero_point_ == 0);
        if (zero_point_ == 0) {
          Y_data_type_ = idtype::u8;
        } else {
          Y_data_type_ = idtype::s8;
        }
        Y_scales_ = ConvertScales({scale_});
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(INPUT);
        const auto& filter = Input(FILTER);
        auto* Y = Output(OUTPUT);

        itensor X_in = X;
        auto X_dims = CanonicalDims(X_in.get_dims(), axis_);
        if (X_in.get_dims() != X_dims) {
          X_in.reshape(X_dims);
        }

        if (cached_X_descriptor_ != X.get_descriptor()) {
          cached_X_descriptor_ = X.dup_descriptor();
          Y_.init({{X.get_dim(0), filter.get_dim(0)}, idtype::f32});
        }

        if (cached_weights_descriptor_ != filter.get_descriptor()) {
          cached_weights_descriptor_ = filter.dup_descriptor();
          CAFFE_ENFORCE(filter.get_data_type() == idtype::s8 && filter.has_scale());

          // INT8 FC is not supported so far.
          filter_ = filter.to_public();
          auto filter_dims = CanonicalDims(filter_.get_dims(), axis_w_);
          if (filter_.get_dims() != filter_dims) {
            filter_.reshape(filter_dims);
          }

          if (InputSize() > BIAS) {
            bias_ = Input(BIAS).to_public();
          }

          Y_.init({{X.get_dim(0), filter.get_dim(0)}, idtype::f32});
        }

        if (InputSize() > BIAS) {
          ideep::inner_product_forward::compute(
              X_in, filter_, bias_, Y_);
        } else {
          ideep::inner_product_forward::compute(X_in, filter_, Y_);
        }
        Y->init({Y_.get_dims(), Y_data_type_});
        Y->set_scale(Y_scales_);
        Y->feed_from(Y_);
        return true;
        */
    }
}
