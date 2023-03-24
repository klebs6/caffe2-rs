crate::ix!();

/**
  | \class A class that does bound shape inference
  | given a C2 net. Depending on its type, each op
  | have a maximum shape that it accepts. We
  | define some initial bound for certain
  | dimension, for example max batch size or max
  | sequnce lookup size. And the inference will
  | first infer the input size and then propagates
  | the bound shape down the network. For now the
  | variable part (bound part) is the first
  | dimension of the shape, which usually
  | corresponds to the batch size or sequence
  | lookup size.
  */
pub struct BoundShapeInferencerBase {

    spec:                 BoundShapeSpec,
    shape_info:           ShapeInfoMap,
    extract_feature_len:  bool,
}

pub trait BoundShapeInferencerBaseTrait {

    /**
      | Initializes BoundShapeInferencer and
      | infers bound shape and type.
      |
      | info: shape information of some tensors,
      | e.g. shape information of external input
      | / output tensors;
      |
      | extract_feature_len:
      |
      | indicating whether to extract feature
      | length from SigridTransform and other
      | related operators.
      |
      | When enabled, extracted feature length
      | information will be used to infer tensor
      | shapes.
      */
    fn infer_bound_shape_and_type(&mut self, 
        net:                 &NetDef,
        info:                &ShapeInfoMap,
        ws:                  *mut Workspace,
        extract_feature_len: Option<bool>)  
    {
        let extract_feature_len: bool = extract_feature_len.unwrap_or(false);

        todo!(); /* */
    }
}

impl BoundShapeInferencerBase {
    
    pub fn new(spec: &BoundShapeSpec) -> Self {
    
        todo!();
        /*
            : spec_(spec) 
        CAFFE_ENFORCE_GE(spec_.max_batch_size, 0);
        CAFFE_ENFORCE_GE(spec_.max_seq_size, 0);
        */
    }
    
    #[inline] pub fn shape_info(&self) -> &ShapeInfoMap {
        
        todo!();
        /*
            return shape_info_;
        */
    }

    /// Print out all the shape info
    #[inline] pub fn print_shape_info(&self) -> String {
        
        todo!();
        /*
            std::stringstream ss;
        for (const auto& kv : shape_info_) {
          const auto& s = kv.second;
          ss << s.shape.name() << ": dim_type: " << s.getDimType() << ", dims: [";
          for (const auto d : s.shape.dims()) {
            ss << d << ", ";
          }
          ss << "], dtype: " << s.shape.data_type() << "\n";
        }
        return ss.str();
        */
    }
}

