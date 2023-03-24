crate::ix!();
  
#[macro_export] macro_rules! declare_standard_arg {
    ($name:ident, $str:ident) => {
        /*
        
          static const char* Arg_##name; 
          OpSchema& Arg##name(const char* description);
        */
    }
}

declare_standard_arg!{IsTest, is_test}

/**
  | Helper function for creating simple
  | tensorproto with dimension and type
  |
  */
#[inline] pub fn create_tensor_shape<T>(dims: Vec<T>, dt: TensorProto_DataType) -> TensorShape {

    todo!();
    /*
        TensorShape ts;
      for (T d : dims) {
        ts.add_dims(d);
      }
      ts.set_data_type(dt);
      return ts;
    */
}


/// Helper function
#[inline] pub fn get_dims_vector(shape: &TensorShape) -> Vec<i64> {
    
    todo!();
    /*
        vector<int64_t> dims;
      for (auto d : shape.dims()) {
        dims.push_back(d);
      }
      return dims;
    */
}

/// Helper function
#[inline] pub fn n_elem_from_dim(
    x:   &TensorShape, 
    dim: Option<i32>) -> u64 {

    let dim: i32 = dim.unwrap_or(0);

    todo!();
    /*
        CAFFE_ENFORCE_GE(dim, 0, "Invalid maximum index specified");

      uint64_t nElem = 1;
      for (int i = dim; i < X.dims_size(); ++i) {
        nElem *= X.dims(i);
      }
      return nElem;
    */
}

/// Helper function
#[inline] pub fn n_elem_between_dim(x: &TensorShape, start: i32, stop: i32) -> u64 {
    
    todo!();
    /*
        CAFFE_ENFORCE_GE(start, 0, "Invalid maximum index specified");
      CAFFE_ENFORCE_LE(stop, X.dims_size(), "Invalid maximum index specified");

      uint64_t nElem = 1;
      for (int i = start; i < stop; ++i) {
        nElem *= X.dims(i);
      }
      return nElem;
    */
}

#[inline] pub fn pointwise_cost_inference<const OpsPerPoint: u64>(
    unused: &OperatorDef,
    inputs: &Vec<TensorShape>) -> OpSchemaCost 
{
    todo!();
    /*
        struct OpSchema::Cost c;
      const TensorShape X = inputs[0];
      uint64_t nElemX = nElemFromDim(X);
      uint64_t nElemRead = 0;
      for (size_t i = 0; i < inputs.size(); ++i) {
        nElemRead += nElemFromDim(inputs[i]);
      }

      c.flops = nElemX * OpsPerPoint;
      c.bytes_read = nElemRead * sizeof(X.data_type());
      c.bytes_written = nElemX * sizeof(X.data_type());
      return c;
    */
}

#[cfg(not(caffe2_no_operator_schema))]
#[macro_export] macro_rules! operator_schema {
    ($name:ident) => {
        todo!();
        /*
        
          C10_EXPORT void CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name(){}; 
          static OpSchema* C10_ANONYMOUS_VARIABLE(name) CAFFE2_UNUSED =     
              &OpSchemaRegistry::NewSchema(#name, __FILE__, __LINE__)
        */
    }
}


#[cfg(caffe2_no_operator_schema)]
#[macro_export] macro_rules! operator_schema {
    ($name:ident) => {
        todo!();
        /*
        
          C10_EXPORT void CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name(){}; 
          static OpSchema* C10_ANONYMOUS_VARIABLE(name) CAFFE2_UNUSED =     
              1 ? nullptr : &OpSchemaRegistry::NewSchema(#name, __FILE__, __LINE__)
        */
    }
}

#[cfg(caffe2_no_gradient_ops)]
#[macro_export] macro_rules! gradient_operator_schema {
    ($name:ident) => {
        todo!();
        /*
        
          C10_EXPORT void CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name(){}; 
          static OpSchema* C10_ANONYMOUS_VARIABLE(name) CAFFE2_UNUSED =     
              1 ? nullptr : &OpSchemaRegistry::NewSchema(#name, __FILE__, __LINE__)
        */
    }
}

#[cfg(not(caffe2_no_gradient_ops))]
#[macro_export] macro_rules! gradient_operator_schema {
    ($name:ident) => {
        todo!();
        /*
                OPERATOR_SCHEMA(name)
        */
    }
}

#[macro_export] macro_rules! define_standarg_arg{
    ($name:ident, $str:ident) => {
        /*
        TORCH_API const char* OpSchema::Arg_##name = #str;                 
        TORCH_API OpSchema& OpSchema::Arg##name(const char* description) { 
            return Arg(#str, description, true);                              
        }
        */
    }
}

define_standarg_arg!{IsTest, is_test}

