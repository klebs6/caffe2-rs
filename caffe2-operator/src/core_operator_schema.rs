crate::ix!();


/**
  | @brief
  | 
  | A struct to store various cost information
  | about an operator such as FLOPs, total
  | memory use and parameters.
  |
  */
pub struct OpSchemaCost {

    /// Floating point operations.
    flops: u64,

    /// Total memory read.
    bytes_read: u64,

    /// Total memory written.
    bytes_written: u64,

    /// Memory read for parameters.
    params_bytes: u64,
}

/**
  | A const value returned by
  | 
  | OpSchema::CalculateOutput() if the
  | number of output cannot be determined.
  |
  */
pub const kCannotComputeNumOutputs: i32 = -1;

///------------------------------
pub struct SchemaArgument {
    name:        *const u8,
    description: *const u8,
    required:    bool,
}

impl SchemaArgument {
    
    pub fn new(
        name:        *const u8,
        description: *const u8,
        required:    bool) -> Self {
    
        todo!();
        /*
            : name_{name}, description_{description}, required_{required}
        */
    }
    
    #[inline] pub fn name(&self) -> *const u8 {
        
        todo!();
        /*
            return name_;
        */
    }
    
    #[inline] pub fn description(&self) -> *const u8 {
        
        todo!();
        /*
            return description_;
        */
    }
    
    #[inline] pub fn is_required(&self) -> bool {
        
        todo!();
        /*
            return required_;
        */
    }
}

/**
  | -----------
  | @brief
  | 
  | A class to record the schema of an op.
  | 
  | OpSchema records the common interface
  | of an op specified by its name. This is
  | optional for each operator implemented
  | in Caffe2 but is strongly recommended.
  | 
  | To register an OpSchema, one can use
  | the macro
  | 
  | OPERATOR_SCHEMA(name) and then append
  | the various functions in the class.
  | For example, for an op that takes in two
  | inputs, one output, and the first input
  | and output could be in-place, can be
  | written as
  | 
  | OPERATOR_SCHEMA(name)
  |  .NumInputs(2)
  |  .NumOutputs(1)
  |  .AllowInplace({{0, 0}});
  |
  */
pub struct OpSchema {

    type_:                         String,
    file:                          String,
    doc:                           String,
    onnx_schema:                   String,

    args:                          Vec<SchemaArgument>, // {};

    /**
      | In default, any in-place operation
      | is neither allowed nor enforced.
      |
      */
    inplace_allowed:               fn(_u0: i32, _u1: i32) -> bool, // = [](int, int) { return false; };
    inplace_enforced:              fn(_u0: i32, _u1: i32) -> bool, // = [](int, int) { return false; };
    tensor_inference_function:     TensorInferenceFunctionType,
    cost_inference_function:       Box<CostInferenceFunctionType>, // = nullptr;
    device_inference_function:     DeviceInferenceFunctionType,

    /**
      | default: 
      | [this](const std::vector<std::vector<int64_t>>& shapes)
      | { return SupplyDenseFillers(shapes); };
      |
      */
    filler_supplier:               fn(_u0: &Vec<Vec<i64>>) -> Vec<TensorFiller>,
    num_inputs_allowed:            fn(_u0: i32) -> bool, // = [](int) { return true; };
    num_outputs_allowed:           fn(_u0: i32) -> bool, // = [](int) { return true; };
    num_inputs_outputs_allowed:    fn(_u0: i32, _u1: i32) -> bool, // = [](int, int) { return true; };
    calculate_output:              fn(_u0: i32) -> i32,
    input_desc:                    Vec<(*const u8,*const u8)>, // {};
    output_desc:                   Vec<(*const u8,*const u8)>, // {};
    inputs_can_cross_devices:      bool, // = false;
    line:                          i32,  // = 0;
    min_input:                     i32,  // = 0;
    max_input:                     i32,  // = int::max;
    min_output:                    i32,  // = 0;
    max_output:                    i32,  // = int::max;
    private:                       bool, // = false;
}

impl Default for OpSchema {
    
    fn default() -> Self {
        todo!();
        /*
            : OpSchema("unknown", "unknown", 0
        */
    }
}

///-------------------------------------------------
pub type TensorInferenceFunctionType = fn(
    opdef:  &OperatorDef, 
    shapes: &Vec<TensorShape>) -> Vec<TensorShape>;

/**
  | @brief
  | 
  | Registers a function that takes in an
  | OperatorDef and a series of input shapes
  | and returns the total "cost" required
  | to run the operator via struct by value.
  |
  */
pub type CostInferenceFunctionType = fn(
    opdef:  &OperatorDef, 
    shapes: &Vec<TensorShape>) -> OpSchemaCost;

pub type DeviceInferenceFunctionType = fn(
    _u0: OperatorDef) -> (Vec<DeviceOption>,Vec<DeviceOption>);

impl OpSchema {
    
    #[inline] pub fn has_cost_inference_function(&self) -> bool {
        
        todo!();
        /*
            return !!cost_inference_function_;
        */
    }
    
    #[inline] pub fn infer_cost(&self, def: &OperatorDef, input_tensor_shape: &Vec<TensorShape>) -> OpSchemaCost {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            cost_inference_function_, "Cost inference function not defined.");
        return (*cost_inference_function_)(def, input_tensor_shape);
        */
    }
    
    #[inline] pub fn input_desc(&self) -> &Vec<(*const u8,*const u8)> {
        
        todo!();
        /*
            return input_desc_;
        */
    }
    
    #[inline] pub fn output_desc(&self) -> &Vec<(*const u8, *const u8)> {
        
        todo!();
        /*
            return output_desc_;
        */
    }
}
  
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
  | @brief
  | 
  | A registry to hold all the operator schemas.
  | 
  | OpSchemaRegistry should not need to
  | be instantiated.
  |
  */
pub struct OpSchemaRegistry {

}

impl OpSchemaRegistry {
    
    #[inline] pub fn new_schema<'a>(
        key:  &String,
        file: &String,
        line: i32) -> &'a mut OpSchema 
    {
        todo!();
        /*
            auto& m = map();
        auto it = m.find(key);
        if (it != m.end()) {
          const auto& schema = it->second;
          std::ios_base::Init init;
          std::cerr << "Trying to register schema with name " << key
                    << " from file " << file << " line " << line
                    << ", but it is already registered from file " << schema.file()
                    << " line " << schema.line();
          abort();
        }
        m.emplace(std::make_pair(key, OpSchema(key, file, line)));
        return m[key];
        */
    }
    
    #[inline] pub fn schema(key: &String) -> *const OpSchema {
        
        todo!();
        /*
            auto& m = map();
        auto it = m.find(key);
        if (it != m.end()) {
          return &it->second;
        } else {
          return nullptr;
        }
        */
    }
}

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

/**
  | Helper function for infer op inputs
  | and outputs device information.
  |
  */
#[inline] pub fn infer_op_input_output_device(op: &OperatorDef) 
    -> (Vec<DeviceOption>,Vec<DeviceOption>) 
{
    
    todo!();
    /*
        auto op_schema = OpSchemaRegistry::Schema(op.type());
      if (op_schema) {
        // op_schema found
        return op_schema->InferDevice(op);

      } else {
        // No schema for op.type registered
        auto temp_schema = OpSchema();
        return temp_schema.InferDevice(op);
      }
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

impl OpSchema {

    /**
      | Functions to set the property of the
      | operator schemas.
      |
      | Sets the number of inputs, either a fixed
      | number or a min and a max.
      */
    
    /**
      | -----------
      | @brief
      | 
      | Input is checked with a specified function.
      |
      */
    #[inline] pub fn num_inputs_with_function(&mut self, func: fn(_u0: i32) -> bool) -> &mut OpSchema {
        
        todo!();
        /*
            num_inputs_allowed_ = func;
      return *this;
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Output is checked with a specified function.
      |
      */
    #[inline] pub fn num_outputs_from_function(&mut self, func: fn(_u0: i32) -> bool) -> &mut OpSchema {
        
        todo!();
        /*
            num_outputs_allowed_ = func;
      return *this;
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Relationship between inputs and outputs
      | is checked with a specified function.
      |
      */
    #[inline] pub fn num_inputs_outputs(&mut self, func: fn(_u0: i32, _u1: i32) -> bool) -> &mut OpSchema {
        
        todo!();
        /*
            num_inputs_outputs_allowed_ = func;
      return *this;
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Set the output calculator to a user-defined
      | function.
      | 
      | Set the function that can calculate
      | the number of output based on the number
      | of input. Use only one function in the
      | set below.
      |
      */
    #[inline] pub fn output_calculator(&mut self, calc: fn(_u0: i32) -> i32) -> &mut OpSchema {
        
        todo!();
        /*
            calculate_output_ = calc;
      return *this;
        */
    }
    
    /**
      | Sets the rule to allow optional in-place
      | operation.
      |
      */
    #[inline] pub fn allow_inplace_with_functor(
        &mut self, 
        inplace: fn(_u0: i32, _u1: i32) -> bool) -> &mut OpSchema {
        
        todo!();
        /*
            inplace_allowed_ = inplace;
      return *this;
        */
    }

    #[inline] pub fn allow_inplace(
        &mut self, 
        inplace: HashSet<(i32,i32)>) -> &mut OpSchema {
        
        todo!();
        /*
            return AllowInplace(
          [inplace](int in, int out)->bool {
            return inplace.count(std::make_pair(in, out));
          });
        */
    }

    #[inline] pub fn allow_one_to_one_inplace(&mut self) -> &mut OpSchema {
        
        todo!();
        /*
            return AllowInplace([](int in, int out) { return in == out; });
        */
    }
    
    ///-----------------------------------------
    /// Sets the rule to enforce in-place operation.
    #[inline] pub fn enforce_inplace_with_function(
        &mut self, 
        inplace: fn(_u0: i32, _u1: i32) -> bool) -> &mut OpSchema {
        
        todo!();
        /*
            inplace_enforced_ = inplace;
      return *this;
        */
    }

    #[inline] pub fn enforce_inplace(&mut self, inplace: HashSet<(i32,i32)>) -> &mut OpSchema {
        
        todo!();
        /*
            return EnforceInplace(
          [inplace](int in, int out)->bool {
            return inplace.count(std::make_pair(in, out));
          });
        */
    }

    #[inline] pub fn enforce_one_to_one_inplace(&mut self) -> &mut OpSchema {
        
        todo!();
        /*
            return EnforceInplace([](int in, int out) { return in == out; });
        */
    }

    /**
      | Calls the passed function with `this` as
      | an argument. Useful for adding docs for
      | templated/macro ops.
      */
    #[inline] pub fn fill_using(&mut self, populator: fn(_u0: &mut OpSchema) -> c_void) -> &mut OpSchema {
        
        todo!();
        /*
            if (populator) {
        populator(*this);
      }
      return *this;
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Input could be one of the values specified
      | in allowed_input_nums.
      |
      */
    #[inline] pub fn num_inputs_from_set(&mut self, 
        allowed_input_nums: HashSet<i32>) -> &mut OpSchema 
    {
        todo!();
        /*
            return NumInputs(
          [allowed_input_nums](int n)->bool {
            return allowed_input_nums.count(n);
          });
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Output could be in range [min, max],
      | inclusive.
      |
      */
    #[inline] pub fn num_outputs_bounded(
        &mut self, 
        min: i32,
        max: i32) -> &mut OpSchema 
    {
        todo!();
        /*
            min_output_ = min;
      max_output_ = max;
      return *this;
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | A single output.
      | 
      | Sets the number of outputs, either a
      | fixed number, a min and a max, or a function
      | that takes in the input number and produces
      | an output number.
      | 
      | Use only one function in the set below.
      |
      */
    #[inline] pub fn num_outputs_fixed(&mut self, n: i32) -> &mut OpSchema {
        
        todo!();
        /*
            return NumOutputs(n, n);
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Output could be one of the values specified
      | in allowed_output_nums.
      |
      */
    #[inline] pub fn num_outputs_from_set(&mut self, allowed_output_nums: HashSet<i32>) -> &mut OpSchema {
        
        todo!();
        /*
            return NumOutputs(
          [allowed_output_nums](int n)->bool {
            return allowed_output_nums.count(n);
          });
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Set the number of outputs to be the same
      | as the number of inputs.
      |
      */
    #[inline] pub fn same_number_of_output(&mut self) -> &mut OpSchema {
        
        todo!();
        /*
            return OutputCalculator([](int n)->int { return n; } );
        */
    }
    
    #[inline] pub fn input(
        &mut self, 
        n:              i32,
        name:           *const u8,
        description:    *const u8) -> &mut OpSchema 
    {
        todo!();
        /*
            if (input_desc_.size() <= (unsigned)n) {
        input_desc_.resize(n + 1);
      }
      input_desc_[n] = std::make_pair(name, description);
      return *this;
        */
    }
    
    #[inline] pub fn output(
        &mut self, 
        n:              i32,
        name:           *const u8,
        description:    *const u8) -> &mut OpSchema 
    {
        todo!();
        /*
            if (output_desc_.size() <= (unsigned)n) {
        output_desc_.resize(n + 1);
      }
      output_desc_[n] = std::make_pair(name, description);
      return *this;
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | A function to allow one to get the number
      | of outputs based on the number of inputs,
      | if this schema supports it.
      |
      */
    #[inline] pub fn calculate_output(&self, num_input: i32) -> i32 {
        
        todo!();
        /*
            if (min_output_ == max_output_) {
        return min_output_;
      } else if (calculate_output_) {
        return calculate_output_(num_input);
      } else {
        return kCannotComputeNumOutputs;
      }
        */
    }
    
    #[inline] pub fn sparse_lengths_filler_helper(
        &mut self, 
        shapes:         &Vec<Vec<i64>>,
        value_index:    usize,
        length_index:   usize,
        fillers:        *mut Vec<TensorFiller>)  
    {
        todo!();
        /*
            CAFFE_ENFORCE_EQ(shapes[length_index].size(), 1);
      // filler.h: SparseLengths->FixedSum will select FD_FIXEDSUM distribution
      (*fillers)[length_index].SparseLengths(shapes[value_index].front());
        */
    }
    
    #[inline] pub fn sparse_weights_filler_helper(
        &mut self, 
        shapes:         &Vec<Vec<i64>>,
        weight_index:   usize,
        fillers:        *mut Vec<TensorFiller>)  
    {
        
        todo!();
        /*
            (*fillers)[weight_index]
          .Min(0)
          .Max(shapes[weight_index].front())
          .Dist(FD_UNIFORM);
        */
    }
    
    #[inline] pub fn sparse_segments_filler_helper(
        &mut self, 
        shapes:         &Vec<Vec<i64>>,
        value_index:    usize,
        segment_index:  usize,
        fillers:        *mut Vec<TensorFiller>)  
    {
        todo!();
        /*
            CAFFE_ENFORCE_EQ(shapes[segment_index].size(), 1);
      // filler.h SparseSegments will select FD_UNIFORM or FD_SYNTHETIC distribution
      (*fillers)[value_index]
          .Min(0)
          .Max(shapes[value_index].front() * 2)
          .Dist(FD_UNIFORM);
      (*fillers)[segment_index].SparseSegments(shapes[value_index].front() - 1);
        */
    }
    
    /**
      | The helper is build sparse input with
      | values, keys, weights and lengths;
      |
      | e.g.:
      | values  = [1, 2, 3, 2, 4, 6, 7, 3, 6]
      | keys    = [0, 1, 4, 0, 1, 2, 5, 1, 2]
      |            \_____/  \________/  \__/
      | lengths =    [3,        4,       2]
      */
    #[inline] pub fn value_key_length_input_fillers(
        &mut self, 
        value_index:    usize,
        key_index:      usize,
        length_index:   usize) -> &mut OpSchema 
    {
        todo!();
        /*
            filler_supplier_ = [this, value_index, key_index, length_index](
                             const std::vector<std::vector<int64_t>>& shapes) {
        auto fillers = SupplyDenseFillers(shapes);
        // fill in the length (value_index is used to get the correct shape)
        SparseLengthsFillerHelper(shapes, key_index, length_index, &fillers);
        // fill in the keys (value_index is used to get the correct shape)
        SparseSegmentsFillerHelper(shapes, value_index, key_index, &fillers);
        return fillers;
      };
      return *this;
        */
    }

    /**
      | The helper is build sparse input with
      | values, keys, weights and lengths;
      |
      | e.g.:
      | values  = [1, 2, 3, 2, 4, 6, 7, 3, 6]
      | keys    = [0, 1, 4, 0, 1, 2, 5, 1, 2]
      | weights = [1, 1, 1, 0, 2, 2, 2, 1, 2]
      |            \_____/  \________/  \__/
      | lengths =    [3,        4,       2]
      */
    #[inline] pub fn weighted_value_key_length_input_fillers(
        &mut self, 
        value_index:    usize,
        key_index:      usize,
        length_index:   usize,
        weight_index:   usize) -> &mut OpSchema 
    {
        todo!();
        /*
            filler_supplier_ = [this, value_index, key_index, length_index, weight_index](
                             const std::vector<std::vector<int64_t>>& shapes) {
        auto fillers = SupplyDenseFillers(shapes);
        // fill in the length (value_index is used to get the correct shape)
        SparseLengthsFillerHelper(shapes, key_index, length_index, &fillers);
        // fill in the keys (value_index is used to get the correct shape)
        SparseSegmentsFillerHelper(shapes, value_index, key_index, &fillers);
        // fill in the weights
        SparseWeightsFillerHelper(shapes, weight_index, &fillers);
        return fillers;
      };
      return *this;
        */
    }

    /**
      | The helper is build sparse input with
      | values and lengths; e.g.:
      |
      | values  = [1, 2, 3, 2, 4, 6, 7, 3, 6]
      |            \_____/  \________/  \__/
      | lengths =    [3,        4,       2]
      */
    #[inline] pub fn value_length_input_fillers(
        &mut self, 
        value_index:  usize,
        length_index: usize) -> &mut OpSchema 
    {
        todo!();
        /*
            filler_supplier_ = [this, value_index, length_index](
                             const std::vector<std::vector<int64_t>>& shapes) {
        auto fillers = SupplyDenseFillers(shapes);
        // fill in the length (value_index is used to get the correct shape)
        SparseLengthsFillerHelper(shapes, value_index, length_index, &fillers);
        return fillers;
      };
      return *this;
        */
    }
    
    #[inline] pub fn disallow_input_fillers(&mut self) -> &mut OpSchema {
        
        todo!();
        /*
            filler_supplier_ =
          [this](const std::vector<std::vector<int64_t>>& /* unused */) {
            throw std::invalid_argument(type_ + " does not have input fillers");
            return std::vector<TensorFiller>();
          };
      return *this;
        */
    }
    
    #[inline] pub fn input_fillers(&self, 
        shapes: &Vec<Vec<i64>>) -> Vec<TensorFiller> 
    {
        
        todo!();
        /*
            return filler_supplier_(shapes);
        */
    }
    
    #[inline] pub fn supply_dense_fillers(&mut self, 
        shapes: &Vec<Vec<i64>>) -> Vec<TensorFiller> 
    {
        
        todo!();
        /*
            std::vector<TensorFiller> fillers;
      for (const auto& shape : shapes) {
        fillers.emplace_back(shape);
      }
      return fillers;
        */
    }
    
    pub fn new(ty: &String, file: &String, line: i32) -> Self {
        todo!();
        /*
            : type_(type), 
        file_(file), 
        line_(line), 
        tensor_inference_function_(
          [](const OperatorDef& def, const vector<TensorShape>&) {
            vector<TensorShape> out;
            for (int i = 0; i < def.output_size(); i++) {
              TensorShape ts;
              ts.set_unknown_shape(true);
              out.push_back(ts);
            }
            return out;
          }), 
        device_inference_function_(
          [](const OperatorDef& def) {
            auto op_device =
                def.has_device_option() ? def.device_option() : DeviceOption();
            vector<DeviceOption> in_dev(def.input_size(), op_device);
            vector<DeviceOption> out_dev(def.output_size(), op_device);
            return std::make_pair(in_dev, out_dev);
          })
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Returns the file that the op schema is
      | registered from.
      |
      */
    #[inline] pub fn file(&self) -> &String {
        
        todo!();
        /*
            return file_;
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Returns the line in file that the op schema
      | is registered from.
      |
      */
    #[inline] pub fn line(&self) -> i32 {
        
        todo!();
        /*
            return line_;
        */
    }

    /**
      | -----------
      | @brief
      | 
      | Returns the docstring of the op schema.
      |
      */
    #[inline] pub fn doc(&self) -> *const u8 {
        
        todo!();
        /*
            return doc_.empty() ? nullptr : doc_.c_str();
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Shortcut to InheritOnnxSchema(type_)
      |
      */
    #[inline] pub fn inherit_onnx_schema_default(&mut self) -> &mut OpSchema {
        
        todo!();
        /*
            return InheritOnnxSchema(type_);
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | A function to allow one to infer the type
      | and shape from the op schema.
      |
      */
    #[inline] pub fn infer_tensor(
        &self, 
        def:              &OperatorDef,
        input_type_shape: &Vec<TensorShape>) -> Vec<TensorShape> 
    {
        todo!();
        /*
            CAFFE_ENFORCE(
            Verify(def),
            "(InferTensor) Operator def did not pass schema checking: ",
            ProtoDebugString(def));
        return tensor_inference_function_(def, input_type_shape);
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Verifies if an operator definition
      | protobuf matches the pattern specified
      | in the schema.
      |
      */
    #[inline] pub fn verify(&self, def: &OperatorDef) -> bool {
        
        todo!();
        /*
            // Check the number of inputs.
      if (def.input_size() < min_input_ || def.input_size() > max_input_) {
        LOG(ERROR) << "Input size " << def.input_size()
                        << " not in range [min=" << min_input_ << ", max="
                        << max_input_ << "].";
        return false;
      }
      if (!num_inputs_allowed_(def.input_size())) {
        LOG(ERROR) << "Input size " << def.input_size()
                        << " not in allowed input sizes.";
        return false;
      }
      // Check the number of outputs.
      if (def.output_size() < min_output_ || def.output_size() > max_output_) {
        LOG(ERROR) << "Output size " << def.output_size()
                        << " not in range [min=" << min_output_ << ", max="
                        << max_output_ << "].";
        return false;
      }
      if (!num_outputs_allowed_(def.output_size())) {
        LOG(ERROR) << "Output size " << def.output_size()
                        << " not in allowed output sizes.";
        return false;
      }
      if (!num_inputs_outputs_allowed_(def.input_size(), def.output_size())) {
        LOG(ERROR) << "Combination of input size " << def.input_size()
                   << "and output size " << def.output_size() << " not in allowed.";
        return false;
      }
      // If the number of outputs can be calculated, check if the number matches.
      if (calculate_output_) {
        int expected_nout = calculate_output_(def.input_size());
        if (expected_nout != kCannotComputeNumOutputs &&
            def.output_size() != expected_nout) {
          LOG(ERROR) << "Output size " << def.output_size()
                          << " not matching expected output size, which is "
                          << expected_nout;
          return false;
        }
      }

      // Check in-place settings.
      for (int in_idx = 0; in_idx < def.input_size(); ++in_idx) {
        for (int out_idx = 0; out_idx < def.output_size(); ++out_idx) {
          // If an input is the same as an output but in-place is not opt-in
          // either as allowed or enforced, we will fail the verification.
          if (def.input(in_idx) == def.output(out_idx) &&
              (!inplace_allowed_(in_idx, out_idx)
              && !inplace_enforced_(in_idx, out_idx))) {
            LOG(ERROR) << "Input index " << in_idx << " and output idx " << out_idx
                       << " (" << def.input(in_idx) << ")"
                       << " are set to be in-place but this is actually not "
                       << "supported by op " << def.type();
            return false;
          }
          if (def.input(in_idx) != def.output(out_idx) &&
              inplace_enforced_(in_idx, out_idx)) {
            LOG(ERROR) << "Input index " << in_idx << " (" << def.input(in_idx) << ")"
                       << " and output idx " << out_idx
                       << " (" << def.output(in_idx) << ")"
                       << " are not in-place but should be as required by op "
                       << def.type();
            return false;
          }
        }
      }

      std::set<std::string> present_args{};
      for (const auto& arg : def.arg()) {
        present_args.insert(arg.name());
      }

      for (const auto& arg : args()) {
        if (arg.is_required() &&
            present_args.find(arg.name()) == present_args.end()) {
          LOG(ERROR) << "SchemaArgument '" << arg.name() << "' is required for Operator '"
                     << def.type() << "'.";
          return false;
        }
      }

      // Phew. All verifications passed.
      return true;
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Input could be in range [min, max], inclusive.
      |
      */
    #[inline] pub fn num_inputs_bounded(
        &mut self, 
        min: i32,
        max: i32) -> &mut OpSchema 
    {
        todo!();
        /*
            min_input_ = min;
      max_input_ = max;
      return *this;
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | A single input.
      |
      */
    #[inline] pub fn num_inputs_fixed(&mut self, n: i32) -> &mut OpSchema {
        
        todo!();
        /*
            return NumInputs(n, n);
        */
    }

    #[inline] pub fn onnx_schema(&self) -> &String {
        
        todo!();
        /*
            return onnx_schema_;
        */
    }
    
    #[inline] pub fn min_input(&self) -> i32 {
        
        todo!();
        /*
            return min_input_;
        */
    }
    
    #[inline] pub fn max_input(&self) -> i32 {
        
        todo!();
        /*
            return max_input_;
        */
    }
    
    #[inline] pub fn min_output(&self) -> i32 {
        
        todo!();
        /*
            return min_output_;
        */
    }
    
    #[inline] pub fn max_output(&self) -> i32 {
        
        todo!();
        /*
            return max_output_;
        */
    }
    
    #[inline] pub fn num_inputs_allowed(&self, x: i32) -> bool {
        
        todo!();
        /*
            return num_inputs_allowed_(x);
        */
    }
    
    #[inline] pub fn num_outputs_allowed(&self, x: i32) -> bool {
        
        todo!();
        /*
            return num_outputs_allowed_(x);
        */
    }
    
    #[inline] pub fn num_inputs_outputs_allowed(&self, 
        x: i32,
        y: i32) -> bool 
    {
        todo!();
        /*
            return num_inputs_outputs_allowed_(x, y);
        */
    }
    
    #[inline] pub fn inf(&self) -> i32 {
        
        todo!();
        /*
            return int::max;
        */
    }
    
    #[inline] pub fn inplace_enforced(&self, 
        x: i32,
        y: i32) -> bool 
    {
        todo!();
        /*
            return inplace_enforced_(x, y);
        */
    }
    
    #[inline] pub fn args(&self) -> &Vec<SchemaArgument> {
        
        todo!();
        /*
            return args_;
        */
    }
    
    #[inline] pub fn private_op(&mut self) -> bool {
        
        todo!();
        /*
            return private_;
        */
    }
    
    #[inline] pub fn can_inputs_cross_devices(&self) -> bool {
        
        todo!();
        /*
            return inputs_can_cross_devices_;
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Infer required device location of an
      | op's inputs and outputs
      |
      */
    #[inline] pub fn infer_device(&self, def: &OperatorDef) -> (Vec<DeviceOption>,Vec<DeviceOption>) {
        
        todo!();
        /*
            return device_inference_function_(def);
        */
    }
    
    /// Remove from documentation
    #[inline] pub fn private(&mut self) -> &mut OpSchema {
        
        todo!();
        /*
            private_ = true;
      return *this;
        */
    }
    
    /// This op can pass data across devices
    #[inline] pub fn inputs_can_cross_devices(&mut self) -> &mut OpSchema {
        
        todo!();
        /*
            inputs_can_cross_devices_ = true;
      return *this;
        */
    }

    /**
      | Functions to deal with type and shape
      | inference. Basically, this registers
      | a function that takes in an OperatorDef
      | and a series of input type and shape specified
      | by
      | 
      | TensorProto objects (whose data fields
      | are empty), and produces a series of
      | output type and shape.
      |
      */
    
    /**
      | -----------
      | @brief
      | 
      | Sets the tensor inference function,
      | which is a std::function object defined
      | in operator_schema.h.
      |
      */
    #[inline] pub fn tensor_inference_function(&mut self, 
        function: TensorInferenceFunctionType) -> &mut OpSchema 
    {
        todo!();
        /*
            tensor_inference_function_ = function;
      return *this;
        */
    }
    
    /**
      | A wrapper that makes an infer tensor
      | function to return unknown shape for
      | all outputs if any one of the inputs has
      | unknown shape
      |
      */
    #[inline] pub fn needs_all_input_shapes(
        f: TensorInferenceFunctionType) -> TensorInferenceFunctionType {
        
        todo!();
        /*
            return [f](const OperatorDef& def, const vector<TensorShape>& in) {
        for (const auto& in_ts : in) {
          if (in_ts.unknown_shape()) {
            vector<TensorShape> out(def.output().size());
            for (auto& out_ts : out) {
              out_ts.set_unknown_shape(true);
            }
            return out;
          }
        }
        return f(def, in);
      };
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Sets the corresponding onnx schema
      | name
      |
      */
    #[inline] pub fn inherit_onnx_schema(&mut self, onnx_schema_name: &String) -> &mut OpSchema {
        
        todo!();
        /*
            onnx_schema_ = onnx_schema_name;
      return *this;
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Sets the tensor inference function
      | to produce the same output as the input.
      |
      */
    #[inline] pub fn identical_type_and_shape(&mut self) -> &mut OpSchema {
        
        todo!();
        /*
            return TensorInferenceFunction(
          [](const OperatorDef&, const vector<TensorShape>& input_types) {
            return vector<TensorShape>(input_types);
          });
        */
    }
    
    #[inline] pub fn identical_type_and_shape_of_input(&mut self, idx: i32) -> &mut OpSchema {
        
        todo!();
        /*
            return TensorInferenceFunction(
          [idx](const OperatorDef&, const vector<TensorShape>& input_types) {
            vector<TensorShape> out(1);
            out[0] = input_types[idx];
            return out;
          });
        */
    }
    
    #[inline] pub fn identical_type_and_shape_of_multiple_inputs(&mut self, 
        indices: &Vec<i32>) -> &mut OpSchema 
    {
        todo!();
        /*
            return TensorInferenceFunction(
          [indices](const OperatorDef&, const vector<TensorShape>& input_types) {
            vector<TensorShape> out(indices.size());
            for (const auto i : c10::irange(indices.size())) {
              out[i] = input_types[indices.at(i)];
            }
            return out;
          });
        */
    }
    
    #[inline] pub fn identical_type_and_shape_of_input_dim(&mut self, 
        idx: i32,
        dim: i32) -> &mut OpSchema 
    {
        todo!();
        /*
            return TensorInferenceFunction(
          [idx, dim](const OperatorDef&, const vector<TensorShape>& input_types) {
            vector<TensorShape> out(1);
            out[0].add_dims(input_types[idx].dims(dim));
            out[0].set_data_type(input_types[idx].data_type());
            return out;
          });
        */
    }
    
    #[inline] pub fn scalar_type(&mut self, dt: TensorProto_DataType) -> &mut OpSchema {
        
        todo!();
        /*
            return TensorInferenceFunction(
          [dt](const OperatorDef& def, const vector<TensorShape>& /*input_types*/) {
            TensorShape shape;
            shape.set_data_type(dt);
            vector<TensorShape> out(def.output_size(), shape);
            return out;
          });
        */
    }

    /**
      | -----------
      | @brief
      | 
      | Register the Cost inference function.
      |
      */
    #[inline] pub fn cost_inference_function(
        &mut self, 
        function: CostInferenceFunctionType) -> &mut OpSchema 
    {
        todo!();
        /*
            cost_inference_function_ =
          std::make_unique<CostInferenceFunctionType>(function);
      return *this;
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Returns the required device location
      | of inputs and outputs.
      |
      */
    #[inline] pub fn device_inference_function(&mut self, 
        function: DeviceInferenceFunctionType) -> &mut OpSchema 
    {
        todo!();
        /*
            device_inference_function_ = function;
      return *this;
        */
    }
    
    /**
      | Functions to do documentation for the
      | operator schema.
      |
      */
    #[inline] pub fn set_doc(&mut self, doc: &String) -> &mut OpSchema {
        
        todo!();
        /*
            doc_ = doc;
      return *this;
        */
    }
    
    #[inline] pub fn arg(&mut self, 
        name:        *const u8,
        description: *const u8,
        required:    Option<bool>) -> &mut OpSchema 
    {
        let required = required.unwrap_or(false);
        todo!();
        /*
            args_.push_back(SchemaArgument(name, description, required));
      return *this;
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

impl fmt::Display for OpSchema {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            if (!schema.args().empty()) {
        out << "SchemaArguments:" << std::endl;
        for (const auto& arg : schema.args()) {
          out << "  " << arg.name() << " : " << arg.description() << std::endl;
        }
      }
      if (schema.max_input_ > 0) {
        out << "Inputs:" << std::endl;
        if (!schema.input_desc_.empty()) {
          for (size_t i = 0; i < schema.input_desc_.size(); ++i) {
            const auto& p = schema.input_desc_[i];
            out << "  " << i << ", " << (p.first ? p.first : "(unnamed)") << " : "
                << (p.second ? p.second : "(no doc)") << std::endl;
          }
        } else {
          out << "  (no explicit description available)" << std::endl;
        }
      }
      if (schema.max_output_ > 0) {
        out << "Outputs:" << std::endl;
        if (!schema.output_desc_.empty()) {
          for (size_t i = 0; i < schema.output_desc_.size(); ++i) {
            const auto& p = schema.output_desc_[i];
            out << "  " << i << ", " << (p.first ? p.first : "(unnamed)") << " : "
                << (p.second ? p.second : "(no doc)") << std::endl;
          }
        } else {
          out << "  (no explicit description available)" << std::endl;
        }
      }
      out << std::endl;
      if (schema.doc()) {
        out << schema.doc();
      } else {
        out << "(no documentation yet)" << std::endl;
      }
      out << std::endl;
      if (schema.line_) {
        out << "Defined at " << schema.file_ << ":" << schema.line_ << std::endl;
      }
      return out;
        */
    }
}

impl OpSchemaRegistry {
    
    /**
      | -----------
      | @brief
      | 
      | Returns the underlying string to
      | 
      | OpSchema map.
      | 
      | You should not manually manipulate
      | the map object returned. Instead, use
      | the macros defined such as OPERATOR_SCHEMA
      | to register your operator schema.
      | 
      | We wrap it inside a function to avoid
      | the static initialization order fiasco.
      |
      */
    #[inline] pub fn map<'a>(&'a mut self) -> &'a mut HashMap<String,OpSchema> {
        
        todo!();
        /*
            static CaffeMap<string, OpSchema> map;
      return map;
        */
    }
}
