crate::ix!();

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


