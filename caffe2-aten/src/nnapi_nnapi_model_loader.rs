crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/nnapi/nnapi_model_loader.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/nnapi/nnapi_model_loader.cpp]

#[macro_export] macro_rules! nnapi_check {
    ($res:ident) => {
        /*
                CAFFE_ENFORCE(res == ANEURALNETWORKS_NO_ERROR, "NNAPI returned error: ", res)
        */
    }
}

/**
  | Serialized format for NNAPI models.
  | It is basically just a list arguments
  | for calls to be made to NNAPI.
  |
  */
pub enum SourceType {
  SOURCE_IMMEDIATE       = 0,
  SOURCE_NUMBERED_BUFFER = 2,
  SOURCE_NUMBERED_MEMORY = 3,
}

pub struct SerializedOperand {
    ty:              i32,
    dimension_count: u32,
    scale:           f32,
    zero_point:      i32,
}

pub struct SerializedValue {
    index:         i32,
    source_type:   i32,
    source_length: u32,
}

pub struct SerializedOperation {
    operation_type: i32,
    input_count:    u32,
    output_count:   u32,
}

pub struct SerializedModel {
    version:         i32,
    operand_count:   i32,
    value_count:     i32,
    operation_count: i32,
    input_count:     i32,
    output_count:    i32,

    // SerializedOperand operands[operand_count];
    // SerializedValue values[value_count];
    // SerializedOperation operations[operation_count];
    // u32 operand_dimensions[sum(dimension_count)]
    // u32 value_data[sum(source_length+pad)/4]
    // u32 operation_args[sum(input_count + output_count)]
    // u32 model_inputs[input_count]
    // u32 model_outputs[output_count]
}

/**
  | Get the physically stored size of a value.
  | All values are padded out to a multiple
  | of 4 bytes to ensure the next value is
  | 4-byte aligned.
  |
  */
pub fn value_physical_size(len: u32) -> u32 {
    
    todo!();
        /*
            u32 phys = len;
      if (len % 4 == 0) {
        return len;
      }
      return len + 4 - (phys % 4);
        */
}

pub fn load_nnapi_model(
    nnapi:              *mut NnapiWrapper,
    model:              *mut ANeuralNetworksModel,
    serialized_model:   *const c_void,
    model_length:       i64,
    num_buffers:        usize,
    buffer_ptrs:        *const *const c_void,
    buffer_sizes:       *mut i32,
    num_memories:       usize,
    memories:           *mut *mut ANeuralNetworksMemory,
    memory_sizes:       *mut i32,
    out_input_count:    *mut i32,
    out_output_count:   *mut i32,
    out_bytes_consumed: *mut usize) -> i32 {

    todo!();
        /*
            i64 required_size = 0;
      const u8* next_pointer = (const u8*)serialized_model;
      const u8* end_of_buf = (const u8*)serialized_model + model_length;

      required_size += sizeof(SerializedModel);
      CAFFE_ENFORCE(model_length >= required_size, "Model is too small.  usize = ", model_length);
      const SerializedModel* ser_model = (SerializedModel*)next_pointer;
      next_pointer = (u8*)serialized_model + required_size;
      CAFFE_ENFORCE(next_pointer <= end_of_buf);

      CAFFE_ENFORCE(ser_model->version == 1);
      // Keep these small to avoid integer overflow.
      CAFFE_ENFORCE(ser_model->operand_count    < (1 << 24));
      CAFFE_ENFORCE(ser_model->value_count      < (1 << 24));
      CAFFE_ENFORCE(ser_model->operation_count  < (1 << 24));
      CAFFE_ENFORCE(ser_model->input_count      < (1 << 24));
      CAFFE_ENFORCE(ser_model->output_count     < (1 << 24));

      required_size += sizeof(SerializedOperand) * ser_model->operand_count;
      CAFFE_ENFORCE(model_length >= required_size, "Model is too small.  usize = ", model_length);
      const SerializedOperand* operands = (const SerializedOperand*)next_pointer;
      next_pointer = (u8*)serialized_model + required_size;
      CAFFE_ENFORCE(next_pointer <= end_of_buf);

      required_size += sizeof(SerializedValue) * ser_model->value_count;
      CAFFE_ENFORCE(model_length >= required_size, "Model is too small.  usize = ", model_length);
      const SerializedValue* values = (const SerializedValue*)next_pointer;
      next_pointer = (u8*)serialized_model + required_size;
      CAFFE_ENFORCE(next_pointer <= end_of_buf);

      required_size += sizeof(SerializedOperation) * ser_model->operation_count;
      CAFFE_ENFORCE(model_length >= required_size, "Model is too small.  usize = ", model_length);
      const SerializedOperation* operations = (const SerializedOperation*)next_pointer;
      next_pointer = (u8*)serialized_model + required_size;
      CAFFE_ENFORCE(next_pointer <= end_of_buf);

      for (int i = 0; i < ser_model->operand_count; i++) {
        required_size += 4 * operands[i].dimension_count;
      }

      for (int i = 0; i < ser_model->value_count; i++) {
        required_size += value_physical_size(values[i].source_length);
      }

      for (int i = 0; i < ser_model->operation_count; i++) {
        required_size += 4 * (operations[i].input_count + operations[i].output_count);
      }

      required_size += 4 * (ser_model->input_count + ser_model->output_count);

      CAFFE_ENFORCE(model_length >= required_size, "Model is too small.  usize = ", model_length);
      CAFFE_ENFORCE(next_pointer <= end_of_buf);

      for (int i = 0; i < ser_model->operand_count; i++) {
        ANeuralNetworksOperandType operand;
        operand.type = operands[i].type;
        operand.scale = operands[i].scale;
        operand.zeroPoint = operands[i].zero_point;
        operand.dimensionCount = operands[i].dimension_count;
        operand.dimensions = operands[i].dimension_count ? (const u32*)next_pointer : NULL;

        next_pointer += 4 * operands[i].dimension_count;
        CAFFE_ENFORCE(next_pointer <= end_of_buf);

        int result = nnapi->Model_addOperand(model, &operand);
        NNAPI_CHECK(result);
      }

      for (int i = 0; i < ser_model->value_count; i++) {
        u32 len = values[i].source_length;
        const u8* stored_pointer = next_pointer;
        const void* value_pointer = NULL;
        usize value_length;

        switch ((SourceType)values[i].source_type) {
          case SOURCE_IMMEDIATE:
            {
              value_pointer = stored_pointer;
              value_length = len;
            }
            break;
          case SOURCE_NUMBERED_BUFFER:
            {
              CAFFE_ENFORCE(len == 12);
              u32 buffer_number = *(u32*)stored_pointer;
              u32 buffer_offset = *(u32*)(stored_pointer + 4);
              u32 operand_length = *(u32*)(stored_pointer + 8);
              CAFFE_ENFORCE(buffer_number < num_buffers);
              CAFFE_ENFORCE(buffer_offset + operand_length >= buffer_offset);  // No integer overflow
              CAFFE_ENFORCE(buffer_offset + operand_length <= (u32)buffer_sizes[buffer_number]);  // No buffer overflow
              value_pointer = (u8*)buffer_ptrs[buffer_number] + buffer_offset;
              value_length = operand_length;
            }
            break;
          case SOURCE_NUMBERED_MEMORY:
            CAFFE_ENFORCE(false, "Memory inputs not implemented yet.");
            break;
          default:
            CAFFE_ENFORCE(false, "Unknown source type: ", values[i].source_type);
        }

        CAFFE_ENFORCE(value_pointer != NULL);

        next_pointer += value_physical_size(len);
        CAFFE_ENFORCE(next_pointer <= end_of_buf);

        int result = nnapi->Model_setOperandValue(
            model,
            values[i].index,
            value_pointer,
            value_length);
        NNAPI_CHECK(result);
      }

      for (int i = 0; i < ser_model->operation_count; i++) {
        const u32* inputs = (const u32*)next_pointer;
        next_pointer += 4 * operations[i].input_count;
        CAFFE_ENFORCE(next_pointer <= end_of_buf);
        const u32* outputs = (const u32*)next_pointer;
        next_pointer += 4 * operations[i].output_count;
        CAFFE_ENFORCE(next_pointer <= end_of_buf);

        int result = nnapi->Model_addOperation(
            model,
            operations[i].operation_type,
            operations[i].input_count,
            inputs,
            operations[i].output_count,
            outputs);
        NNAPI_CHECK(result);
      }

      const u32* model_inputs = (const u32*)next_pointer;
      next_pointer += 4 * ser_model->input_count;
      CAFFE_ENFORCE(next_pointer <= end_of_buf);
      const u32* model_outputs = (const u32*)next_pointer;
      next_pointer += 4 * ser_model->output_count;
      CAFFE_ENFORCE(next_pointer <= end_of_buf);

      int result = nnapi->Model_identifyInputsAndOutputs(
          model,
          ser_model->input_count,
          model_inputs,
          ser_model->output_count,
          model_outputs);
      NNAPI_CHECK(result);

      *out_input_count = ser_model->input_count;
      *out_output_count = ser_model->output_count;

      // TODO: Maybe eliminate required_size and just rely on next_pointer for bounds checking.
      CAFFE_ENFORCE(next_pointer <= end_of_buf);
      CAFFE_ENFORCE(next_pointer == (const u8*)serialized_model + required_size);
      if (out_bytes_consumed != NULL) {
        *out_bytes_consumed = next_pointer - (const u8*)serialized_model;
      }

      return 0;
        */
}
