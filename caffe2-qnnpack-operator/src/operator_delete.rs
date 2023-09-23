crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/operator-delete.c]

pub fn pytorch_qnnp_delete_operator(op: PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
            if (op == NULL) {
        return pytorch_qnnp_status_invalid_parameter;
      }

      free(op->indirection_buffer);
      free(op->packed_weights);
      free(op->a_sum);
      free(op->zero_buffer);
      free(op->lookup_table);
      free(op);
      return pytorch_qnnp_status_success;
        */
}
