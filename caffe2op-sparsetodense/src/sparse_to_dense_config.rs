crate::ix!();

register_cpu_operator!{SparseToDense, SparseToDenseOp<CPUContext>}

num_inputs!{SparseToDense, (2,3)}

num_outputs!{SparseToDense, 1}

inputs!{SparseToDense, 
    0 => ("indices",           "1-D int32/int64 tensor of concatenated ids of data"),
    1 => ("values",            "Data tensor, first dimension has to match `indices`, basic numeric types are supported"),
    2 => ("data_to_infer_dim", "Optional: if provided, the first dimension of output is the first dimension of this tensor.")
}

outputs!{SparseToDense, 
    0 => ("output",            "Output tensor of the same type as `values` of shape `[len(lengths), len(mask)] + shape(default_value)` (if `lengths` is not provided the first dimension is omitted)")
}

tensor_inference_function!{SparseToDense, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          vector<TensorShape> out(1);
          if (in.size() == 3) {
            out[0].add_dims(in[2].dims(0));
          } else {
            out[0].set_unknown_shape(true);
            return out;
          }
          for (int i = 1; i < in[1].dims().size(); i++) {
            out[0].add_dims(in[1].dims(i));
          }
          out[0].set_data_type(in[1].data_type());
          return out;
        */
    }
}

input_tags!{
    SparseToDenseOp
    {
        Indices,
        Values,
        DataToInferDim
    }
}
