crate::ix!();

num_inputs!{SpatialSoftmaxWithLoss, (2,3)}

num_outputs!{SpatialSoftmaxWithLoss, 2}

inputs!{SpatialSoftmaxWithLoss, 
    0 => ("logits",        "Unscaled log probabilities"),
    1 => ("labels",        "Ground truth"),
    2 => ("weight_tensor", "Optional blob to be used to weight the samples for the loss. With spatial set, weighting is by x,y of the input")
}

outputs!{SpatialSoftmaxWithLoss, 
    0 => ("softmax", "Tensor with softmax cross entropy loss"),
    1 => ("loss",    "Average loss")
}

tensor_inference_function!{SpatialSoftmaxWithLoss, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          ArgumentHelper helper(def);
          vector<TensorShape> out(2);

          auto logits = in[0]; // Tensor with Shape [batch_size, num_classes]
          auto labels = in[1]; // Tensor with shape [batch_size, ]
          auto batch_size = logits.dims().Get(0);
          auto num_classes = logits.dims().Get(1);

          CAFFE_ENFORCE_EQ(logits.dims_size(), 4);
          CAFFE_ENFORCE_EQ(labels.dims_size(), 3);
          out[0].set_data_type(logits.data_type());
          out[0].add_dims(batch_size);
          out[0].add_dims(num_classes);
          out[0].add_dims(in[0].dims(2));
          out[0].add_dims(in[0].dims(3));
          // Output 2 is scalar shape, so no dims added
          return out;
        */
    }
}
