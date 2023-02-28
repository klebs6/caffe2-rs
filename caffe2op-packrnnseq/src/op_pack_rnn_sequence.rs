crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorStorage,
    OperatorDef
};

/**
| Pack values based on the length blob. Each number
| from length blob represents the corresponding
| values that need to be packed. The dimension for
| each pack is the same as the maximum number from
| the length blob (padding with zero is implemented
| for smaller length value). The overall output
| dimension is:
|
| T * N * D, where T is the max number of lengths,
| N is the size of lengths, and D is the dimension
| of each feature value. The following example shows
| the input and output of this operator:
|
|
| Given:
|   values = [v1, v2, v3, v4, v5, v6, v7, v8]
|   lengths = [2, 3, 1, 2];
|
|
| Output:
|   output = [
|     [v1, v3, v6, v7],
|     [v2, v4, 0,  v8],
|     [0,  v5, 0,  0 ],
|   ]
|
|
| One application for this operator is the transfer
| data into the format that is used for RNN
| models. Note that the gradient operator of
| PackRNNSequence is UnpackRNNSequence.
*/
pub struct PackRNNSequenceOpBase<Context,const Forward: bool> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{PackRNNSequence, 2}

num_outputs!{PackRNNSequence, 1}

inputs!{PackRNNSequence, 
    0 => ("values", "Data tensor, contains a sequence of features"),
    1 => ("lengths", "lengths with each number representing the pack size.")
}

outputs!{PackRNNSequence, 
    0 => ("output", "Output tensor after packing")
}

impl<Context,const Forward: bool> 
PackRNNSequenceOpBase<Context,Forward> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t, float, double>>::call(
            this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<ValT>(&mut self) -> bool {
        todo!();
        /*
            // The value is copied from the sequence to the pack
        // if Forward is true, and vice versa
        int dim_offset = Forward ? 1 : 2;
        auto& values = Input(0);
        CAFFE_ENFORCE_GT(values.dim(), dim_offset);

        // block_size is the size for each individual feature
        int64_t block_size = values.size_from_dim(dim_offset);
        auto values_vec = values.template data<ValT>();

        auto& lengths = Input(LENGTHS);
        CAFFE_ENFORCE_EQ(lengths.dim(), 1);
        const auto cols = lengths.numel();
        const int32_t* lengths_vec = lengths.template data<int32_t>();
        // the total number of rows is defined as the max number from lengths
        // if when the lengths is empty, we set rows = 0 to support zero lengths
        const auto rows =
            cols ? *std::max_element(lengths_vec, lengths_vec + cols) : 0;
        CAFFE_ENFORCE_GE(rows, 0);
        int length_sum = 0;
        if (cols > 0) {
          math::Sum<int, Context>(cols, lengths_vec, &length_sum, &context_);
        }

        vector<int64_t> shape;
        // the output shape is rows * cols for the pack,
        // or length_sum for the sequence
        if (Forward) {
          shape.push_back(rows);
          shape.push_back(cols);
        } else {
          shape.push_back(length_sum);
        }
        // insert the dim for the feature
        shape.insert(
            shape.end(), values.sizes().begin() + dim_offset, values.sizes().end());

        auto* output = Output(OUTPUTVALUE, shape, at::dtype<ValT>());

        auto output_data = output->template mutable_data<ValT>();
        // initialize output_data with zero, as it is the default value for padding
        // when certain length is smaller than rows
        math::Set<ValT, Context>(output->numel(), 0, output_data, &context_);

        int32_t offset = 0;
        for (int c = 0; c < cols; c++) {
          for (int r = 0; r < lengths_vec[c]; r++) {
            auto input_offset = Forward ? (offset + r) : (r * cols + c);
            auto output_offset = Forward ? (r * cols + c) : (offset + r);
            context_.CopyItemsSameDevice(
                values.dtype(),
                block_size,
                values_vec + input_offset * block_size,
                output_data + output_offset * block_size);
          }
          offset += lengths_vec[c];
        }
        return true;
        */
    }
}

input_tags!{
    PackRNNSequenceOp {
        Inputvalue,
        Lengths
    }
}

output_tags!{
    PackRNNSequenceOp {
        Outputvalue
    }
}

register_cpu_operator!{PackRNNSequence, PackRNNSequenceOpBase<CPUContext, true>}

/**
| This is the reverse operator for
| PackRNNSequence. It maps the packed values back to
| sequence values based on the length blob. Each
| number from length blob represents the
| corresponding values that has been grouped. The
| dimension for each pack is the same as the maximum
| number from the length blob (padding with zero was
| implemented for smaller length value). The overall
| output dimension is: M * D, where M is the sum of
| lengths, and D is the dimension of each feature
| value. The following example shows the input and
| output of this operator:
|
|
| Given:
|   values = [
|     [v1, v3, v6, v7],
|     [v2, v4, 0,  v8],
|     [0,  v5, 0,  0 ],
|   ]
|   lengths = [2, 3, 1, 2]
|
|
| Output:
|   output = [v1, v2, v3, v4, v5, v6, v7, v8];
|
|
| One application for this operator is the transfer
| data from the format of RNN back to sequence
| values. Note that the gradient operator of
| UnpackRNNSequence is PackRNNSequence.
*/
register_cpu_operator!{
    UnpackRNNSequence,
    PackRNNSequenceOpBase<CPUContext, false>
}

num_inputs!{UnpackRNNSequence, 2}

num_outputs!{UnpackRNNSequence, 1}

inputs!{UnpackRNNSequence, 
    0 => ("values", "Data tensor, contains the packed features"),
    1 => ("lengths", "lengths with each number representing the pack size.")
}

outputs!{UnpackRNNSequence, 
    0 => ("output", "Output tensor before packing")
}

///--------------------------
pub struct GetPackRNNSequenceGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetPackRNNSequenceGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(def_.input_size(), 2);
        return SingleGradientDef(
            "UnpackRNNSequence",
            "",
            vector<string>{GO(0), I(1)},
            vector<string>{GI(0)});
        */
    }
}

pub struct GetUnpackRNNSequenceGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetUnpackRNNSequenceGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(def_.input_size(), 2);
        return SingleGradientDef(
            "PackRNNSequence",
            "",
            vector<string>{GO(0), I(1)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{PackRNNSequence,   GetPackRNNSequenceGradient}

register_gradient!{UnpackRNNSequence, GetUnpackRNNSequenceGradient}
