crate::ix!();

tensor_inference_function!{SpatialBN,

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
            ArgumentHelper helper(def);
            bool is_test = helper.GetSingleArgument<int>(OpSchema::Arg_IsTest, 0);

            if (!is_test) {
                vector<TensorShape> out;
                StorageOrder order = StringToStorageOrder(
                    helper.GetSingleArgument<string>("order", "NCHW"));
                const TensorShape& X = in[0];
                const int C =
                    (order == StorageOrder::NCHW ? X.dims(1)
                     : X.dims(X.dims_size() - 1));

                out.push_back(in[0]);
                TensorShape meanvar_tp =
                    CreateTensorShape(vector<int>{C}, TensorProto::FLOAT);
                out.push_back(meanvar_tp); // RUNNING_MEAN
                out.push_back(meanvar_tp); // RUNNING_MEAN
                out.push_back(meanvar_tp); // SAVED_MEAN
                out.push_back(meanvar_tp); // SAVED_VAR
                return out;
            } else {
                return vector<TensorShape>{in[0]};
            }
        */
    }
}
