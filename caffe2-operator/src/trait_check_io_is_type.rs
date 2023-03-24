crate::ix!();

pub trait CheckInputIsType {

    #[inline] fn input_is_type<T>(idx: i32) -> bool {
        todo!();
        /*
            CAFFE_ENFORCE(
                isLegacyOperator(),
                "InputIsType(idx) not (yet) supported for operators exported to c10.");
            static_assert(
                !std::is_same<T, Tensor>::value,
                "You should use InputIsTensorType(int, DeviceType) for "
                "Tensor.");
            return inputs_.at(idx)->template IsType<T>();
        */
    }

    #[inline] fn input_is_tensor_type(
        &mut self, 
        idx: i32,
        device_type: DeviceType) -> bool 
    {
        todo!();
        /*
            CAFFE_ENFORCE(
            isLegacyOperator(),
            "InputIsTensorType(idx, device_type) not (yet) supported for operators exported to c10.");
        return BlobIsTensorType(*inputs_.at(idx), device_type);
        */
    }
}

pub trait CheckOutputIsType {

    #[inline] fn output_is_type<T>(idx: i32) -> bool {
        todo!();
        /*
            CAFFE_ENFORCE(
                isLegacyOperator(),
                "OutputIsType(idx) not (yet) supported for operators exported to c10.");
            static_assert(
                !std::is_same<T, Tensor>::value,
                "You should use OutputIsTensorType(int, DeviceType) for "
                "Tensor.");
            return outputs_.at(idx)->template IsType<T>();
        */
    }

    #[inline] fn output_is_tensor_type(
        &mut self, 
        idx: i32,
        ty: DeviceType) -> bool 
    {
        todo!();
        /*
            CAFFE_ENFORCE(
            isLegacyOperator(),
            "OutputIsTensorType(idx, type) not (yet) supported for operators exported to c10.");
        return BlobIsTensorType(*outputs_.at(idx), type);
        */
    }
}
