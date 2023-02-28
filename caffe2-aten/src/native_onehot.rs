crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Onehot.cpp]

pub fn one_hot(
        self_:       &Tensor,
        num_classes: i64) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.dtype() == kLong, "one_hot is only applicable to index tensor.");
        auto shape = self.sizes().vec();

        // empty tensor could be converted to one hot representation,
        // but shape inference is not possible.
        if (self.numel() == 0) {
            if (num_classes <= 0) {
                AT_ERROR("Can not infer total number of classes from empty tensor.");
            } else {
                shape.push_back(num_classes);
                return empty(shape, self.options());
            }
        }

        // non-empty tensor
        if (self.device().type() != kCUDA) {
          //for cuda, rely on device assert thrown by scatter
          TORCH_CHECK(self.min().item().toLong() >= 0, "Class values must be non-negative.");
        }
        if (num_classes == -1) {
            num_classes = self.max().item().toLong() + 1;
        } else {
            if (self.device().type() != kCUDA) {
              //rely on device asserts from scatter to avoid sync here
              TORCH_CHECK(num_classes > self.max().item().toLong(), "Class values must be smaller than num_classes.");
            } else {
                //for cuda, assert that num_classes is at least 1
                TORCH_CHECK(num_classes >= 1, "num_classes should be positive");
            }
        }

        shape.push_back(num_classes);
        Tensor ret = zeros(shape, self.options());
        ret.scatter_(-1, self.unsqueeze(-1), 1);
        return ret;
        */
}
