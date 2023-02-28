crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/TensorTransformations.h]

#[inline] pub fn roll_common(
        self_:  &Tensor,
        shifts: &[i32],
        dims:   &[i32]) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(shifts.size() > 0, "`shifts` required");
      if (dims.size() == 0 && shifts.size() == 1) {
        auto flattened = self.contiguous().view(self.numel());
        return roll(flattened, shifts[0], 0).view(self.sizes());
      }
      TORCH_CHECK(
        shifts.size() == dims.size(),
        "shifts and dimensions must align. shifts: ", shifts.size(), ", dims:", dims.size()
      );
      AT_ASSERT(dims.size() > 1);
      auto tail_shifts = shifts.slice(1);
      auto tail_dims = dims.slice(1);
      auto first_dim_rolled = roll(self, shifts[0], dims[0]);
      return roll(first_dim_rolled, tail_shifts, tail_dims);
        */
}

pub type FlipFn = fn(_0: &mut TensorIterator, _1: bool) -> ();

declare_dispatch!{flip_fn, flip_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/TensorTransformations.cpp]

pub fn flip(
        self_: &Tensor,
        dims:  &[i32]) -> Tensor {
    
    todo!();
        /*
            const i64 total_dims = self.dim();
      // It wraps the dims and checks that there are no repeated dims
      auto flip_dims_b = dim_list_to_bitset(dims, total_dims);

      Tensor out_tensor = empty_like(self, MemoryFormat::Preserve);

      // Count dimensions in which we need to do work
      int n = 0;
      auto strides = DimVector(self.strides());
      for(i64 i = 0; i < total_dims; i++) {
        if(flip_dims_b[i] && self.size(i) > 1 && self.stride(i) != 0) {
          n++;
          strides[i] = 0;
        }
      }

      // Nothing to do, we return fast
      if (n == 0 || self.numel() <=1) {
        out_tensor.copy_(self);
        return out_tensor;
      }

      //create dummy output with 0 strides at flipped dimension, to prevent tensorIterator from coalescing flipped dims
      const auto restrided_self = self.as_strided(self.sizes(), strides);
      auto iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .declare_static_dtype_and_device(self.scalar_type(), self.device())
        .add_output(out_tensor)
        .add_input(self)
        .add_input(restrided_self)
        .build();

      auto* data = reinterpret_cast<char*>(iter.data_ptr(0));
      const auto sizes = iter.shape();
      // This is a SmallVector of _signed_ ints
      auto strides_bytes = DimVector(iter.strides(0));
      const auto strides_self = iter.strides(1);
      const auto strides_dummy = iter.strides(2);

      // To understand this transformation, think of a 3D cube.
      //   - The data ptr points to the lower-left most vertex of the cube
      //   - The strides tell us how to move in each dimension,
      //     that is, data + stride[i] advances one element in the dimension i
      // To flip a dimension:
      //   - We move the pointer to the opposite vertex of the cube
      //   - We iterate in the opposite direction (invert the strides)

      for (int i=0; i<iter.ndim(); i++){
        // We know that an dimension has a zero stride and self[i] does not, as we defined above
        // Note that it may be the case that strides_dummy[i] = 0 not because we set it, but because
        // strides_self[i] == 0. We do not want to do anything there
        if (strides_dummy[i] == 0 && strides_self[i] != 0) {
          data += strides_bytes[i] * (sizes[i]-1);
          strides_bytes[i] *= -1;
        }
      }
      iter._unsafe_set_arg_strides(0, strides_bytes);
      iter._unsafe_set_arg_data(0, reinterpret_cast<void*>(data));

      flip_stub(iter.device_type(), iter, self.is_quantized());

      return out_tensor;
        */
}

pub fn roll_cpu(
        self_:  &Tensor,
        shifts: &[i32],
        dims:   &[i32]) -> Tensor {
    
    todo!();
        /*
            if (dims.size() != 1 || shifts.size() != 1) {
        return roll_common(self, shifts, dims);
      }
      // avoid a div zero error below.
      if (self.numel() == 0) {
        return self.clone(MemoryFormat::Preserve);
      }
      i64 dim = dims[0];
      i64 size = self.size(dim);
      i64 start = (size - shifts[0]) % size;
      // Behavior of % is different in C++ vs Python for negative numbers. This
      // corrects the difference.
      if (start < 0) {
        start = start + size;
      }
      auto t0 = self.narrow(dim, start, size-start);
      auto t1 = self.narrow(dim, 0, start);
      return cat({t0, t1}, dim);
        */
}

pub fn rot90(
        self_: &Tensor,
        k:     i64,
        dims:  &[i32]) -> Tensor {
    
    todo!();
        /*
            const i64 total_dims = self.dim(), total_rot_dims = dims.size();

      TORCH_CHECK(total_rot_dims == 2,
        "expected total rotation dims == 2, but got dims = ", total_rot_dims);

      TORCH_CHECK(total_dims >= 2,
        "expected total dims >= 2, but got total dims = ", total_dims);

      TORCH_CHECK(dims[0] != dims[1] && abs(dims[0] - dims[1]) != total_dims,
        "expected rotation dims to be different, but got dim0 = ", dims[0],
        " and dim1 = ", dims[1]);

      // check range of dims
      TORCH_CHECK(dims[0] < total_dims && dims[0] >= -total_dims,
        "Rotation dim0 out of range, dim0 = ", dims[0]);

      TORCH_CHECK(dims[1] < total_dims && dims[1] >= -total_dims,
        "Rotation dim1 out of range, dim1 = ", dims[1]);

      // handle modulo with negative k
      k = (4 + (k % 4)) % 4;

      switch(k) {
        case 1:
          return self.flip({dims[1]}).transpose_(dims[0], dims[1]);
        case 2:
          return self.flip(dims);
        case 3:
          return self.flip({dims[0]}).transpose_(dims[0], dims[1]);
        default:
          return self.clone(MemoryFormat::Contiguous);
      }
        */
}

pub fn fliplr(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.dim() >= 2, "Input must be >= 2-d.");

      return self.flip({1});
        */
}

pub fn flipud(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.dim() >= 1, "Input must be >= 1-d.");

      return self.flip({0});
        */
}

pub trait AtLeast1D {

    type Output;

    fn atleast_1d(&self) -> Self::Output;
}

impl AtLeast1D for Tensor {

    type Output = Tensor;

    fn atleast_1d(&self) -> Self::Output {
        
        todo!();
            /*
                switch (self.dim()) {
            case 0:
              return self.reshape({1});
            default:
              return self;
          }
            */
    }
}

impl AtLeast1D for TensorList {

    type Output = Vec<Tensor>;

    fn atleast_1d(&self) -> Self::Output {
        
        todo!();
            /*
                vector<Tensor> result(self.size());
          auto transform_lambda = [](const Tensor& input) -> Tensor {
            return native::atleast_1d(input);
          };
          transform(self.cbegin(), self.cend(), result.begin(), transform_lambda);
          return result;
            */
    }
}

pub trait AtLeast2D {

    type Output;

    fn atleast_2d(&self) -> Self::Output;
}

impl AtLeast2D for Tensor {

    type Output = Tensor;

    fn atleast_2d(&self) -> Self::Output {
        
        todo!();
            /*
                switch (self.dim()) {
            case 0:
              return self.reshape({1, 1});
            case 1: {
              return self.unsqueeze(0);
            }
            default:
              return self;
          }
            */
    }
}

impl AtLeast2D for TensorList {

    type Output = Vec<Tensor>;

    fn atleast_2d(&self) -> Self::Output {
        
        todo!();
            /*
                vector<Tensor> result(self.size());
          auto transform_lambda = [](const Tensor& input) -> Tensor {
            return native::atleast_2d(input);
          };
          transform(self.cbegin(), self.cend(), result.begin(), transform_lambda);
          return result;
            */
    }
}

pub trait AtLeast3D {

    type Output;

    fn atleast_3d(&self) -> Self::Output;
}

impl AtLeast3D for Tensor {

    type Output = Tensor;

    fn atleast_3d(&self) -> Self::Output {
        
        todo!();
            /*
                switch (self.dim()) {
            case 0:
              return self.reshape({1, 1, 1});
            case 1: {
              return self.unsqueeze(0).unsqueeze(-1);
            }
            case 2: {
              return self.unsqueeze(-1);
            }
            default:
              return self;
          }
            */
    }
}

impl AtLeast3D for TensorList {

    type Output = Vec<Tensor>;

    fn atleast_3d(&self) -> Self::Output {
        
        todo!();
            /*
                vector<Tensor> result(self.size());
          auto transform_lambda = [](const Tensor& input) -> Tensor {
            return native::atleast_3d(input);
          };
          transform(self.cbegin(), self.cend(), result.begin(), transform_lambda);
          return result;
            */
    }
}

define_dispatch!{flip_stub}
