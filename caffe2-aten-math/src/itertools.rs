crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Itertools.cpp]

pub fn triu_mask(
        n:        i64,
        dims:     i64,
        diagonal: bool,
        opt:      TensorOptions) -> Tensor {
    
    todo!();
        /*
            // get a mask that has value 1 whose indices satisfies i < j < k < ...
      // or i <= j <= k <= ... (depending on diagonal)
      Tensor range = arange(n, opt.dtype(kLong));
      vector<Tensor> index_grids = meshgrid(vector<Tensor>(dims, range));
      Tensor mask = full(index_grids[0].sizes(), true, opt.dtype(kBool));
      if(diagonal) {
        for(i64 i = 0; i < dims - 1; i++) {
          mask *= index_grids[i] <= index_grids[i+1];
        }
      } else {
        for(i64 i = 0; i < dims - 1; i++) {
          mask *= index_grids[i] < index_grids[i+1];
        }
      }
      return mask;
        */
}

pub fn cartesian_prod(tensors: &[Tensor]) -> Tensor {
    
    todo!();
        /*
            for(const Tensor &t : tensors) {
        TORCH_CHECK(t.dim() == 1, "Expect a 1D vector, but got shape ", t.sizes());
      }
      if (tensors.size() == 1) {
        return tensors[0];
      }
      vector<Tensor> grids = meshgrid(tensors);
      for(Tensor &t : grids) {
        t = t.flatten();
      }
      return stack(grids, 1);
        */
}

pub fn combinations(
        self_:            &Tensor,
        r:                i64,
        with_replacement: bool) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.dim() == 1, "Expect a 1D vector, but got shape ", self.sizes());
      TORCH_CHECK(r > 0, "Expect a positive number, but got ", r);
      i64 num_elements = self.numel();
      vector<Tensor> grids = meshgrid(vector<Tensor>(r, self));
      Tensor mask = _triu_mask(num_elements, r, with_replacement, self.options());
      for(Tensor &t : grids) {
        t = t.masked_select(mask);
      }
      return stack(grids, 1);
        */
}
