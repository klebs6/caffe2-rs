crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/NamedTensor.cpp]

lazy_static!{
    /*
    thread_local bool NamesMode_enabled = true;
    */
}

impl NamesMode {
    
    pub fn is_enabled(&mut self) -> bool {
        
        todo!();
        /*
            return NamesMode_enabled;
        */
    }
    
    pub fn set_enabled(&mut self, enabled: bool)  {
        
        todo!();
        /*
            NamesMode_enabled = enabled;
      tls_set_dispatch_key_excluded(DispatchKey::Named, !enabled);
        */
    }
}

pub fn internal_set_names_inplace_with_dimname_list(
    tensor: &Tensor,
    names:  Option<DimnameList>) -> &Tensor {
    
    todo!();
        /*
        internal_set_names_inplace(tensor.unsafeGetTensorImpl(), names, /*validate_names=*/true);
      return tensor;
        */
}

pub fn internal_set_names_inplace(
    tensor:         &Tensor,
    names:          Vec<Dimname>,
    validate_names: bool) -> &Tensor {

    /*
    pub fn internal_set_names_inplace(
        impl_:          *mut TensorImpl,
        names:          Vec<Dimname>,
        validate_names: bool)  {
        
        todo!();
            /*
                if (validate_names) {
            check_names_valid_for(impl, names);
          }
          // Do this after validation!
          if (all_of(names.begin(), names.end(), [](const Dimname& n) { return n.isWildcard(); })) {
            impl->set_named_tensor_meta(nullptr);
            return;
          }
          auto* meta = get_named_tensor_meta(impl);
          if (meta == nullptr) {
            impl->set_named_tensor_meta(make_unique<NamedTensorMetaInterface>(NamedTensorMetaInterface::HasNonWildcard, names));
          } else {
            meta->set_names(NamedTensorMetaInterface::HasNonWildcard, names);
          }
            */
    }
    */
        
    todo!();
        /*
            internal_set_names_inplace(tensor.unsafeGetTensorImpl(), move(names), validate_names);
      return tensor;
        */
}

pub fn default_names(len: usize) -> DimnameList {
    
    todo!();
        /*
            static vector<Dimname> all_unnamed(kMaxNamedTensorDim, Dimname::wildcard());
        TORCH_INTERNAL_ASSERT(
            len <= kMaxNamedTensorDim,
            "Only tensors with up to ", kMaxNamedTensorDim, " are supported.");
      return DimnameList(&all_unnamed.front(), len);
        */
}

pub fn check_unique_names(names: DimnameList)  {
    
    todo!();
        /*
            // Strategy: Compare each element with the ones that come after it.
      // Although this is O(N^2), in practice N is small (no more than 25).
      for (auto it = names.begin(); it != names.end(); ++it) {
        if (it->isWildcard()) continue;
        auto dup = find(it + 1, names.end(), *it);
        while (dup != names.end()) {
          TORCH_CHECK(false,
              "Cannot construct a tensor with duplicate names. Got names: ",
              names, ".");
        }
      }
        */
}

pub fn check_names_valid_for(
        tensor: &Tensor,
        names:  DimnameList)  {

    /*
    fn check_names_valid_for(
            tensor_dim: usize,
            names:      DimnameList)  {
        
        todo!();
            /*
                TORCH_CHECK(
              tensor_dim <= kMaxNamedTensorDim,
              "Named tensors only support up to ", kMaxNamedTensorDim, " dims: "
              "Attempted to create a tensor with dim ", tensor_dim, " with names ", names);
          TORCH_CHECK(tensor_dim == names.size(),
              "Number of names (", names.size(), ") and "
              "number of dimensions in tensor (", tensor_dim, ") ",
              "do not match. Attempted to create a tensor with names ", names);
          check_unique_names(names);
            */
    }

    fn check_names_valid_for(
            impl_: *mut TensorImpl,
            names: DimnameList)  {
        
        todo!();
            /*
                check_names_valid_for(impl->dim(), names);
            */
    }
    */
        
    todo!();
        /*
            return check_names_valid_for(tensor.unsafeGetTensorImpl(), names);
        */
}

pub fn get_named_tensor_meta_mut(impl_: *mut TensorImpl) -> *mut dyn NamedTensorMetaInterface {
    
    todo!();
        /*
            if (!NamesMode::is_enabled()) {
        return nullptr;
      }
      return static_cast<dyn NamedTensorMetaInterface*>(impl->named_tensor_meta());
        */
}

pub fn get_named_tensor_meta(impl_: *const TensorImpl) -> *const dyn NamedTensorMetaInterface {
    
    todo!();
        /*
            if (!NamesMode::is_enabled()) {
        return nullptr;
      }
      return static_cast<const NamedTensorMetaInterface*>(impl->named_tensor_meta());
        */
}

pub fn internal_set_names_inplace_with_maybe_dimname_list(
    impl_:          *mut TensorImpl,
    names:          Option<DimnameList>,
    validate_names: bool)  {
    
    todo!();
        /*
            if (!names) {
        impl->set_named_tensor_meta(nullptr);
        return;
      }
      if (validate_names) {
        check_names_valid_for(impl, *names);
      }
      // Do this after validation!
      if (all_of(names->begin(), names->end(), [](const Dimname& n) { return n.isWildcard(); })) {
        impl->set_named_tensor_meta(nullptr);
        return;
      }
      auto* meta = get_named_tensor_meta(impl);
      if (meta == nullptr) {
        // Constructor is private
        impl->set_named_tensor_meta(make_unique<NamedTensorMetaInterface>(NamedTensorMetaInterface::HasNonWildcard, *names));
      } else {
        meta->set_names(NamedTensorMetaInterface::HasNonWildcard, *names);
      }
        */
}

pub fn get_opt_names(impl_: *const TensorImpl) -> Option<DimnameList> {
    
    todo!();
        /*
            const auto* meta = get_named_tensor_meta(impl);
      if (meta == nullptr) {
        return nullopt;
      } else {
        return meta->names();
      }
        */
}

pub fn get_names(impl_: *const TensorImpl) -> DimnameList {
    
    todo!();
        /*
            auto maybe_names = get_opt_names(impl);
      if (maybe_names) {
        return *maybe_names;
      }
      return default_names(impl->dim());
        */
}

pub fn has_names(impl_: *const TensorImpl) -> bool {
    
    todo!();
        /*
            return impl->has_named_tensor_meta() && NamesMode::is_enabled();
        */
}
