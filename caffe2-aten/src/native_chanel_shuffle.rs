crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ChanelShuffle.cpp]

pub fn channel_shuffle(
        self_:  &Tensor,
        groups: i64) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.dim() > 2,
                  "channel_shuffle expects input with > 2 dims, but got input with sizes ",
                  self.sizes());
      i64 b = self.size(0);
      i64 c = self.size(1);
      TORCH_CHECK(groups > 0,
                  "Number of groups to divide channels in must be positive.",
                  " Value of groups:", groups);
      TORCH_CHECK((c % groups) == 0,
                  "Number of channels must be divisible by groups. Got ",
                  c, " channels and ", groups, " groups.");

    #if defined(C10_MOBILE) && defined(USE_XNNPACK)
      if (self.is_contiguous(MemoryFormat::ChannelsLast) &&
          xnnpack::use_channel_shuffle(self, groups)) {
        return xnnpack::channel_shuffle(self, groups);
      }
    #endif

      i64 oc = c / groups;

      auto input_reshaped = self.view({b, groups, oc, -1});
      // TODO: contiguous can be made to preserve the memory format
      // of the input. However since the above reshape clobbers h and w
      // it may not be safe to do that, since channels_last contiguous
      // may think oc and and the last dim correspond to h,w?
      // It is not clear, however from initial looking around it feels that
      // this may not be correct.
      // In this case channels last will likely require custom implementation
      // if we want to preseve the memory order.
      // XNNPACK has channel shuffle op for NHWC. For mobile usecase this is good.
      // For server we will have to do a custom implementation.
      // For ChannelsFirst, a.k.a Contiguous, memory format we will also need
      // a fast custom implementation perhaps.
      Tensor output_tensor =
          input_reshaped.permute({0 /* b */, 2 /* oc */, 1 /* groups */, 3})
          .contiguous()
          .reshape(self.sizes());
      return namedinference::propagate_names_if_nonempty(
          output_tensor,
          self.has_names() ? self.names() : ArrayRef<Dimname>{});
        */
}
