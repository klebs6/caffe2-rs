crate::ix!();

impl<Context> NumpyTileOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& input = Input(0);
        const auto& repeats = Input(1);

        // Check that the `repeats` tensor has the correct rank, has a number of
        // elements equal to the number of axes of `input`.
        CAFFE_ENFORCE_EQ(repeats.dim(), 1, "repeats input must be a 1-d tensor");
        CAFFE_ENFORCE_EQ(
            repeats.numel(),
            input.dim(),
            "repeats input have the same"
            " number of elements as `inputs` has dimensions.");
        const int64_t* repeats_data = repeats.template data<int64_t>();
        for (size_t i = 0; i < repeats.numel(); ++i) {
          CAFFE_ENFORCE_GE(repeats_data[i], 0);
        }

        auto* output = Output(0);

        // Alternate inputs and outputs between two buffers. Repeatedly apply the
        // Tile kernel along each axis. Then copy out the resulting data into the
        // output tensor.
        Tensor *src = &buffer, *dst = output;
        src->CopyFrom(input);
        vector<int64_t> output_dims(input.sizes().vec());
        for (size_t i = 0; i < repeats.numel(); ++i) {
          if (repeats_data[i] == 1) {
            continue;
          }
          // size up to (and not including) axis
          const auto outer_dim = src->size_to_dim(i);
          // size from axis up
          const auto inner_dim = src->size_from_dim(i);

          dst->Resize(outer_dim, inner_dim * repeats_data[i]);

          /**
           * How this works:
           * Imagine a 2D tensor (matrix) of size 3x10, tiled 2 times.
           * - Tiling along axis 0 (row) means copying the entire 3x10 Matrix 2
           * times. outer_dim = 0, inner_dim = 30.
           * - Tiling along axis 1 (column) means copying each row 2 times, then
           * proceed to the next row, until the end. outer_dim = 3, inner_dim = 10.
           */
          const char* src_data = static_cast<const char*>(src->raw_data());
          char* dst_data = static_cast<char*>(dst->raw_mutable_data(src->dtype()));

          DoTile(
              src->dtype(),
              src->itemsize(),
              outer_dim,
              inner_dim,
              repeats_data[i],
              src_data,
              dst_data);

          output_dims[i] *= repeats_data[i];
          dst->Reshape(output_dims);

          std::swap(src, dst);
        }

        // NB: because we have the swap at the end of the above loop, our real
        // result tensor is going to live in *src when we reach this line
        // whether we entered the loop or not :)
        if (output != src)
          output->CopyFrom(*src);

        return true;
        */
    }
    
    #[inline] pub fn do_tile(&mut self, 
        meta:        TypeMeta,
        item_size:   i32,
        outer_dim:   i32,
        inner_dim:   i32,
        num_tiles:   i64,
        input_data:  *const u8,
        output_data: *mut u8)  {

        todo!();
        /*
            for (auto i = 0; i < outer_dim; ++i) {
          for (auto t = 0; t < num_tiles; ++t) {
            context_.CopyItemsSameDevice(meta, inner_dim, input_data, output_data);
            output_data += inner_dim * item_size;
          }
          input_data += inner_dim * item_size;
        }
        */
    }
}
