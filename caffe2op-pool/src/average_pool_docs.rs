crate::ix!();

#[inline] pub fn average_pool_doc_generator(dim: *const u8) -> fn(_u0: &mut OpSchema) -> c_void {
    
    todo!();
    /*
        return [=](OpSchema& schema) {
        std::string doc = "AveragePool{dim} {pool_doc}";
        c10::ReplaceAll(doc, "{dim}", dim);
        c10::ReplaceAll(doc, "{pool_doc}", kAveragePoolDoc);
        schema.SetDoc(doc);
        schema.Input(
            0,
            "X",
            "*(type: Tensor`<float>`)* Input data tensor of shape NCHW or NHWC.");
        schema.Output(0, "Y", "*(type: Tensor`<float>`)* Output data tensor.");
        // schema.Arg(
        //     "kernel", "*(type: int)* Size of the window to take an average
        //     over.");
        // schema.Arg("stride", "*(type: int)* Stride of the window.");
        // schema.Arg(
        //     "pad",
        //     "*(type: int)* Implicit zero padding to be added on both sides.");
        // schema.Arg(
        //     "dilation",
        //     "*(type: int)* Parameter that controls the stride of elements in the
        //     " "window.");
        // schema.Arg(
        //     "order",
        //     "*(type: string; default: 'NCHW')* Order of the blob dimensions.");
        // schema.Arg(
        //     "count_include_pad",
        //     "*(type: bool; default: False)* When True, will include the "
        //     "zero-padding in the averaging.");
      };
    */
}
