crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ReplicationPadding.cpp]

lazy_static!{
    /*
    TORCH_META_FUNC(replication_pad1d) (
      const Tensor& input, IntArrayRef paddingSize  // no out argument!
    ) {

      i64 dimw = 1;
      i64 dimslices = 0;
      i64 nbatch = 1;

      TORCH_CHECK(paddingSize.size() == 2, "padding size is expected to be 2");

      i64 pad_l = paddingSize[0];
      i64 pad_r = paddingSize[1];

      // allow empty batch size but not other dimensions.
      TORCH_CHECK((input.dim() == 2 && input.size(0) != 0 && input.size(1) != 0) ||
                  (input.dim() == 3 && input.size(1) != 0 && input.size(2) != 0),
                  "Expected 2D or 3D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ",
                  input.sizes());

      if (input.ndimension() == 3) {
        nbatch = input.size(0);
        dimw++;
        dimslices++;
      }

      /* sizes */
      i64 nslices = input.size(dimslices);
      i64 iwidth = input.size(dimw);
      i64 owidth = iwidth + pad_l + pad_r;

      TORCH_CHECK(owidth >= 1,
          "input (W: ", iwidth, ") is too small."
          " Calculated output W: ", owidth);

      if (input.ndimension() == 2) {
        set_output({nslices, owidth}, input.options());
      } else {
        set_output({nbatch, nslices, owidth}, input.options());
      }
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(replication_pad1d_backward) (
      const Tensor& gradOutput,
      const Tensor& input,
      IntArrayRef paddingSize
    ) {
      i64 dimw = 1;
      i64 dimslices = 0;
      i64 nbatch = 1;
      TORCH_CHECK(paddingSize.size() == 2, "padding size is expected to be 2");
      i64 pad_l = paddingSize[0];
      i64 pad_r = paddingSize[1];

      if (input.ndimension() == 3)
      {
        nbatch = input.size(0);
        dimw++;
        dimslices++;
      }

      /* sizes */
      i64 iwidth = input.size(dimw);
      i64 owidth  = iwidth + pad_l + pad_r;

      TORCH_CHECK(owidth == gradOutput.size(dimw),
          "gradOutput width unexpected. Expected: ", owidth,
          " Got: ", gradOutput.size(dimw));

      set_output(input.sizes(), input.options());
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(replication_pad2d) (
      const Tensor& input, IntArrayRef paddingSize
    ) {
      TORCH_CHECK(paddingSize.size() == 4, "padding size is expected to be 4");
      i64 pad_l = paddingSize[0];
      i64 pad_r = paddingSize[1];
      i64 pad_t = paddingSize[2];
      i64 pad_b = paddingSize[3];
      i64 dimw = 2;
      i64 dimh = 1;
      i64 dimslices = 0;
      i64 nbatch = 1;

      // allow 0 dim batch size and nothing else.
      bool valid_dims = input.size(1) != 0 && input.size(2) != 0;
      TORCH_CHECK(
          (input.dim() == 3 && input.size(0) != 0 && valid_dims) ||
          (input.dim() == 4 && valid_dims && input.size(3) != 0),
          "Expected 3D or 4D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ",
          input.sizes());

      if (input.dim() == 4)
      {
        nbatch = input.size(0);
        dimw++;
        dimh++;
        dimslices++;
      }

      /* sizes */
      i64 nslices = input.size(dimslices);
      i64 iheight = input.size(dimh);
      i64 iwidth = input.size(dimw);
      i64 oheight = iheight + pad_t + pad_b;
      i64 owidth  = iwidth + pad_l + pad_r;

      TORCH_CHECK(owidth >= 1 || oheight >= 1,
          "input (H: ", iheight, ", W: ", iwidth, " ) is too small."
          " Calculated output H: ", oheight, " W: ", owidth);

      if (input.dim() == 3) {
        set_output({nslices, oheight, owidth}, input.options());
      } else {
        set_output({nbatch, nslices, oheight, owidth}, input.options());
      }
    }
    */
}

#[inline] pub fn shape_check3d(
        input:   &Tensor,
        pleft:   i32,
        pright:  i32,
        ptop:    i32,
        pbottom: i32,
        pfront:  i32,
        pback:   i32)  {
    
    todo!();
        /*
            int dimw = 3;
      int dimh = 2;
      int dimd = 1;
      int dimslices = 0;

      // allow batch size of 0-dim.
      bool valid_dims = input.size(1) != 0 && input.size(2) != 0 && input.size(3) != 0;
      TORCH_CHECK(
          (input.dim() == 4 && input.size(0) != 0 && valid_dims) ||
          (input.dim() == 5 && valid_dims && input.size(4) != 0),
          "Expected 4D or 5D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ",
          input.sizes());

      if (input.dim() == 5)
      {
        dimw++;
        dimh++;
        dimd++;
        dimslices++;
      }

      /* sizes */
      // i64 nslices = input.size(dimslices);
      i64 idepth = input.size(dimd);
      i64 iheight = input.size(dimh);
      i64 iwidth = input.size(dimw);
      i64 odepth = idepth + pfront + pback;
      i64 oheight = iheight + ptop + pbottom;
      i64 owidth  = iwidth + pleft + pright;

      TORCH_CHECK(owidth >= 1 || oheight >= 1 || odepth >= 1,
          "input (D: ", idepth, " H: ", iheight, ", W: ", iwidth,
          ") is too small."
          " Calculated output D: ", odepth, " H: ", oheight, " W: ", owidth);
        */
}

lazy_static!{
    /*
    TORCH_META_FUNC(replication_pad3d) (
      const Tensor& input, IntArrayRef paddingSize
    ) {
      TORCH_CHECK(paddingSize.size() == 6, "padding size is expected to be 6");
      i64 pleft = paddingSize[0];
      i64 pright = paddingSize[1];
      i64 ptop = paddingSize[2];
      i64 pbottom = paddingSize[3];
      i64 pfront = paddingSize[4];
      i64 pback = paddingSize[5];
      i64 dimw = 3;
      i64 dimh = 2;
      i64 dimd = 1;
      i64 dimslices = 0;
      i64 nbatch = 1;

      shapeCheck3d(input, pleft, pright, ptop, pbottom, pfront, pback);

      if (input.dim() == 5)
      {
        nbatch = input.size(0);
        dimw++;
        dimh++;
        dimd++;
        dimslices++;
      }

      /* sizes */
      i64 nslices = input.size(dimslices);
      i64 idepth = input.size(dimd);
      i64 iheight = input.size(dimh);
      i64 iwidth = input.size(dimw);
      i64 odepth = idepth + pfront + pback;
      i64 oheight = iheight + ptop + pbottom;
      i64 owidth  = iwidth + pleft + pright;

      /* resize output */
      if (input.dim() == 4) {
        set_output({nslices, odepth, oheight, owidth}, input.options());
      } else {
        set_output({nbatch, nslices, odepth, oheight, owidth}, input.options());
      }
    }
    */
}

pub fn replication_pad1d_out_frame<Scalar>(
    input_p:  *mut Scalar,
    output_p: *mut Scalar,
    nslices:  i64,
    iwidth:   i64,
    owidth:   i64,
    pad_l:    i32,
    pad_r:    i32)  {

    todo!();
        /*
            int iStartX = max(0, -pad_l);
      int oStartX = max(0, pad_l);

      parallel_for(0, nslices, 0, [&](i64 start, i64 end) {
        long ip_x;
        for (auto k = start; k < end; k++)
        {
          for (long j = 0; j < owidth; j++) {
            if (j < pad_l) {
              ip_x = pad_l;
            } else if (j >= pad_l && j < iwidth + pad_l) {
              ip_x = j;
            } else {
              ip_x = iwidth + pad_l - 1;
            }
            ip_x = ip_x - oStartX + iStartX;

            Scalar *dest_p = output_p + k*owidth + j;
            Scalar *src_p = input_p + k*iwidth + ip_x;
            *dest_p = *src_p;
          }
        }
      });
        */
}



pub fn replication_pad1d_out_batch<Scalar>(
        input_data:  *mut Scalar,
        output_data: *mut Scalar,
        nslices:     i64,
        iwidth:      i64,
        owidth:      i64,
        pad_l:       i32,
        pad_r:       i32,
        nbatch:      i32)  {

    todo!();
        /*
            parallel_for(0, nbatch, 0, [&](i64 start, i64 end) {
        for (auto p = start; p < end; p++)
        {
          Scalar *input_p = input_data+p*nslices*iwidth;
          Scalar *output_p = output_data+p*nslices*owidth;
          replication_pad1d_out_frame(input_p, output_p, nslices, iwidth, owidth, pad_l, pad_r);
        }
      });
        */
}



pub fn replication_pad1d_backward_out_frame<Scalar>(
        ginput_p:  *mut Scalar,
        goutput_p: *mut Scalar,
        nslices:   i64,
        iwidth:    i64,
        owidth:    i64,
        pad_l:     i32,
        pad_r:     i32)  {

    todo!();
        /*
            int iStartX = max(0, -pad_l);
      int oStartX = max(0, pad_l);

      parallel_for(0, nslices, 0, [&](i64 start, i64 end) {
        long ip_x;
        for (auto k = start; k < end; k++)
        {
          for (long j = 0; j < owidth; j++) {
            if (j < pad_l) {
              ip_x = pad_l;
            } else if (j >= pad_l && j < iwidth + pad_l) {
              ip_x = j;
            } else {
              ip_x = iwidth + pad_l - 1;
            }
            ip_x = ip_x - oStartX + iStartX;

            Scalar *src_p = goutput_p + k*owidth + j;
            Scalar *dest_p = ginput_p + k*iwidth + ip_x;
            *dest_p += *src_p;
          }
        }
      });
        */
}



pub fn replication_pad1d_backward_out_batch<Scalar>(
        ginput_data:  *mut Scalar,
        goutput_data: *mut Scalar,
        nslices:      i64,
        iwidth:       i64,
        owidth:       i64,
        pad_l:        i32,
        pad_r:        i32,
        nbatch:       i32)  {

    todo!();
        /*
            parallel_for(0, nbatch, 0, [&](i64 start, i64 end) {
        for (auto p = start; p < end; p++)
        {
          Scalar *ginput_p = ginput_data + p * nslices * iwidth;
          Scalar *goutput_p = goutput_data + p * nslices * owidth;
          replication_pad1d_backward_out_frame(ginput_p, goutput_p,
            nslices, iwidth, owidth, pad_l, pad_r);
        }
      });
        */
}



pub fn replication_pad2d_out_frame<Scalar>(
        input_p:  *mut Scalar,
        output_p: *mut Scalar,
        nslices:  i64,
        iwidth:   i64,
        iheight:  i64,
        owidth:   i64,
        oheight:  i64,
        pad_l:    i32,
        pad_r:    i32,
        pad_t:    i32,
        pad_b:    i32)  {

    todo!();
        /*
            int iStartX = max(0, -pad_l);
      int iStartY = max(0, -pad_t);
      int oStartX = max(0, pad_l);
      int oStartY = max(0, pad_t);

      parallel_for(0, nslices, 0, [&](i64 start, i64 end) {
        i64 ip_x, ip_y;
        for (auto k = start; k < end; k++)
        {
          for (i64 i = 0; i < oheight; i++) {
            for (i64 j = 0; j < owidth; j++) {
              if (j < pad_l) {
                ip_x = pad_l;
              } else if (j >= pad_l && j < iwidth + pad_l) {
                ip_x = j;
              } else {
                ip_x = iwidth + pad_l - 1;
              }
              ip_x = ip_x - oStartX + iStartX;

              if (i < pad_t) {
                ip_y = pad_t;
              } else if (i >= pad_t && i < iheight + pad_t) {
                ip_y = i;
              } else {
                ip_y = iheight + pad_t - 1;
              }
              ip_y = ip_y - oStartY + iStartY;

              Scalar *dest_p = output_p + k*owidth*oheight + i * owidth + j;
              Scalar *src_p = input_p + k*iwidth*iheight + ip_y * iwidth + ip_x;
              *dest_p = *src_p;
            }
          }
        }
      });
        */
}



pub fn replication_pad2d_out_batch<Scalar>(
        input_data:  *mut Scalar,
        output_data: *mut Scalar,
        nslices:     i64,
        iwidth:      i64,
        iheight:     i64,
        owidth:      i64,
        oheight:     i64,
        pad_l:       i32,
        pad_r:       i32,
        pad_t:       i32,
        pad_b:       i32,
        nbatch:      i32)  {

    todo!();
        /*
            parallel_for(0, nbatch, 0, [&](i64 start, i64 end) {
        for (auto p = start; p < end; p++)
        {
          Scalar *input_p = input_data+p*nslices*iwidth*iheight;
          Scalar *output_p = output_data+p*nslices*owidth*oheight;
          replication_pad2d_out_frame(input_p, output_p, nslices,
              iwidth, iheight, owidth, oheight, pad_l, pad_r, pad_t, pad_b);
        }
      });
        */
}



pub fn replication_pad2d_backward_out_frame<Scalar>(
        ginput_p:  *mut Scalar,
        goutput_p: *mut Scalar,
        nslices:   i64,
        iwidth:    i64,
        iheight:   i64,
        owidth:    i64,
        oheight:   i64,
        pad_l:     i32,
        pad_r:     i32,
        pad_t:     i32,
        pad_b:     i32)  {

    todo!();
        /*
            int iStartX = max(0, -pad_l);
      int iStartY = max(0, -pad_t);
      int oStartX = max(0, pad_l);
      int oStartY = max(0, pad_t);

      parallel_for(0, nslices, 0, [&](i64 start, i64 end) {
        i64 ip_x, ip_y;
        for (auto k = start; k < end; k++)
        {
          for (i64 i = 0; i < oheight; i++) {
            for (i64 j = 0; j < owidth; j++) {
              if (j < pad_l) {
                ip_x = pad_l;
              } else if (j >= pad_l && j < iwidth + pad_l) {
                ip_x = j;
              } else {
                ip_x = iwidth + pad_l - 1;
              }
              ip_x = ip_x - oStartX + iStartX;

              if (i < pad_t) {
                ip_y = pad_t;
              } else if (i >= pad_t && i < iheight + pad_t) {
                ip_y = i;
              } else {
                ip_y = iheight + pad_t - 1;
              }
              ip_y = ip_y - oStartY + iStartY;

              Scalar *src_p = goutput_p + k*owidth*oheight + i * owidth + j;
              Scalar *dest_p = ginput_p + k*iwidth*iheight + ip_y * iwidth + ip_x;
              *dest_p += *src_p;
            }
          }
        }
      });
        */
}



pub fn replication_pad2d_backward_out_batch<Scalar>(
        ginput_data:  *mut Scalar,
        goutput_data: *mut Scalar,
        nslices:      i64,
        iwidth:       i64,
        iheight:      i64,
        owidth:       i64,
        oheight:      i64,
        pad_l:        i32,
        pad_r:        i32,
        pad_t:        i32,
        pad_b:        i32,
        nbatch:       i32)  {

    todo!();
        /*
            parallel_for(0, nbatch, 0, [&](i64 start, i64 end) {
        for (auto p = start; p < end; p++)
        {
          Scalar *ginput_p = ginput_data + p * nslices * iheight * iwidth;
          Scalar *goutput_p = goutput_data + p * nslices * oheight * owidth;
          replication_pad2d_backward_out_frame(ginput_p, goutput_p, nslices,
              iwidth, iheight, owidth, oheight, pad_l, pad_r, pad_t, pad_b);
        }
      });
        */
}


pub fn replication_pad2d_backward_out_cpu_template(
        grad_input:   &mut Tensor,
        grad_output:  &Tensor,
        input:        &Tensor,
        padding_size: &[i32]) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(paddingSize.size() == 4, "padding size is expected to be 4");
      int pad_l = paddingSize[0];
      int pad_r = paddingSize[1];
      int pad_t = paddingSize[2];
      int pad_b = paddingSize[3];
      int dimw = 2;
      int dimh = 1;
      int dimslices = 0;
      i64 nbatch = 1;

      if (input.dim() == 4)
      {
        nbatch = input.size(0);
        dimw++;
        dimh++;
        dimslices++;
      }

      /* sizes */
      i64 nslices = input.size(dimslices);
      i64 iheight = input.size(dimh);
      i64 iwidth = input.size(dimw);
      i64 oheight = iheight + pad_t + pad_b;
      i64 owidth  = iwidth + pad_l + pad_r;

      TORCH_CHECK(owidth == gradOutput_.size(dimw),
          "gradOutput width unexpected. Expected: ", owidth, ", Got: ",
          gradOutput_.size(dimw));
      TORCH_CHECK(oheight == gradOutput_.size(dimh),
          "gradOutput height unexpected. Expected: ", oheight, ", Got: ",
          gradOutput_.size(dimh));

      /* get contiguous gradOutput */
      auto gradOutput = gradOutput_.contiguous();

      /* resize */
      gradInput.resize_as_(input);
      if (gradInput.numel() == 0) {
        return gradInput;
      }

      gradInput.zero_();

      /* backprop */
      if (input.dim() == 3)
      {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
          input.scalar_type(), "replication_pad2d_backward_cpu", [&] {
          replication_pad2d_backward_out_frame<Scalar>(
            gradInput.data_ptr<Scalar>(),
            gradOutput.data_ptr<Scalar>(),
            nslices,
            iwidth, iheight,
            owidth, oheight,
            pad_l, pad_r,
            pad_t, pad_b);
          }
        );
      }
      else
      {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
          input.scalar_type(), "replication_pad2d_backward_cpu", [&] {
          replication_pad2d_backward_out_batch<Scalar>(
            gradInput.data_ptr<Scalar>(),
            gradOutput.data_ptr<Scalar>(),
            nslices,
            iwidth, iheight,
            owidth, oheight,
            pad_l, pad_r,
            pad_t, pad_b,
            nbatch);
          }
        );
      }
      return gradInput;
        */
}



pub fn replication_pad3d_out_frame<Scalar>(
        input_p:  *mut Scalar,
        output_p: *mut Scalar,
        nslices:  i64,
        iwidth:   i64,
        iheight:  i64,
        idepth:   i64,
        owidth:   i64,
        oheight:  i64,
        odepth:   i64,
        pleft:    i32,
        pright:   i32,
        ptop:     i32,
        pbottom:  i32,
        pfront:   i32,
        pback:    i32)  {

    todo!();
        /*
            int iStartX = max(0, -pleft);
      int iStartY = max(0, -ptop);
      int iStartZ = max(0, -pfront);
      int oStartX = max(0, pleft);
      int oStartY = max(0, ptop);
      int oStartZ = max(0, pfront);

      parallel_for(0, nslices, 0, [&](i64 start, i64 end) {
        i64 ip_x, ip_y, ip_z;
        for (auto k = start; k < end; k++) {
          for (i64 z = 0; z < odepth; z++) {
            for (i64 i = 0; i < oheight; i++) {
              for (i64 j = 0; j < owidth; j++) {
                if (j < pleft) {
                  ip_x = pleft;
                } else if (j >= pleft && j < iwidth + pleft) {
                  ip_x = j;
                } else {
                  ip_x = iwidth + pleft - 1;
                }
                ip_x = ip_x - oStartX + iStartX;

                if (i < ptop) {
                  ip_y = ptop;
                } else if (i >= ptop && i < iheight + ptop) {
                  ip_y = i;
                } else {
                  ip_y = iheight + ptop - 1;
                }
                ip_y = ip_y - oStartY + iStartY;

                if (z < pfront) {
                  ip_z = pfront;
                } else if (z >= pfront && z < idepth + pfront) {
                  ip_z = z;
                } else {
                  ip_z = idepth + pfront - 1;
                }
                ip_z = ip_z - oStartZ + iStartZ;

                Scalar *dest_p = output_p + k * owidth * oheight * odepth +
                  z * owidth * oheight + i * owidth + j;
                Scalar *src_p = input_p + k * iwidth * iheight * idepth +
                  ip_z * iwidth * iheight + ip_y * iwidth + ip_x;
                *dest_p = *src_p;
              }
            }
          }
        }
      });
        */
}



pub fn replication_pad3d_out_batch<Scalar>(
        input_data:  *mut Scalar,
        output_data: *mut Scalar,
        nslices:     i64,
        iwidth:      i64,
        iheight:     i64,
        idepth:      i64,
        owidth:      i64,
        oheight:     i64,
        odepth:      i64,
        pleft:       i32,
        pright:      i32,
        ptop:        i32,
        pbottom:     i32,
        pfront:      i32,
        pback:       i32,
        nbatch:      i32)  {

    todo!();
        /*
            parallel_for(0, nbatch, 0, [&](i64 start, i64 end) {
        for (auto p = start; p < end; p++)
        {
          Scalar *input_p = input_data + p * nslices * iwidth * iheight * idepth;
          Scalar *output_p = output_data + p * nslices * owidth * oheight * odepth;
          replication_pad3d_out_frame(input_p, output_p, nslices,
              iwidth, iheight, idepth, owidth, oheight, odepth,
              pleft, pright, ptop, pbottom, pfront, pback);
        }
      });
        */
}



pub fn replication_pad3d_backward_out_frame<Scalar>(
        ginput_p:  *mut Scalar,
        goutput_p: *mut Scalar,
        nslices:   i64,
        iwidth:    i64,
        iheight:   i64,
        idepth:    i64,
        owidth:    i64,
        oheight:   i64,
        odepth:    i64,
        pleft:     i32,
        pright:    i32,
        ptop:      i32,
        pbottom:   i32,
        pfront:    i32,
        pback:     i32)  {

    todo!();
        /*
            int iStartX = max(0, -pleft);
      int iStartY = max(0, -ptop);
      int iStartZ = max(0, -pfront);
      int oStartX = max(0, pleft);
      int oStartY = max(0, ptop);
      int oStartZ = max(0, pfront);

      parallel_for(0, nslices, 0, [&](i64 start, i64 end) {
        i64 ip_x, ip_y, ip_z;
        for (auto k = start; k < end; k++) {
          for (i64 z = 0; z < odepth; z++) {
            for (i64 i = 0; i < oheight; i++) {
              for (i64 j = 0; j < owidth; j++) {
                if (j < pleft) {
                  ip_x = pleft;
                } else if (j >= pleft && j < iwidth + pleft) {
                  ip_x = j;
                } else {
                  ip_x = iwidth + pleft - 1;
                }
                ip_x = ip_x - oStartX + iStartX;

                if (i < ptop) {
                  ip_y = ptop;
                } else if (i >= ptop && i < iheight + ptop) {
                  ip_y = i;
                } else {
                  ip_y = iheight + ptop - 1;
                }
                ip_y = ip_y - oStartY + iStartY;

                if (z < pfront) {
                  ip_z = pfront;
                } else if (z >= pfront && z < idepth + pfront) {
                  ip_z = z;
                } else {
                  ip_z = idepth + pfront - 1;
                }
                ip_z = ip_z - oStartZ + iStartZ;

                Scalar *src_p = goutput_p + k * owidth * oheight * odepth +
                  z * owidth * oheight + i * owidth + j;
                Scalar *dest_p = ginput_p + k * iwidth * iheight * idepth +
                  ip_z * iwidth * iheight + ip_y * iwidth + ip_x;
                *dest_p += *src_p;
              }
            }
          }
        }
      });
        */
}



pub fn replication_pad3d_backward_out_batch<Scalar>(
        ginput_data:  *mut Scalar,
        goutput_data: *mut Scalar,
        nslices:      i64,
        iwidth:       i64,
        iheight:      i64,
        idepth:       i64,
        owidth:       i64,
        oheight:      i64,
        odepth:       i64,
        pleft:        i32,
        pright:       i32,
        ptop:         i32,
        pbottom:      i32,
        pfront:       i32,
        pback:        i32,
        nbatch:       i32)  {

    todo!();
        /*
            parallel_for(0, nbatch, 0, [&](i64 start, i64 end) {
        for (auto p = start; p < end; p++)
        {
          Scalar *ginput_p = ginput_data + p * nslices * idepth * iheight * iwidth;
          Scalar *goutput_p = goutput_data + p * nslices * odepth * oheight * owidth;
          replication_pad3d_backward_out_frame(ginput_p, goutput_p, nslices,
              iwidth, iheight, idepth, owidth, oheight, odepth,
              pleft, pright, ptop, pbottom, pfront, pback);
        }
      });
        */
}


pub fn replication_pad3d_backward_out_cpu_template(
        grad_input:   &mut Tensor,
        grad_output:  &Tensor,
        input:        &Tensor,
        padding_size: &[i32]) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(paddingSize.size() == 6, "padding size is expected to be 6");
      int pleft = paddingSize[0];
      int pright = paddingSize[1];
      int ptop = paddingSize[2];
      int pbottom = paddingSize[3];
      int pfront = paddingSize[4];
      int pback = paddingSize[5];
      int dimw = 3;
      int dimh = 2;
      int dimd = 1;
      int dimslices = 0;
      i64 nbatch = 1;

      if (input.dim() == 5)
      {
        nbatch = input.size(0);
        dimw++;
        dimh++;
        dimd++;
        dimslices++;
      }

      /* sizes */
      i64 nslices = input.size(dimslices);
      i64 idepth = input.size(dimd);
      i64 iheight = input.size(dimh);
      i64 iwidth = input.size(dimw);
      i64 odepth = idepth + pfront + pback;
      i64 oheight = iheight + ptop + pbottom;
      i64 owidth  = iwidth + pleft + pright;

      shapeCheck3d(input, pleft, pright,
          ptop, pbottom, pfront, pback);

      /* get contiguous gradOutput */
      auto gradOutput = gradOutput_.contiguous();

      /* resize */
      gradInput.resize_as_(input);
      if (gradInput.numel() == 0) {
        return gradInput;
      }
      gradInput.zero_();

      /* backprop */
      if (input.dim() == 4)
      {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
          input.scalar_type(), "replication_pad3d_backward_cpu", [&] {
          replication_pad3d_backward_out_frame<Scalar> (
            gradInput.data_ptr<Scalar>(),
            gradOutput.data_ptr<Scalar>(),
            nslices,
            iwidth, iheight, idepth,
            owidth, oheight, odepth,
            pleft, pright,
            ptop, pbottom,
            pfront, pback);
          }
        );
      }
      else
      {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
          input.scalar_type(), "replication_pad3d_backward_cpu", [&] {
          replication_pad3d_backward_out_batch<Scalar> (
            gradInput.data_ptr<Scalar>(),
            gradOutput.data_ptr<Scalar>(),
            nslices,
            iwidth, iheight, idepth,
            owidth, oheight, odepth,
            pleft, pright,
            ptop, pbottom,
            pfront, pback,
            nbatch);
          }
        );
      }
      return gradInput;
        */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(replication_pad1d_out_cpu) (
      const Tensor& input_, IntArrayRef paddingSize, const Tensor& output
    ) {
      constexpr i64 dimw = -1;
      constexpr i64 dimslices = -2;

      i64 pad_l = paddingSize[0];
      i64 pad_r = paddingSize[1];

      /* get contiguous input */
      auto input = input_.contiguous();

      i64 nbatch = 1;
      if (input.ndimension() == 3) {
        nbatch = input.size(0);
      }

      /* sizes */
      long nslices = input.size(dimslices);
      long iwidth = input.size(dimw);
      long owidth = output.size(dimw);

      if (input.ndimension() == 2)
      {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "replication_pad1d_cpu", [&] {
          auto input_data = input.data_ptr<Scalar>();
          auto output_data = output.data_ptr<Scalar>();
          replication_pad1d_out_frame<Scalar>(
            input_data,
            output_data,
            nslices,
            iwidth,
            owidth,
            pad_l, pad_r);
          }
        );
      }
      else
      {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "replication_pad1d_cpu", [&] {
          auto input_data = input.data_ptr<Scalar>();
          auto output_data = output.data_ptr<Scalar>();
          replication_pad1d_out_batch<Scalar>(
            input_data,
            output_data,
            nslices,
            iwidth,
            owidth,
            pad_l, pad_r,
            nbatch);
          }
        );
      }
    }
    */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(replication_pad1d_backward_out_cpu) (
      const Tensor& gradOutput_, const Tensor& input, IntArrayRef paddingSize, const Tensor& gradInput
    ) {
      i64 dimw = 1;
      i64 dimslices = 0;
      i64 nbatch = 1;
      i64 pad_l = paddingSize[0];
      i64 pad_r = paddingSize[1];

      if (input.ndimension() == 3)
      {
        nbatch = input.size(0);
        dimw++;
        dimslices++;
      }

      /* get contiguous gradOutput */
      auto gradOutput = gradOutput_.contiguous();

      /* sizes */
      i64 nslices = input.size(dimslices);
      i64 iwidth  = input.size(dimw);
      i64 owidth  = gradOutput.size(dimw);

      TORCH_CHECK(owidth == gradOutput.size(dimw),
          "gradOutput width unexpected. Expected: ", owidth,
          " Got: ", gradOutput_.size(dimw));

      if (gradInput.numel() == 0) {
        return;
      }

      gradInput.zero_();

      /* backprop */
      if (input.ndimension() == 2)
      {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
          input.scalar_type(), "replication_pad1d_backward_cpu", [&] {
          Scalar *gradInput_data = gradInput.data_ptr<Scalar>();
          Scalar *gradOutput_data = gradOutput.data_ptr<Scalar>();

          replication_pad1d_backward_out_frame<Scalar> (
            gradInput_data,
            gradOutput_data,
            nslices,
            iwidth,
            owidth,
            pad_l, pad_r);
          }
        );
      }
      else
      {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
          input.scalar_type(), "replication_pad1d_backward_cpu", [&] {
          Scalar *gradInput_data = gradInput.data_ptr<Scalar>();
          Scalar *gradOutput_data = gradOutput.data_ptr<Scalar>();

          replication_pad1d_backward_out_batch<Scalar> (
            gradInput_data,
            gradOutput_data,
            nslices,
            iwidth,
            owidth,
            pad_l, pad_r,
            nbatch);
          }
        );
      }
    }
    */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(replication_pad2d_out_cpu) (
      const Tensor& input_, IntArrayRef paddingSize, const Tensor& output
    ) {
      i64 pad_l = paddingSize[0];
      i64 pad_r = paddingSize[1];
      i64 pad_t = paddingSize[2];
      i64 pad_b = paddingSize[3];
      i64 dimw = 2;
      i64 dimh = 1;
      i64 dimslices = 0;
      i64 nbatch = 1;
      if (input_.dim() == 4) {
        nbatch = input_.size(0);
        dimw++;
        dimh++;
        dimslices++;
      }

      i64 nslices = input_.size(dimslices);
      i64 iheight = input_.size(dimh);
      i64 iwidth = input_.size(dimw);
      i64 oheight = iheight + pad_t + pad_b;
      i64 owidth  = iwidth + pad_l + pad_r;

      /* get contiguous input */
      auto input = input_.contiguous();

      /* resize output */
      if (input.dim() == 3)
      {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "replication_pad2d_cpu", [&] {
          auto input_data = input.data_ptr<Scalar>();
          auto output_data = output.data_ptr<Scalar>();
          replication_pad2d_out_frame<Scalar> (input_data, output_data,
            nslices,
            iwidth, iheight,
            owidth, oheight,
            pad_l, pad_r,
            pad_t, pad_b);
          }
        );
      }
      else
      {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "replication_pad2d_cpu", [&] {
          auto input_data = input.data_ptr<Scalar>();
          auto output_data = output.data_ptr<Scalar>();
          replication_pad2d_out_batch<Scalar> (input_data, output_data,
            nslices,
            iwidth, iheight,
            owidth, oheight,
            pad_l, pad_r,
            pad_t, pad_b,
            nbatch);
          }
        );
      }
    }
    */
}


pub fn replication_pad2d_backward_out_cpu(
        grad_output:  &Tensor,
        input:        &Tensor,
        padding_size: &[i32],
        grad_input:   &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            replication_pad2d_backward_out_cpu_template(
          gradInput, gradOutput, input, paddingSize);
      return gradInput;
        */
}


pub fn replication_pad2d_backward_cpu(
        grad_output:  &Tensor,
        input:        &Tensor,
        padding_size: &[i32]) -> Tensor {
    
    todo!();
        /*
            auto gradInput = zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      replication_pad2d_backward_out_cpu_template(
          gradInput, gradOutput, input, paddingSize);
      return gradInput;
        */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(replication_pad3d_out_cpu) (
      const Tensor& input_, IntArrayRef paddingSize, const Tensor& output
    ) {
      i64 pleft = paddingSize[0];
      i64 pright = paddingSize[1];
      i64 ptop = paddingSize[2];
      i64 pbottom = paddingSize[3];
      i64 pfront = paddingSize[4];
      i64 pback = paddingSize[5];
      i64 dimw = 3;
      i64 dimh = 2;
      i64 dimd = 1;
      i64 dimslices = 0;
      i64 nbatch = 1;

      /* get contiguous input */
      auto input = input_.contiguous();

      if (input.dim() == 5) {
        nbatch = input.size(0);
        dimw++;
        dimh++;
        dimd++;
        dimslices++;
      }

      /* sizes */
      i64 nslices = input.size(dimslices);
      i64 idepth  = input.size(dimd);
      i64 iheight = input.size(dimh);
      i64 iwidth  = input.size(dimw);
      i64 odepth  = output.size(dimd);
      i64 oheight = output.size(dimh);
      i64 owidth  = output.size(dimw);

      /* resize output */
      if (input.dim() == 4) {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "replication_pad3d_cpu", [&] {
          auto input_data = input.data_ptr<Scalar>();
          auto output_data = output.data_ptr<Scalar>();
          replication_pad3d_out_frame<Scalar>(
            input_data, output_data, nslices, iwidth, iheight, idepth,
            owidth, oheight, odepth, pleft, pright, ptop, pbottom, pfront,
            pback);
          }
        );
      }
      else
      {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "replication_pad3d_cpu", [&] {
          auto input_data = input.data_ptr<Scalar>();
          auto output_data = output.data_ptr<Scalar>();
          replication_pad3d_out_batch<Scalar>(
            input_data, output_data, nslices, iwidth, iheight, idepth,
            owidth, oheight, odepth, pleft, pright, ptop, pbottom, pfront,
            pback,
            nbatch);
          }
        );
      }
    }
    */
}

pub fn replication_pad3d_backward_out_cpu(
        grad_output:  &Tensor,
        input:        &Tensor,
        padding_size: &[i32],
        grad_input:   &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            replication_pad3d_backward_out_cpu_template(
          gradInput, gradOutput, input, paddingSize);
      return gradInput;
        */
}

pub fn replication_pad3d_backward_cpu(
        grad_output:  &Tensor,
        input:        &Tensor,
        padding_size: &[i32]) -> Tensor {
    
    todo!();
        /*
            auto gradInput = zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      replication_pad3d_backward_out_cpu_template(
          gradInput, gradOutput, input, paddingSize);
      return gradInput;
        */
}
