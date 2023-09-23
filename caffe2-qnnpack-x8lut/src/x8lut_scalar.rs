crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/x8lut/scalar.c]

pub fn pytorch_x8lut_ukernel_scalar(
        n: usize,
        x: *const u8,
        t: [u8; 256],
        y: *mut u8)  {
    
    todo!();
        /*
            assert(n != 0);

      while (n >= 4) {
        const usize vx0 = x[0];
        const usize vx1 = x[1];
        const usize vx2 = x[2];
        const usize vx3 = x[3];
        x += 4;

        const u8 vt0 = t[vx0];
        const u8 vt1 = t[vx1];
        const u8 vt2 = t[vx2];
        const u8 vt3 = t[vx3];

        y[0] = vt0;
        y[1] = vt1;
        y[2] = vt2;
        y[3] = vt3;
        y += 4;

        n -= 4;
      }
      while (n != 0) {
        const usize vx = *x++;
        const u8 vt = t[vx];
        *y++ = vt;

        n--;
      };
        */
}
