crate::ix!();

use crate::CPUContext;

/// Z=X*Y
#[inline] pub fn dot<XT, YT, ZT>(
    n:      i32,
    x:      *const XT,
    y:      *const YT,
    z:      *mut ZT,
    ctx:    *mut CPUContext)
{
    todo!();
    /*
        CAFFE_THROW("Unsupported, see specialized implementations");
    */
}

//TODO: can specialize the above generic
#[inline] pub fn dot_float_x3(n: i32, x: *const f32, y: *const f32, z: *mut f32, ctx: *mut CPUContext)  {
    
    todo!();
    /*
        math::Dot<float, CPUContext>(N, x, y, z, ctx);
    */
}

//TODO: can specialize the above generic
#[inline] pub fn dot_float_half_float(
    n:     i32,
    x:     *const f32,
    y:     *const f16,
    z:     *mut f32,
    ctx:   *mut CPUContext)  
{
    todo!();
    /*
        #ifdef _MSC_VER
      std::vector<float> tmp_y_vec(N);
      float* tmp_y = tmp_y_vec.data();
    #else
      float tmp_y[N];
    #endif
      for (int i = 0; i < N; i++) {
    #ifdef __F16C__
        tmp_y[i] = _cvtss_sh(y[i], 0); // TODO: vectorize
    #else
        tmp_y[i] = y[i];
    #endif
      }
      math::Dot<float, CPUContext>(N, x, tmp_y, z, ctx);
    */
}
