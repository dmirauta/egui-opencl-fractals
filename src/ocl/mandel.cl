
#ifndef EXTERNAL_CONCAT
    #include "mandelutils.c"
#endif

__kernel void apply_log_int(__global int *res_g)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    res_g[i*M+j] = log((float) res_g[i*M+j]);
}

__kernel void apply_log_fpn(__global FPN *res_g)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    res_g[i*M+j] = log((float) res_g[i*M+j]);
}

__kernel void escape_iter(__global int *res_g,
                          FParam_t param)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    Complex_t p = {param.view_rect.left + j*(param.view_rect.right-param.view_rect.left)/M,
                   param.view_rect.bot  + i*(param.view_rect.top  -param.view_rect.bot )/N};

    Complex_t _c = param.mandel ? p : param.c;

    res_g[i*M+j] = _escape_iter(p, _c, param.MAXITER);
}

__kernel void escape_iter_fpn(__global FPN *res_g,
                              FParam_t param)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    Complex_t p = {param.view_rect.left + j*(param.view_rect.right-param.view_rect.left)/M,
                   param.view_rect.bot  + i*(param.view_rect.top  -param.view_rect.bot )/N};

    Complex_t _c = param.mandel ? p : param.c;

    res_g[i*M+j] = ((FPN) _escape_iter(p, _c, param.MAXITER))/((FPN) param.MAXITER);
}

__kernel void min_prox(__global FPN *res_g,
                       FParam_t param,
                       ProxType_t PROXTYPE)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    Complex_t p = {param.view_rect.left + j*(param.view_rect.right-param.view_rect.left)/M,
                   param.view_rect.bot  + i*(param.view_rect.top  -param.view_rect.bot )/N};

    Complex_t _c = param.mandel ? p : param.c;

    res_g[i*M+j] = _minprox(p, _c, param.MAXITER, PROXTYPE);
}

__kernel void orbit_trap(__global Complex_t *res_g,
                         __global FParam_t  *param,
                         __global Box_t     *trap)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    Complex_t p = {param->view_rect.left + j*(param->view_rect.right-param->view_rect.left)/M,
                   param->view_rect.bot  + i*(param->view_rect.top  -param->view_rect.bot )/N};

    Complex_t _c = param->mandel ? p : param->c;

    res_g[i*M+j] = _orbit_trap(p, _c, *trap, param->MAXITER);
}

__kernel void orbit_trap_re(__global FPN       *res_g,
                            FParam_t  param,
                            Box_t     trap)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    Complex_t p = {param.view_rect.left + j*(param.view_rect.right-param.view_rect.left)/M,
                   param.view_rect.bot  + i*(param.view_rect.top  -param.view_rect.bot )/N};

    Complex_t _c = param.mandel ? p : param.c;

    res_g[i*M+j] = _orbit_trap(p, _c, trap, param.MAXITER).re;
}

__kernel void orbit_trap_im(__global FPN       *res_g,
                            FParam_t  param,
                            Box_t     trap)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    Complex_t p = {param.view_rect.left + j*(param.view_rect.right-param.view_rect.left)/M,
                   param.view_rect.bot  + i*(param.view_rect.top  -param.view_rect.bot )/N};

    Complex_t _c = param.mandel ? p : param.c;

    res_g[i*M+j] = _orbit_trap(p, _c, trap, param.MAXITER).im;
}

__kernel void map_img   (__global Complex_t *res_g, // result of orbit trap
                         __global Pixel_t   *sim_g, // sample image
                         __global Pixel_t   *mim_g, // mapped image
                         __global ImDims_t  *dims)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    int _i = (int) ( ((FPN) (dims->imH-1)) * res_g[i*M+j].im );
    int _j = (int) ( ((FPN) (dims->imW-1)) * res_g[i*M+j].re );

    mim_g[i*M+j] = sim_g[_i*dims->imW + _j];

}

__kernel void map_img2  (__global FPN     *res1_g,
                         __global FPN     *res2_g,
                         __global Pixel_t   *sim_g, // sample image
                         __global Pixel_t   *mim_g, // mapped image
                         ImDims_t  dims)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    // expecting res in [0, 1) range
    int _i = min((int) ( ((FPN) dims.imH) * res1_g[i*M+j] ), dims.imH-1);
    int _j = min((int) ( ((FPN) dims.imW) * res2_g[i*M+j] ), dims.imW-1);

    mim_g[i*M+j] = sim_g[_i*dims.imW + _j];

}

//      u
//     <-> ta
// tl * ___*__ * tr 
//    |    |   |    |v
//    |    *   |    
//    |    |   |
// bl *____*___* br
//         ba

inline FPN blinterp_f(FPN tl, FPN tr, 
                      FPN bl, FPN br, 
                      FPN u, FPN v) {
    FPN ui = 1.0-u;
    FPN vi = 1.0-v;
    FPN ta = ui*tl + u*tr;
    FPN ba = ui*bl + u*br;
    return vi*ta + v*ba;
}

inline Pixel_t blinterp(Pixel_t tl, Pixel_t tr, 
                        Pixel_t bl, Pixel_t br, 
                        FPN u, FPN v) {
    return (Pixel_t){ blinterp_f(tl.r, tr.r, bl.r, br.r, u, v),
                      blinterp_f(tl.g, tr.g, bl.g, br.g, u, v),
                      blinterp_f(tl.b, tr.b, bl.b, br.b, u, v) };
}

__kernel void map_img3  (__global FPN     *res1_g,
                         __global FPN     *res2_g,
                         __global Pixel_t   *sim_g, // sample image
                         __global Pixel_t   *mim_g, // mapped image
                         ImDims_t  dims)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    FPN fH = dims.imH;
    FPN fW = dims.imW;

    // expecting res in [0, 1) range
    FPN fi = fH * res1_g[i*M+j];
    FPN fj = fW * res2_g[i*M+j];
    fi = min(fi, fH - 2.0);
    fj = min(fj, fW - 2.0);
    // discard fractional part
    int _i = fi;
    int _j = fj;

    mim_g[i*M+j] = blinterp(sim_g[    _i*dims.imW + _j], sim_g[    _i*dims.imW + _j+1],
                            sim_g[(_i+1)*dims.imW + _j], sim_g[(_i+1)*dims.imW + _j+1],
                            fi - _i, fj - _j);
}

__kernel void pack (__global FPN     *res1_g,
                    __global FPN     *res2_g,
                    __global FPN     *res3_g,
                    __global Pixel_t *img_g)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    img_g[i*M+j] = (Pixel_t){255*res1_g[i*M+j], 255*res2_g[i*M+j], 255*res3_g[i*M+j]};

}

__kernel void pack_norm(__global FPN     *res1_g,
                        __global FPN     *res2_g,
                        __global FPN     *res3_g,
                        __global Pixel_t *img_g)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    FPN s = res1_g[i*M+j]+res2_g[i*M+j]+res3_g[i*M+j];
    img_g[i*M+j] = (Pixel_t){255*res1_g[i*M+j]/s, 255*res2_g[i*M+j]/s, 255*res3_g[i*M+j]/s};

}

__kernel void map_sines(__global FPN     *res_g,
                        __global Pixel_t *img_g,
                        Freqs_t freqs)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    int fi = i*M + j;

    img_g[fi] = (Pixel_t){127*(sin(res_g[fi]*freqs.f1)+1),
                          127*(sin(res_g[fi]*freqs.f2)+1),
                          127*(sin(res_g[fi]*freqs.f3)+1)};

}
