
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
                       __global FParam_t *param,
                       __global      int *PROXTYPE)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    Complex_t p = {param->view_rect.left + j*(param->view_rect.right-param->view_rect.left)/M,
                   param->view_rect.bot  + i*(param->view_rect.top  -param->view_rect.bot )/N};

    Complex_t _c = param->mandel ? p : param->c;

    res_g[i*M+j] = _minprox(p, _c, param->MAXITER, *PROXTYPE);
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

    res_g[i*M+j] = _orbit_trap(p, _c, *trap, param->MAXITER).re;
}

__kernel void orbit_trap_im(__global FPN       *res_g,
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

    res_g[i*M+j] = _orbit_trap(p, _c, *trap, param->MAXITER).im;
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

    int _i = (int) ( ((float) (dims->imH-1)) * res_g[i*M+j].im );
    int _j = (int) ( ((float) (dims->imW-1)) * res_g[i*M+j].re );

    mim_g[i*M+j] = sim_g[_i*dims->imW + _j];

}

__kernel void map_img2  (__global FPN     *res1_g,
                         __global FPN     *res2_g,
                         __global Pixel_t   *sim_g, // sample image
                         __global Pixel_t   *mim_g, // mapped image
                         __global ImDims_t  *dims)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    int _i = (int) ( ((float) (dims->imH-1)) * res1_g[i*M+j] );
    int _j = (int) ( ((float) (dims->imW-1)) * res2_g[i*M+j] );

    mim_g[i*M+j] = sim_g[_i*dims->imW + _j];

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
                        __global unsigned char *img_g,
                        Freqs_t freqs)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = get_global_size(0);
    int M = get_global_size(1);

    int fi = i*M + j;
    int pix_idx = fi*3;

    img_g[pix_idx + 0] =127*(sin(res_g[fi]*freqs.f1)+1);
    img_g[pix_idx + 1] =127*(sin(res_g[fi]*freqs.f2)+1);
    img_g[pix_idx + 2] =127*(sin(res_g[fi]*freqs.f3)+1);

}
