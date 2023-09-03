#ifdef USE_FLOAT
typedef float FPN;
#define FZERO 0
#define FONE 1
#else
typedef double FPN;
#define FZERO 0.0
#define FONE 1.0
#endif

typedef struct Complex {
  FPN re;
  FPN im;
} Complex_t;

typedef struct Box {
  FPN left;
  FPN right;
  FPN bot;
  FPN top;
} Box_t;

typedef struct Pixel {
  unsigned char r;
  unsigned char g;
  unsigned char b;
} Pixel_t;

typedef struct FParam {
  // General fract iter params
  int mandel;  // mandel or julia
  Complex_t c; // not given when mandel selected
  Box_t view_rect;
  int MAXITER;
} FParam_t;

typedef struct ImDims {
  int imH;
  int imW;
} ImDims_t;

typedef struct Freqs {
  FPN f1;
  FPN f2;
  FPN f3;
} Freqs_t;
