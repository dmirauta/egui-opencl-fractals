extern crate ocl;
use eframe::egui;
use ocl::{builders::ProgramBuilder, Buffer, ProQue};
use std::{fs, path::Path};

static OCL_SOURCE: &str = "./src/ocl/mandel.cl";

struct BBox {
    left: f64,
    right: f64,
    bot: f64,
    top: f64,
}

struct Complex {
    re: f64,
    im: f64,
}

struct FParam {
    mandel: i32,
    c: Complex,
    view_rect: BBox,
    max_iter: i32,
}

struct MyApp {
    imdims: (usize, usize),
    pro_que: ProQue,
    iters_buff: Buffer<i32>,
    iters_vec: Vec<i32>,
    fparam: FParam,
}

impl Default for MyApp {
    fn default() -> Self {
        let imdims = (120, 30);

        let src =
            fs::read_to_string(OCL_SOURCE).expect(format!("could not load {OCL_SOURCE}").as_str());
        let mut prog_build = ProgramBuilder::new();
        prog_build.src(src).cmplr_opt("-I./src/ocl");
        dbg!(prog_build.get_compiler_options().unwrap());
        let pro_que = ProQue::builder()
            .prog_bldr(prog_build)
            .dims((imdims.1, imdims.0))
            .build()
            .expect("proque error");

        // automatically sized like workgroup
        let iters_buff = pro_que.create_buffer::<i32>().expect("buffer create error");

        // let fp_param = Buffer::<CFPParam>::builder()
        //     .queue(pro_que.queue().clone())
        //     .len(1)
        //     .fill_val(Default::default())
        //     .build();

        // let kernel = pro_que
        //     .kernel_builder("escape_iter")
        //     .arg(&iters)
        //     .arg(&fp_param)
        //     .build()?;

        let fparam = FParam {
            mandel: 1,
            c: Complex { re: 0.0, im: 0.0 },
            view_rect: BBox {
                left: -2.0,
                right: 0.5,
                bot: -1.1,
                top: 1.1,
            },
            max_iter: 1000,
        };

        let buff_len = iters_buff.len();
        Self {
            imdims,
            pro_que,
            iters_buff,
            iters_vec: vec![0i32; buff_len],
            fparam,
        }
    }
}

impl MyApp {
    fn run_kernel(&mut self) -> ocl::Result<()> {
        let kernel = self
            .pro_que
            .kernel_builder("escape_iter_args")
            .arg(&self.iters_buff)
            .arg(self.fparam.view_rect.left)
            .arg(self.fparam.view_rect.right)
            .arg(self.fparam.view_rect.bot)
            .arg(self.fparam.view_rect.top)
            .arg(self.fparam.c.re)
            .arg(self.fparam.c.im)
            .arg(self.fparam.mandel) // mandel
            .arg(self.fparam.max_iter)
            .build()?;

        unsafe {
            kernel.enq()?;
        }

        self.iters_buff.read(&mut self.iters_vec).enq()?;

        Ok(())
    }
}

fn main() {
    let mut myapp = MyApp::default();

    myapp.run_kernel().unwrap();

    // dbg!(iters_vec);
    let (width, height) = myapp.imdims;
    for i in 0..height {
        for j in 0..width {
            if myapp.iters_vec[i * width + j] > myapp.fparam.max_iter - 5 {
                print!("*");
            } else {
                print!(" ");
            }
        }
        println!();
    }
}
