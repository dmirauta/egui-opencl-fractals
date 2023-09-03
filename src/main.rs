extern crate ocl;
use std::{fs, path::Path};

use ocl::{builders::ProgramBuilder, Buffer, ProQue};

static OCL_SOURCE: &str = "./src/ocl/mandel.cl";

struct CFPParam;

fn main() -> ocl::Result<()> {
    let imdim = (120, 30);

    let src =
        fs::read_to_string(OCL_SOURCE).expect(format!("could not load {OCL_SOURCE}").as_str());
    let mut prog_build = ProgramBuilder::new();
    prog_build.src(src).cmplr_opt("-I./src/ocl");
    dbg!(prog_build.get_compiler_options().unwrap());
    let pro_que = ProQue::builder()
        .prog_bldr(prog_build)
        .dims((imdim.1, imdim.0))
        .build()?;

    // automatically sized like workgroup
    let iters_buff = pro_que.create_buffer::<i32>()?;

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

    let MAXITER: i32 = 1000;

    let kernel = pro_que
        .kernel_builder("escape_iter_args")
        .arg(&iters_buff)
        .arg(-2f64) //left
        .arg(0.5f64) //right
        .arg(-1.1f64) // bot
        .arg(1.1f64) // top
        .arg(0f64) // cre
        .arg(0f64) // cim
        .arg(1) // mandel
        .arg(MAXITER)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    let mut iters_vec = vec![0i32; iters_buff.len()];
    iters_buff.read(&mut iters_vec).enq()?;

    // dbg!(iters_vec);
    let (width, height) = imdim;
    for i in 0..height {
        for j in 0..width {
            if iters_vec[i * width + j] > MAXITER - 5 {
                print!("*");
            } else {
                print!(" ");
            }
        }
        println!();
    }

    Ok(())
}
