extern crate ocl;
use eframe::egui;
use egui::Slider;
use egui_extras::image::RetainedImage;
use epaint::ColorImage;
use ocl::{builders::ProgramBuilder, Buffer, ProQue};
use std::fs;

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
    rgba_vec: Vec<u8>,
    fparam: FParam,
}

impl Default for MyApp {
    fn default() -> Self {
        let imdims = (600, 400);

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
            rgba_vec: vec![255; buff_len * 4],
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

    fn image_from_iters(&mut self) {
        let (height, width) = self.imdims;
        for i in 0..height {
            for j in 0..width {
                for k in 0..3 {
                    self.rgba_vec[i * width * 4 + j * 4 + k] =
                        (255.0f32 * (self.iters_vec[i * width + j] as f32)
                            / (self.fparam.max_iter as f32)) as u8;
                }
            }
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add(
                Slider::new(
                    &mut self.fparam.view_rect.left,
                    -2f64..=self.fparam.view_rect.right,
                )
                .text("left"),
            );
            ui.add(
                Slider::new(
                    &mut self.fparam.view_rect.right,
                    self.fparam.view_rect.left..=0.5f64,
                )
                .text("right"),
            );
            ui.add(
                Slider::new(
                    &mut self.fparam.view_rect.bot,
                    -1f64..=self.fparam.view_rect.top,
                )
                .text("bot"),
            );
            ui.add(
                Slider::new(
                    &mut self.fparam.view_rect.top,
                    self.fparam.view_rect.bot..=1f64,
                )
                .text("top"),
            );

            self.run_kernel().unwrap();
            self.image_from_iters();
            let cimage = ColorImage::from_rgba_unmultiplied(
                [self.imdims.0, self.imdims.1],
                self.rgba_vec.as_slice(),
            );
            let image = RetainedImage::from_color_image("iters", cimage);
            image.show(ui);
        });
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(800.0, 600.0)),
        ..Default::default()
    };
    eframe::run_native(
        "Fractal viewer",
        options,
        Box::new(|_cc| Box::<MyApp>::default()),
    )
}
