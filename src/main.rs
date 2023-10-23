extern crate ocl;
use eframe::egui;
use egui_extras::image::RetainedImage;
use egui_inspect::EguiInspect;
use epaint::ColorImage;
use ocl::{builders::ProgramBuilder, Buffer, ProQue};
use std::fs;

static OCL_SOURCE: &str = "./src/ocl/mandel.cl";

#[derive(EguiInspect)]
struct BBox {
    #[inspect(min=-2.0, max=2.0)]
    left: f64,
    #[inspect(min=-2.0, max=2.0)]
    right: f64,
    #[inspect(min=-2.0, max=2.0)]
    bot: f64,
    #[inspect(min=-2.0, max=2.0)]
    top: f64,
}

impl Default for BBox {
    fn default() -> Self {
        Self {
            left: -2.0,
            right: 2.0,
            // for mandel
            // left: -2.0,
            // right: 0.5,
            bot: -1.1,
            top: 1.1,
        }
    }
}

#[derive(EguiInspect)]
struct Complex {
    #[inspect(min=-2.0, max=2.0)]
    re: f64,
    #[inspect(min=-2.0, max=2.0)]
    im: f64,
}

impl Default for Complex {
    fn default() -> Self {
        Self { re: -0.7, im: 0.3 }
    }
}

#[derive(Default, EguiInspect)]
struct FParam {
    #[inspect(hide)]
    mandel: i32,
    mandel_mode: bool,
    c: Complex,
    view_rect: BBox,
    #[inspect(min = 1.0, max = 1000.0)]
    max_iter: i32,
}

impl FParam {
    fn new() -> Self {
        Self {
            mandel: 1,
            max_iter: 100,
            ..Default::default()
        }
    }
}

struct ItersImage {
    dims: (usize, usize),
    buff: Buffer<i32>,
    iters: Vec<i32>,
    rgba: Vec<u8>,
    max_iter: f32,
}

impl ItersImage {
    fn new(dims: (usize, usize), iters_buff: Buffer<i32>) -> Self {
        let buff_len = iters_buff.len();
        Self {
            dims,
            buff: iters_buff,
            iters: vec![0i32; buff_len],
            rgba: vec![255; buff_len * 4],
            max_iter: 100.0,
        }
    }

    fn rgba_from_iters(&mut self) {
        let (height, width) = self.dims;
        for i in 0..height {
            for j in 0..width {
                for k in 0..3 {
                    self.rgba[i * width * 4 + j * 4 + k] =
                        (255.0f32 * (self.iters[i * width + j] as f32) / self.max_iter) as u8;
                }
            }
        }
    }
}

impl EguiInspect for ItersImage {
    fn inspect(&self, _label: &str, _ui: &mut egui::Ui) {
        todo!()
    }

    fn inspect_mut(&mut self, _label: &str, ui: &mut egui::Ui) {
        // TODO: should only update on FPParams change
        self.rgba_from_iters();
        let cimage = ColorImage::from_rgba_unmultiplied(self.dims.into(), self.rgba.as_slice());
        let image = RetainedImage::from_color_image("iters", cimage);
        image.show(ui);
    }
}

#[derive(EguiInspect)]
struct FractalViewer {
    #[inspect(hide)]
    pro_que: ProQue,
    iters_image: ItersImage,
    fparam: FParam,
}

impl Default for FractalViewer {
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

        Self {
            pro_que,
            iters_image: ItersImage::new(imdims, iters_buff),
            fparam: FParam::new(),
        }
    }
}

impl FractalViewer {
    fn run_kernel(&mut self) -> ocl::Result<()> {
        let kernel = self
            .pro_que
            .kernel_builder("escape_iter_args")
            .arg(&self.iters_image.buff)
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

        self.iters_image
            .buff
            .read(&mut self.iters_image.iters)
            .enq()?;

        Ok(())
    }
}

impl eframe::App for FractalViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.run_kernel().unwrap();

            self.fparam
                .mandel_mode
                .inspect_mut("mandel mode (else julia)", ui);
            self.fparam.mandel = self.fparam.mandel_mode as i32;
            self.iters_image.max_iter = self.fparam.max_iter as f32;

            if !self.fparam.mandel_mode {
                self.fparam.c.inspect_mut("c", ui);
            }

            self.fparam.view_rect.inspect_mut("view rect", ui);

            self.iters_image.inspect_mut("Fractal view", ui);
        });
    }
}

fn main() {
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(800.0, 600.0)),
        ..Default::default()
    };
    eframe::run_native(
        "Fractal viewer",
        options,
        Box::new(|_cc| Box::<FractalViewer>::default()),
    );
}
