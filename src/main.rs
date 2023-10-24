extern crate ocl;
use eframe::egui;
use egui::RichText;
use egui_extras::image::RetainedImage;
use egui_inspect::EguiInspect;
use epaint::{Color32, ColorImage};
use ocl::{builders::ProgramBuilder, Buffer, ProQue};
use std::{
    error::Error,
    fs,
    sync::{Arc, Mutex},
    thread::JoinHandle,
};

static OCL_SOURCE: &str = "./src/ocl/mandel.cl";

#[derive(EguiInspect, Clone, PartialEq)]
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

#[derive(EguiInspect, PartialEq, Clone)]
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

#[derive(Default, EguiInspect, PartialEq, Clone)]
enum FractalMode {
    #[default]
    Mandel,
    Julia {
        c: Complex,
    },
}

impl FractalMode {
    fn get_c(&self) -> Complex {
        match self {
            FractalMode::Mandel => Complex::default(),
            FractalMode::Julia { c } => c.clone(),
        }
    }
}

#[derive(Default, EguiInspect, Clone, PartialEq)]
struct FParam {
    mode: FractalMode,
    view_rect: BBox,
    #[inspect(min = 1.0, max = 1000.0)]
    max_iter: i32,
}

impl FParam {
    fn new() -> Self {
        Self {
            max_iter: 100,
            ..Default::default()
        }
    }
}

fn rgba_from_iters(dims: (usize, usize), iters: Vec<i32>, max_iter: i32) -> Vec<u8> {
    let (height, width) = dims;
    let mut rgba = vec![255; width * height * 4];
    for i in 0..height {
        for j in 0..width {
            for k in 0..3 {
                rgba[i * width * 4 + j * 4 + k] =
                    (255.0f32 * (iters[i * width + j] as f32) / (max_iter as f32)) as u8;
            }
        }
    }
    rgba
}

fn gen_ret_image(dims: (usize, usize), rgba: Vec<u8>) -> RetainedImage {
    let cimage = ColorImage::from_rgba_unmultiplied(dims.into(), rgba.as_slice());
    RetainedImage::from_color_image("iters", cimage)
}

struct ItersImage {
    dims: (usize, usize),
    ret_image: RetainedImage,
}

impl ItersImage {
    fn new(dims: (usize, usize)) -> Self {
        let buff_len = dims.0 * dims.1;
        let ret_image = gen_ret_image(dims, vec![255; buff_len * 4]);
        Self { dims, ret_image }
    }
}

impl EguiInspect for ItersImage {
    fn inspect(&self, _label: &str, _ui: &mut egui::Ui) {
        todo!()
    }

    fn inspect_mut(&mut self, _label: &str, ui: &mut egui::Ui) {
        self.ret_image.show(ui);
    }
}

struct CLData {
    pro_que: ProQue,
    buff: Buffer<i32>,
}

type ThreadResult = Result<Vec<i32>, String>;

#[derive(EguiInspect)]
struct FractalViewer {
    fparam: FParam,
    #[inspect(hide)]
    old_fparam: FParam,
    iters_image: ItersImage,
    #[inspect(hide)]
    cl_data: Arc<Mutex<CLData>>,
    #[inspect(hide)]
    join_handle: Option<JoinHandle<ThreadResult>>,
}

impl Default for FractalViewer {
    fn default() -> Self {
        let imdims = (1280, 768);

        let src =
            fs::read_to_string(OCL_SOURCE).expect(format!("could not load {OCL_SOURCE}").as_str());
        let mut prog_build = ProgramBuilder::new();
        prog_build.src(src).cmplr_opt("-I./src/ocl");
        // dbg!(prog_build.get_compiler_options().unwrap());

        let pro_que = ProQue::builder()
            .prog_bldr(prog_build)
            .dims((imdims.1, imdims.0)) // workgroup
            .build()
            .expect("proque error");

        // automatically sized like workgroup dims
        let buff = pro_que.create_buffer::<i32>().expect("buffer create error");

        // TODO: c_struct params and custom length buffers
        // let fp_param = Buffer::<CFPParam>::builder()
        //     .queue(pro_que.queue().clone())
        //     .len(1)
        //     .fill_val(Default::default())
        //     .build();

        Self {
            iters_image: ItersImage::new(imdims),
            fparam: FParam::new(),
            old_fparam: FParam::default(), // should differ by maxiter
            cl_data: Arc::new(Mutex::new(CLData { pro_que, buff })),
            join_handle: None,
        }
    }
}

impl FractalViewer {
    fn run_kernel_in_background(&mut self) {
        let cl_data_arc = self.cl_data.clone();
        let fparam = self.fparam.clone();

        self.join_handle = Some(std::thread::spawn(move || {
            match cl_data_arc.try_lock() {
                Ok(guard) => {
                    let julia_c = fparam.mode.get_c();
                    let mode_int = match fparam.mode {
                        FractalMode::Mandel => 1,
                        FractalMode::Julia { .. } => 0,
                    };
                    let kernel = guard
                        .pro_que
                        .kernel_builder("escape_iter_args")
                        .arg(&guard.buff)
                        .arg(fparam.view_rect.left)
                        .arg(fparam.view_rect.right)
                        .arg(fparam.view_rect.bot)
                        .arg(fparam.view_rect.top)
                        .arg(julia_c.re)
                        .arg(julia_c.im)
                        .arg(mode_int) // mandel
                        .arg(fparam.max_iter)
                        .build()?;

                    unsafe {
                        kernel.enq()?;
                    }

                    // TODO: Could do read back from main thread
                    let mut new_iters: Vec<i32> = vec![0; guard.buff.len()];
                    guard.buff.read(&mut new_iters).enq()?;
                    Ok(new_iters)
                }
                Err(_) => Err("mutex is locked".to_string()),
            }
        }));
    }
}

impl eframe::App for FractalViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut collect_res = false;
        let job_still_running = match &mut self.join_handle {
            Some(handle) => {
                collect_res = handle.is_finished();
                !collect_res
            }
            None => false,
        };

        if collect_res {
            let handle = self.join_handle.take().unwrap();
            let res = handle.join().expect("thread join error");
            match res {
                Ok(new_iters) => {
                    let rgba =
                        rgba_from_iters(self.iters_image.dims, new_iters, self.fparam.max_iter);
                    self.iters_image.ret_image = gen_ret_image(self.iters_image.dims, rgba);
                }
                Err(err) => println!("Error on other thread: {}", err),
            }
        }

        let mut status_text = RichText::new("GPU Busy").color(Color32::RED);
        let params_updated = self.old_fparam != self.fparam;
        if !job_still_running {
            if params_updated {
                self.run_kernel_in_background();
                self.old_fparam = self.fparam.clone();
            } else {
                status_text = RichText::new("GPU Waiting").color(Color32::GREEN);
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label(status_text);

            self.inspect_mut("", ui)
        });
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(800.0, 600.0)),
        ..Default::default()
    };
    eframe::run_native(
        "Fractal viewer",
        options,
        Box::new(|_cc| Box::<FractalViewer>::default()),
    )?;
    Ok(())
}
