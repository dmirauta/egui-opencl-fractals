extern crate ocl;
use eframe::NativeOptions;
use egui::RichText;
use egui_inspect::EguiInspect;
use epaint::{Color32, ColorImage, TextureHandle};
use ocl::{builders::ProgramBuilder, Buffer, OclPrm, Platform, ProQue};
use std::{
    error::Error,
    fs,
    sync::{Arc, Mutex},
    thread::JoinHandle,
};

#[repr(C)]
#[derive(EguiInspect, Clone, Copy, Debug, Default, PartialEq)]
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

// TODO: Could really benefit from logarithmic sliders and variable bounds...
#[repr(C)]
#[derive(Debug, EguiInspect, PartialEq, Clone, Copy)]
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

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct FParam {
    mode_int: i32,
    c: Complex,
    view: BBox,
    max_iter: i32,
}

unsafe impl OclPrm for FParam {}

#[derive(Default, EguiInspect, Clone, PartialEq)]
struct FParamUI {
    mode: FractalMode,
    center: Complex,
    delta: Complex,
    #[inspect(min = 1.0, max = 1000.0)]
    max_iter: i32,
}

impl FParamUI {
    fn new() -> Self {
        Self {
            max_iter: 100,
            center: Complex { re: -0.4, im: 0.0 },
            delta: Complex { re: 1.5, im: 1.2 },
            ..Default::default()
        }
    }

    fn get_bbox(&self) -> BBox {
        BBox {
            left: self.center.re - self.delta.re,
            right: self.center.re + self.delta.re,
            bot: self.center.im - self.delta.im,
            top: self.center.im + self.delta.im,
        }
    }

    fn get_c_struct(&self) -> FParam {
        FParam {
            mode_int: match self.mode {
                FractalMode::Mandel => 1,
                FractalMode::Julia { .. } => 0,
            },
            c: self.mode.get_c(),
            view: self.get_bbox(),
            max_iter: self.max_iter,
        }
    }
}

fn rgb_from_iters(dims: (usize, usize), iters: &Vec<i32>, rgb: &mut Vec<u8>, max_iter: i32) {
    let (height, width) = dims;
    for i in 0..height {
        for j in 0..width {
            for k in 0..3 {
                rgb[i * width * 3 + j * 3 + k] =
                    (255.0f32 * (iters[i * width + j] as f32) / (max_iter as f32)) as u8;
            }
        }
    }
}

struct ItersImage {
    dims: (usize, usize),
    iters: Vec<i32>,
    rgb: Vec<u8>,
    cimage: ColorImage,
    texture: Option<TextureHandle>,
}

impl ItersImage {
    fn new(dims: (usize, usize)) -> Self {
        let buff_len = dims.0 * dims.1;
        let iters = vec![0; buff_len];
        let rgba = vec![255; buff_len * 3];
        Self {
            dims,
            iters,
            rgb: rgba,
            texture: None,
            cimage: Default::default(),
        }
    }

    fn update(&mut self, max_iter: i32) {
        rgb_from_iters(self.dims, &self.iters, &mut self.rgb, max_iter);
        self.cimage = ColorImage::from_rgb(self.dims.into(), self.rgb.as_slice());
        self.texture = None;
    }
}

impl EguiInspect for ItersImage {
    fn inspect(&self, _label: &str, _ui: &mut egui::Ui) {
        todo!()
    }

    fn inspect_mut(&mut self, _label: &str, ui: &mut egui::Ui) {
        let handle: &egui::TextureHandle = self.texture.get_or_insert_with(|| {
            ui.ctx()
                .load_texture("test_img", self.cimage.clone(), Default::default())
        });
        ui.image(handle);
    }
}

struct CLData {
    pro_que: ProQue,
    buff: Buffer<i32>,
}

type ThreadResult = Result<(), String>;

#[derive(EguiInspect)]
struct FractalViewer {
    fparam: FParamUI,
    #[inspect(hide)]
    old_fparam: FParamUI,
    iters_image: ItersImage,
    #[inspect(hide)]
    cl_data: Arc<Mutex<CLData>>,
    #[inspect(hide)]
    join_handle: Option<JoinHandle<ThreadResult>>,
}

impl FractalViewer {
    fn new() -> Self {
        let imdims = (1280, 768);

        let ocl_main = "./src/ocl/mandel.cl";
        let src =
            fs::read_to_string(ocl_main).expect(format!("could not load {ocl_main}").as_str());
        let mut prog_build = ProgramBuilder::new();
        prog_build.src(src).cmplr_opt("-I./src/ocl");

        let pro_que = ProQue::builder()
            .prog_bldr(prog_build)
            .dims((imdims.1, imdims.0)) // workgroup
            .build()
            .expect("proque error");

        // automatically sized like workgroup dims
        let buff = pro_que.create_buffer::<i32>().expect("buffer create error");

        Self {
            iters_image: ItersImage::new(imdims),
            fparam: FParamUI::new(),
            old_fparam: FParamUI::default(), // should differ by maxiter
            cl_data: Arc::new(Mutex::new(CLData { pro_que, buff })),
            join_handle: None,
        }
    }

    fn run_kernel_in_background(&mut self) {
        let cl_data_arc = self.cl_data.clone();
        let fparam = self.fparam.clone();

        self.join_handle = Some(std::thread::spawn(move || match cl_data_arc.try_lock() {
            Ok(guard) => {
                let kernel = guard
                    .pro_que
                    .kernel_builder("escape_iter")
                    .arg(&guard.buff)
                    .arg(fparam.get_c_struct())
                    .build()?;

                unsafe {
                    kernel.enq()?;
                }

                Ok(())
            }
            Err(_) => Err("mutex is locked".to_string()),
        }));
    }

    fn collect_result(&mut self) {
        let handle = self.join_handle.take().unwrap();
        match handle.join().expect("thread join error") {
            Ok(_) => match self.cl_data.try_lock() {
                Ok(guard) => {
                    guard
                        .buff
                        .read(&mut self.iters_image.iters)
                        .enq()
                        .expect("readback error");
                    self.iters_image.update(self.old_fparam.max_iter);
                }
                Err(err) => println!("could not aquire mutex in update: {err}"),
            },
            Err(err) => println!("Error on other thread: {}", err),
        }
    }
}

impl eframe::App for FractalViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let job_still_running = match &mut self.join_handle {
            Some(handle) => {
                if handle.is_finished() {
                    // On main thread, assuming readback is much quicker than computation,
                    // saves having to pass result between threads.
                    // Alternatively an Arc<Mutex<>> could be used on image.iters.
                    self.collect_result();
                    false
                } else {
                    true
                }
            }
            None => false,
        };

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
    // Note: Work around to strange segfault issue when building proque in eframe::App
    dbg!(Platform::default());

    eframe::run_native(
        "Fractal viewer",
        NativeOptions::default(),
        Box::new(|_cc| Box::new(FractalViewer::new())),
    )?;
    Ok(())
}
