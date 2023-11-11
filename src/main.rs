extern crate ocl;
use eframe::NativeOptions;
use egui::RichText;
use egui_inspect::EguiInspect;
use epaint::{Color32, ColorImage, TextureHandle};
use ndarray::{Array2, Array3};
use ocl::{OclPrm, Platform, ProQue};
use simple_ocl::{prog_que_from_source_path, PairedBuffers2};
use std::{
    error::Error,
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

struct ItersImage {
    mat_dims: (usize, usize),
    rgb: Array3<u8>,
    cimage: ColorImage,
    texture: Option<TextureHandle>,
}

impl ItersImage {
    fn new(mat_dims: (usize, usize)) -> Self {
        Self {
            mat_dims,
            rgb: Array3::zeros((mat_dims.0, mat_dims.1, 3)),
            texture: None,
            cimage: Default::default(),
        }
    }

    fn update(&mut self, max_iter: i32, iters: &Array2<i32>) {
        for ((i, j), it) in iters.indexed_iter() {
            let d = (255.0f32 * (*it as f32) / (max_iter as f32)) as u8;
            for k in 0..3 {
                self.rgb[(i, j, k)] = d;
            }
        }

        self.cimage = ColorImage::from_rgb(
            [self.mat_dims.1, self.mat_dims.0],
            self.rgb.as_slice().unwrap(),
        );
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

struct OCLHelper {
    pro_que: ProQue,
    buff: PairedBuffers2<i32>,
}

impl OCLHelper {
    fn new(im_mat_dims: (usize, usize)) -> Self {
        let mut pro_que =
            prog_que_from_source_path("./src/ocl/mandel.cl", vec!["-I./src/ocl".to_string()]);
        let buff = PairedBuffers2::create_from(Array2::<i32>::zeros(im_mat_dims), &mut pro_que);
        OCLHelper { pro_que, buff }
    }

    fn arc_mut_new(im_mat_dims: (usize, usize)) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self::new(im_mat_dims)))
    }
}

type ThreadResult = Result<(), String>;

#[derive(EguiInspect)]
struct FractalViewer {
    fparam: FParamUI,
    #[inspect(hide)]
    old_fparam: FParamUI,
    iters_image: ItersImage,
    #[inspect(hide)]
    ocl_helper: Arc<Mutex<OCLHelper>>,
    #[inspect(hide)]
    join_handle: Option<JoinHandle<ThreadResult>>,
}

impl FractalViewer {
    fn new() -> Self {
        let im_mat_dims = (768, 1280);

        Self {
            iters_image: ItersImage::new(im_mat_dims),
            fparam: FParamUI::new(),
            old_fparam: FParamUI::default(), // should differ by maxiter at init
            ocl_helper: OCLHelper::arc_mut_new(im_mat_dims),
            join_handle: None,
        }
    }

    fn run_kernel_in_background(&mut self) {
        let cl_data_arc = self.ocl_helper.clone();
        let fparam = self.fparam.clone();
        let dims = self.iters_image.mat_dims;

        self.join_handle = Some(std::thread::spawn(move || match cl_data_arc.try_lock() {
            Ok(mut guard) => {
                guard.pro_que.set_dims(dims);
                let kernel = guard
                    .pro_que
                    .kernel_builder("escape_iter")
                    .arg(&guard.buff.device)
                    .arg(fparam.get_c_struct())
                    .build()?;

                unsafe {
                    kernel.enq()?;
                }

                guard.buff.from_device()?;

                Ok(())
            }
            Err(_) => Err("mutex is locked".to_string()),
        }));
    }

    fn collect_result(&mut self) {
        let handle = self.join_handle.take().unwrap();
        match handle.join().expect("thread join error") {
            Ok(_) => match self.ocl_helper.try_lock() {
                Ok(guard) => {
                    self.iters_image
                        .update(self.old_fparam.max_iter, &guard.buff.host);
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
