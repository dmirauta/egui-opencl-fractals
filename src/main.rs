extern crate ocl;
use eframe::NativeOptions;
use egui::RichText;
use egui_inspect::EguiInspect;
use epaint::{Color32, ColorImage, TextureHandle};
use ndarray::{Array2, Array3};
use ocl::{OclPrm, Platform, ProQue};
use simple_ocl::{prog_que_from_source_path, PairedBuffers2, PairedBuffers3};
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

#[repr(C)]
#[derive(Debug, EguiInspect, PartialEq, Clone, Copy)]
struct Freqs {
    f1: f64,
    f2: f64,
    f3: f64,
}

impl Default for Freqs {
    fn default() -> Self {
        Self {
            f1: 1.1,
            f2: 3.3,
            f3: 9.9,
        }
    }
}

unsafe impl OclPrm for Freqs {}

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

#[derive(EguiInspect, Clone, PartialEq)]
struct FParamUI {
    mode: FractalMode,
    center: Complex,
    delta: Complex,
    #[inspect(min = 1.0, max = 1000.0)]
    max_iter: i32,
}

impl Default for FParamUI {
    fn default() -> Self {
        Self {
            max_iter: 100,
            center: Complex { re: -0.4, im: 0.0 },
            delta: Complex { re: 1.5, im: 1.2 },
            mode: Default::default(),
        }
    }
}

impl FParamUI {
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
    cimage: ColorImage,
    texture: Option<TextureHandle>,
}

impl ItersImage {
    fn new(mat_dims: (usize, usize)) -> Self {
        Self {
            mat_dims,
            texture: None,
            cimage: Default::default(),
        }
    }

    fn update(&mut self, rgb: &Array3<u8>) {
        self.cimage =
            ColorImage::from_rgb([self.mat_dims.1, self.mat_dims.0], rgb.as_slice().unwrap());
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
    field_1: PairedBuffers2<f64>,
    rgb: PairedBuffers3<u8>,
}

impl OCLHelper {
    fn new(im_mat_dims: (usize, usize)) -> Self {
        let mut pro_que =
            prog_que_from_source_path("./src/ocl/mandel.cl", vec!["-I./src/ocl".to_string()]);
        let field_1 = PairedBuffers2::create_from(Array2::<f64>::zeros(im_mat_dims), &mut pro_que);
        let (n, m) = im_mat_dims;
        let rgb = PairedBuffers3::create_from(Array3::<u8>::zeros((n, m, 3)), &mut pro_que);
        OCLHelper {
            pro_que,
            field_1,
            rgb,
        }
    }

    fn arc_mut_new(im_mat_dims: (usize, usize)) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self::new(im_mat_dims)))
    }
}

#[derive(Clone, PartialEq, EguiInspect)]
enum FractalField {
    ItersToEscape { fparam: FParamUI, freqs: Freqs },
}

type ThreadResult = Result<(), String>;

#[derive(EguiInspect)]
struct FractalViewer {
    ff: FractalField,
    #[inspect(hide)]
    old_ff: FractalField,
    iters_image: ItersImage,
    #[inspect(hide)]
    ocl_helper: Arc<Mutex<OCLHelper>>,
    #[inspect(hide)]
    join_handle: Option<JoinHandle<ThreadResult>>,
}

impl FractalViewer {
    fn new() -> Self {
        let im_mat_dims = (768, 1280);
        let fparam = FParamUI::default();
        let mut old_fparam = FParamUI::default();
        old_fparam.max_iter = 0;

        Self {
            iters_image: ItersImage::new(im_mat_dims),
            ocl_helper: OCLHelper::arc_mut_new(im_mat_dims),
            join_handle: None,
            ff: FractalField::ItersToEscape {
                fparam,
                freqs: Default::default(),
            },
            old_ff: FractalField::ItersToEscape {
                fparam: old_fparam,
                freqs: Default::default(),
            },
        }
    }

    fn run_kernel_in_background(&mut self) {
        let cl_data_arc = self.ocl_helper.clone();
        match self.ff.clone() {
            FractalField::ItersToEscape { fparam, freqs } => {
                let dims = self.iters_image.mat_dims;

                self.join_handle = Some(std::thread::spawn(move || match cl_data_arc.try_lock() {
                    Ok(mut guard) => {
                        guard.pro_que.set_dims(dims);

                        let kernel = guard
                            .pro_que
                            .kernel_builder("escape_iter_fpn")
                            .arg(&guard.field_1.device)
                            .arg(fparam.get_c_struct())
                            .build()?;

                        unsafe {
                            kernel.enq()?;
                        }

                        let kernel = guard
                            .pro_que
                            .kernel_builder("map_sines")
                            .arg(&guard.field_1.device)
                            .arg(&guard.rgb.device)
                            .arg(freqs)
                            .build()?;

                        unsafe {
                            kernel.enq()?;
                        }

                        guard.rgb.from_device()?;

                        Ok(())
                    }
                    Err(_) => Err("mutex is locked".to_string()),
                }));
            }
        }
    }

    fn collect_result(&mut self) {
        let handle = self.join_handle.take().unwrap();
        match handle.join().expect("thread join error") {
            Ok(_) => match self.ocl_helper.try_lock() {
                Ok(guard) => match &self.old_ff {
                    FractalField::ItersToEscape { .. } => {
                        self.iters_image.update(&guard.rgb.host);
                    }
                },
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
        let params_updated = self.old_ff != self.ff;
        if !job_still_running {
            if params_updated {
                self.run_kernel_in_background();
                self.old_ff = self.ff.clone();
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
