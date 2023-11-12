extern crate ocl;
use eframe::NativeOptions;
use egui::RichText;
use egui_extras::syntax_highlighting::{highlight, CodeTheme};
use egui_inspect::{EguiInspect, InspectNumber};
use epaint::{Color32, ColorImage, TextureHandle};
use ndarray::{Array2, Array3};
use ocl::{Platform, ProQue};
use simple_ocl::{try_prog_que_from_source, PairedBuffers2, PairedBuffers3};
use std::{
    error::Error,
    sync::{Arc, Mutex},
    thread::JoinHandle,
};

mod wrapper_types;
use wrapper_types::{BBox, Complex, Freqs, ProxType, SFParam};

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

/// UI for Shared fractal params
#[derive(EguiInspect, Clone, PartialEq)]
struct SFParamUI {
    mode: FractalMode,
    view_center: Complex,
    #[inspect(log_slider, min = 1e-19, max = 2.0)]
    zoom: f64,
    #[inspect(log_slider, min = 0.1, max = 10.0)]
    aspect: f64,
    #[inspect(log_slider, min = 1.0, max = 10000.0)]
    max_iter: i32,
}

impl Default for SFParamUI {
    fn default() -> Self {
        Self {
            max_iter: 100,
            view_center: Complex { re: -0.4, im: 0.0 },
            mode: Default::default(),
            zoom: 1.0,
            aspect: 1.0,
        }
    }
}

impl SFParamUI {
    fn get_view_bbox(&self) -> BBox {
        let delta_re = self.zoom;
        let delta_im = self.zoom * self.aspect;
        BBox {
            left: self.view_center.re - delta_re,
            right: self.view_center.re + delta_re,
            bot: self.view_center.im - delta_im,
            top: self.view_center.im + delta_im,
        }
    }

    fn get_c_struct(&self) -> SFParam {
        SFParam {
            mode_int: match self.mode {
                FractalMode::Mandel => 1,
                FractalMode::Julia { .. } => 0,
            },
            c: self.mode.get_c(),
            view: self.get_view_bbox(),
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

// ocl source baked into binary at build time
static OCL_STRUCTS: &str = include_str!("./ocl/mandelstructs.h");
static OCL_FUNCS: &str = include_str!("./ocl/mandelutils.c");
static OCL_KERNELS: &str = include_str!("./ocl/mandel.cl");

fn insert_custom_func(custom_func: String) -> String {
    let funcs = OCL_FUNCS.to_string();
    let mut sp = funcs.split("//>>");
    let [before_func, remainder] = [sp.next().unwrap(), sp.next().unwrap()];
    let rs = remainder.to_string();
    let mut sp = rs.split("//<<");
    let [_, after_func] = [sp.next().unwrap(), sp.next().unwrap()];
    format!("{before_func}{custom_func}{after_func}")
}

struct OCLHelper {
    pro_que: ProQue,
    field_1: PairedBuffers2<f64>,
    rgb: PairedBuffers3<u8>,
}

impl OCLHelper {
    fn new(im_mat_dims: (usize, usize), custom_iter_func: Option<String>) -> ocl::Result<Self> {
        let ocl_funcs_custom = match custom_iter_func {
            Some(cf) => insert_custom_func(cf),
            None => OCL_FUNCS.to_string(),
        };
        let full_source = format!("{OCL_STRUCTS}{ocl_funcs_custom}{OCL_KERNELS}");
        let mut pro_que =
            try_prog_que_from_source(full_source, "mandel", vec!["-DEXTERNAL_CONCAT".to_string()])?;
        let field_1 = PairedBuffers2::create_from(Array2::<f64>::zeros(im_mat_dims), &mut pro_que);
        let (n, m) = im_mat_dims;
        let rgb = PairedBuffers3::create_from(Array3::<u8>::zeros((n, m, 3)), &mut pro_que);
        Ok(OCLHelper {
            pro_que,
            field_1,
            rgb,
        })
    }

    fn run_escape_iter(&mut self, fparam: SFParam) -> ocl::Result<()> {
        let kernel = self
            .pro_que
            .kernel_builder("escape_iter_fpn")
            .arg(&self.field_1.device)
            .arg(fparam)
            .build()?;

        unsafe {
            kernel.enq()?;
        }

        Ok(())
    }

    fn run_min_prox(&mut self, fparam: SFParam, prox_type: ProxType) -> ocl::Result<()> {
        let kernel = self
            .pro_que
            .kernel_builder("min_prox")
            .arg(&self.field_1.device)
            .arg(fparam)
            .arg(prox_type)
            .build()?;

        unsafe {
            kernel.enq()?;
        }

        Ok(())
    }

    fn run_box_trap_partial(&mut self, fparam: SFParam, box_: BBox, real: bool) -> ocl::Result<()> {
        let kernel_name = if real {
            "orbit_trap_re"
        } else {
            "orbit_trap_im"
        };

        let kernel = self
            .pro_que
            .kernel_builder(kernel_name)
            .arg(&self.field_1.device)
            .arg(fparam)
            .arg(box_)
            .build()?;

        unsafe {
            kernel.enq()?;
        }

        Ok(())
    }

    fn run_map_sines(&mut self, freqs: Freqs) -> ocl::Result<()> {
        let kernel = self
            .pro_que
            .kernel_builder("map_sines")
            .arg(&self.field_1.device)
            .arg(&self.rgb.device)
            .arg(freqs)
            .build()?;

        unsafe {
            kernel.enq()?;
        }

        Ok(())
    }
}

/// Scalar field selection for visualisation channels
#[derive(Clone, PartialEq, EguiInspect, Default)]
enum FractalFieldType {
    #[default]
    ItersToEscape,
    ChainMinProximity {
        prox_type: ProxType,
    },
    BoxTrapRe {
        #[inspect(name = "box")]
        box_: BBox,
    },
    BoxTrapIm {
        #[inspect(name = "box")]
        box_: BBox,
    },
}

#[derive(Clone, PartialEq, EguiInspect)]
enum FractalVisualisationType {
    SingleFieldCmaped {
        field_type: FractalFieldType,
        cmap_freqs: Freqs,
    },
    // DualFieldImageMap,
    // TriFieldRGB,
}

impl Default for FractalVisualisationType {
    fn default() -> Self {
        Self::SingleFieldCmaped {
            field_type: Default::default(),
            cmap_freqs: Default::default(),
        }
    }
}

#[derive(Clone, Default, PartialEq, EguiInspect)]
struct FractalParams {
    #[inspect(name = "Shared")]
    sfparam: SFParamUI,
    vis_type: FractalVisualisationType,
}

/// Basic editing with syntax highlighting through egui_extras, better highlighting available
/// in syntect
struct FunctionEditor {
    code: String,
    theme: CodeTheme,
}

impl Default for FunctionEditor {
    fn default() -> Self {
        Self {
            code: "// Define custom iteration function f: (Complex_t, Complex_t) -> Complex_t
// first argument is spatially dependent while the second is either 
// z_0 for mandel-like and user input for julia-like options.
// Complex_t has fields re and im. Convenience functions
// `complex_add: (Complex_t, Complex_t) -> Complex_t`
// `complex_mult: (Complex_t, Complex_t) -> Complex_t`
// and `complex_pow: (Complex_t, int) -> Complex_t` are in scope.
inline Complex_t f(Complex_t z, Complex_t c) {
  return complex_add(complex_pow(z, 2), c);
}"
            .to_string(),
            theme: Default::default(),
        }
    }
}

impl EguiInspect for FunctionEditor {
    fn inspect(&self, _label: &str, _ui: &mut egui::Ui) {
        todo!()
    }

    fn inspect_mut(&mut self, label: &str, ui: &mut egui::Ui) {
        let mut layouter = |ui: &egui::Ui, string: &str, wrap_width: f32| {
            let mut layout_job = highlight(ui.ctx(), &self.theme, string, "c");
            layout_job.wrap.max_width = wrap_width;
            ui.fonts(|f| f.layout_job(layout_job))
        };

        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.label(label);
            ui.add(
                egui::TextEdit::multiline(&mut self.code)
                    .font(egui::TextStyle::Monospace) // for cursor height
                    .code_editor()
                    .desired_rows(10)
                    .lock_focus(true)
                    .desired_width(f32::INFINITY)
                    .layouter(&mut layouter),
            );
        });
    }
}

type ThreadResult = Result<(), String>;

struct FractalViewer {
    fp: FractalParams,
    old_fp: FractalParams,
    editor: FunctionEditor,
    error: Option<String>,
    iters_image: ItersImage,
    ocl_helper: Arc<Mutex<OCLHelper>>,
    join_handle: Option<JoinHandle<ThreadResult>>,
}

impl FractalViewer {
    fn new() -> Self {
        let im_mat_dims = (768, 1280);
        let mut old_fp = FractalParams::default();
        old_fp.sfparam.max_iter = 0;

        Self {
            editor: Default::default(),
            iters_image: ItersImage::new(im_mat_dims),
            ocl_helper: Arc::new(Mutex::new(OCLHelper::new(im_mat_dims, None).unwrap())),
            join_handle: None,
            fp: Default::default(),
            old_fp,
            error: None,
        }
    }

    fn run_kernel_in_background(&mut self) {
        let helper_arc = self.ocl_helper.clone();
        let frac_param = self.fp.clone();
        let dims = self.iters_image.mat_dims;

        self.join_handle = Some(std::thread::spawn(move || {
            let FractalParams {
                sfparam,
                vis_type: fields,
            } = frac_param;
            let sfparam_c = sfparam.get_c_struct();
            match helper_arc.try_lock() {
                Ok(mut guard) => match fields {
                    FractalVisualisationType::SingleFieldCmaped {
                        field_type,
                        cmap_freqs: freqs,
                    } => {
                        guard.pro_que.set_dims(dims);
                        match field_type {
                            FractalFieldType::ItersToEscape => {
                                guard.run_escape_iter(sfparam_c)?;
                            }
                            FractalFieldType::ChainMinProximity { prox_type } => {
                                guard.run_min_prox(sfparam_c, prox_type)?;
                            }
                            FractalFieldType::BoxTrapRe { box_ } => {
                                guard.run_box_trap_partial(sfparam_c, box_, true)?;
                            }
                            FractalFieldType::BoxTrapIm { box_ } => {
                                guard.run_box_trap_partial(sfparam_c, box_, false)?;
                            }
                        }
                        guard.run_map_sines(freqs)?;
                        guard.rgb.from_device()?;
                        Ok(())
                    }
                    // FractalVisualisationType::DualFieldImageMap => todo!(),
                    // FractalVisualisationType::TriFieldRGB => todo!(),
                },
                Err(_) => Err("mutex is locked".to_string()),
            }
        }));
    }

    fn collect_result(&mut self) {
        let handle = self.join_handle.take().unwrap();
        match handle.join().expect("thread join error") {
            Ok(_) => match self.ocl_helper.try_lock() {
                Ok(guard) => self.iters_image.update(&guard.rgb.host),
                Err(err) => println!("could not aquire mutex in update: {err}"),
            },
            Err(err) => println!("Error on other thread: {}", err),
        }
    }

    fn try_recompile(&mut self) {
        if self.join_handle.is_none() {
            if let Ok(mut guard) = self.ocl_helper.try_lock() {
                match OCLHelper::new(self.iters_image.mat_dims, Some(self.editor.code.clone())) {
                    Ok(new_helper) => {
                        *guard = new_helper;
                        self.old_fp.sfparam.max_iter = 0; // trigger recompute
                        self.error = None;
                    }
                    Err(err) => self.error = Some(format!("{err}")),
                }
            }
        } else {
            self.error = Some(format!("Could not recompile, GPU was busy."));
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
        let params_updated = self.old_fp != self.fp;
        if !job_still_running {
            if params_updated {
                self.run_kernel_in_background();
                self.old_fp = self.fp.clone();
            } else {
                status_text = RichText::new("GPU Waiting").color(Color32::GREEN);
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.columns(2, |columns| {
                columns[0].label(status_text);

                self.fp.inspect_mut("Fractal parameters", &mut columns[0]);
                self.iters_image.inspect_mut("", &mut columns[0]);

                self.editor.inspect_mut("Custom function", &mut columns[1]);
                if columns[1].button("Recompile").clicked() {
                    self.try_recompile();
                }
                if let Some(err) = &self.error {
                    columns[1].label(err.as_str());
                }
            });
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
