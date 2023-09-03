extern crate ocl_core;

// while ProQueue seems nice, we want access to build options and possibly different buffer
// dimensions

use ocl_core::ContextProperties;
use ocl_core::{CommandQueue, Context, Program};
use std::{ffi::CString, fs};

struct MyCLContext {
    context: Context,
    program: Program,
    queue: CommandQueue,
}

impl MyCLContext {
    fn new(ocl_source: &str, build_options: &str) -> MyCLContext {
        let platform_id = ocl_core::default_platform().unwrap();
        let device_ids = ocl_core::get_device_ids(&platform_id, None, None).unwrap();
        let device_id = device_ids[0];
        let context_properties = ContextProperties::new().platform(platform_id);
        let context =
            ocl_core::create_context(Some(&context_properties), &[device_id], None, None).unwrap();

        let src = fs::read_to_string(ocl_source).expect("source not found");
        let src_cstring = CString::new(src).unwrap();
        let program = ocl_core::create_program_with_source(&context, &[src_cstring]).unwrap();
        ocl_core::build_program(
            &program,
            None::<&[()]>,
            &CString::new(build_options).unwrap(),
            None,
            None,
        )
        .unwrap();

        let queue = ocl_core::create_command_queue(&context, &device_id, None).unwrap();
        MyCLContext {
            context,
            program,
            queue,
        }
    }
}
