#![feature(raw)]
use base64;
use core::ptr;
use core::raw::TraitObject;
use exr::prelude::rgba_image::*;
use image;
use lazy_static::lazy_static;
use ndspy_sys;
use nsi;
use num_enum::IntoPrimitive;
use std::{
    cell::RefCell,
    ffi::CStr,
    os::raw::{c_char, c_int, c_uint, c_void},
    rc::Rc,
};

use polyhedron_ops as p_ops;

lazy_static! {
    /// Initialize the display driver table.
    static ref __UNUSED: ndspy_sys::PtDspyError = unsafe{ ndspy_sys::DspyRegisterDriver(
        b"ferris\0" as *const u8 as _,
        Some(image_open as _),
        Some(image_data as _),
        Some(image_close as _),
        Some(image_query as _),
    ) };
}

#[cfg(features = "juypiter")]
mod juypiter;
mod render;

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, IntoPrimitive)]
pub enum Error {
    None = ndspy_sys::PtDspyError_PkDspyErrorNone as _,
    NoMemory = ndspy_sys::PtDspyError_PkDspyErrorNoMemory as _,
    Unsupported = ndspy_sys::PtDspyError_PkDspyErrorUnsupported as _,
    BadParameters = ndspy_sys::PtDspyError_PkDspyErrorBadParams as _,
    NoResource = ndspy_sys::PtDspyError_PkDspyErrorNoResource as _,
    Undefined = ndspy_sys::PtDspyError_PkDspyErrorUndefined as _,
    Stop = ndspy_sys::PtDspyError_PkDspyErrorStop as _,
}

/*
impl From<Error> for ndspy_sys::PtDspyError {
    fn from(e: Error) -> ndspy_sys::PtDspyError {
        e.into()
    }
}*/

enum Query {}

struct Flags;

//type FnDisplayOpen<'a> = dyn FnMut(&str, usize, usize) -> Error + 'a;
pub type FnOpen<'a> = dyn FnMut(
        // Filename.
        &str,
        // Width.
        usize,
        // Height.
        usize,
        // Pixel format.
        &[&str],
    ) -> Error
    + 'a;
//Result<(Vec<Format>, Flags), Error>;

pub type FnWrite<'a> = dyn FnMut(
        // Filename.
        &str,
        // Width.
        usize,
        // Height.
        usize,
        // x_min.
        usize,
        // x_max_plus_one
        usize,
        // y_min.
        usize,
        // y_max_plus_one,
        usize,
        // Pixel format.
        &[&str],
        // Pixel data.
        &[f32],
    ) -> Error
    + 'a;

pub type FnFinish<'a> = dyn FnMut(
        // Filename.
        &str,
        // Width.
        usize,
        // Height.
        usize,
        // Pixel format.
        &[&str],
        // Pixel data.
        &[f32],
    ) -> Error
    + 'a;

type FnQuery<'a> = dyn FnMut(Query, f32) -> Error + 'a;

pub struct FnOpenType<'a>(Box<Box<Box<FnOpen<'a>>>>);

impl<'a> FnOpenType<'a> {
    fn new<F>(fn_open: F) -> Self
    where
        F: FnMut(&str, usize, usize, &[&str]) -> Error + 'a,
    {
        FnOpenType(Box::new(Box::new(Box::new(fn_open))))
    }
}

impl nsi::CallbackTrait for FnOpenType<'_> {
    fn to_ptr(self) -> *const core::ffi::c_void {
        Box::into_raw(self.0) as *const _ as _
    }
}

pub struct FnWriteType<'a>(Box<Box<Box<FnWrite<'a>>>>);

impl<'a> FnWriteType<'a> {
    fn new<F>(fn_write: F) -> Self
    where
        F: FnMut(&str, usize, usize, usize, usize, usize, usize, &[&str], &[f32]) -> Error + 'a,
    {
        FnWriteType(Box::new(Box::new(Box::new(fn_write))))
    }
}

impl nsi::CallbackTrait for FnWriteType<'_> {
    fn to_ptr(self) -> *const core::ffi::c_void {
        Box::into_raw(self.0) as *const _ as _
    }
}

pub struct FnFinishType<'a>(Box<Box<Box<FnFinish<'a>>>>);

impl<'a> FnFinishType<'a> {
    fn new<F>(fn_finish: F) -> Self
    where
        F: FnMut(&str, usize, usize, &[&str], &[f32]) -> Error + 'a,
    {
        FnFinishType(Box::new(Box::new(Box::new(fn_finish))))
    }
}

impl nsi::CallbackTrait for FnFinishType<'_> {
    fn to_ptr(self) -> *const core::ffi::c_void {
        Box::into_raw(self.0) as *const _ as _
    }
}

struct FnQueryType<'a>(Box<Box<Box<FnQuery<'a>>>>);

struct DisplayData<'a> {
    name: &'a str,
    width: usize,
    height: usize,
    pixel_format: Vec<&'a str>,
    pixel_data: Vec<f32>,
    fn_open: Option<Box<Box<Box<FnOpen<'a>>>>>,
    fn_write: Option<Box<Box<Box<FnWrite<'a>>>>>,
    fn_finish: Option<Box<Box<Box<FnFinish<'a>>>>>,
    fn_query: Option<Box<Box<Box<FnQuery<'a>>>>>,
}

/// A utility function to get user parameters.
///
/// The template argument is the expected type of the resp. parameter.
///
/// # Arguments
///
/// * `name` - A string slice that holds the name of the parameter we
///   are searching for
/// * `parameter_count` - Number of parameters
/// * `parameter`       - Array of `ndspy_sys::UserParameter` structs to
///   search
///
/// # Example
///
/// ```
/// let associate_alpha =
///     1 == get_parameter::<i32>("associatealpha", _parameter_count, _parameter).unwrap_or(0);
/// ```
fn get_parameter<'a, T>(
    name: &str,
    type_: u8,
    len: usize,
    parameters: &[ndspy_sys::UserParameter],
) -> Option<&'a T> {
    for p in parameters.iter() {
        let p_name = unsafe { CStr::from_ptr(p.name) }.to_str().unwrap();

        if name == p_name && type_ == p.valueType as _ && len == p.valueCount as _ {
            if p.value != ptr::null() {
                return Some(unsafe { &*(p.value as *const T) });
            } else {
                // Parameter exists buy value is missing –
                // exit quietly.
                break;
            }
        }
    }

    None
}

fn get_parameter_mut<'a, T>(
    name: &str,
    type_: u8,
    len: usize,
    parameters: &mut [ndspy_sys::UserParameter],
) -> Option<&'a mut T> {
    for p in parameters.iter() {
        let p_name = unsafe { CStr::from_ptr(p.name) }.to_str().unwrap();

        if name == p_name && type_ == p.valueType as _ && len == p.valueCount as _ {
            if p.value != ptr::null_mut() {
                return Some(unsafe { &mut *(p.value as *mut T) });
            } else {
                // Parameter exists buy value is missing –
                // exit quietly.
                break;
            }
        }
    }

    None
}

fn get_parameter_triple_box<T: ?Sized>(
    name: &str,
    type_: u8,
    len: usize,
    parameters: &mut [ndspy_sys::UserParameter],
) -> Option<Box<Box<Box<T>>>> {
    for p in parameters.iter() {
        let p_name = unsafe { CStr::from_ptr(p.name) }.to_str().unwrap();

        if name == p_name && type_ == p.valueType as _ && len == p.valueCount as _ {
            if p.value != ptr::null_mut() {
                return Some(unsafe { Box::from_raw(p.value as *mut Box<Box<T>>) });
            } else {
                // Parameter exists buy value is missing –
                // exit quietly.
                break;
            }
        }
    }

    None
}

fn inspect_boxed_trait_object(foo: &FnOpenType) {
    let obj: &FnOpen = &foo.0;
    let tobj = unsafe { std::mem::transmute::<&FnOpen, TraitObject>(obj) };
    println!("{:X}, {:X}", tobj.data as usize, tobj.vtable as usize);
}

#[no_mangle]
pub extern "C" fn image_open(
    mut image_handle_ptr: *mut ndspy_sys::PtDspyImageHandle,
    _driver_name: *const c_char,
    output_filename: *const c_char,
    width: c_int,
    height: c_int,
    parameters_count: c_int,
    parameters: *const ndspy_sys::UserParameter,
    format_count: c_int,
    format: *mut ndspy_sys::PtDspyDevFormat,
    flag_stuff: *mut ndspy_sys::PtFlagStuff,
) -> ndspy_sys::PtDspyError {
    // FIXME: check that driver_name is "ferris".
    if (image_handle_ptr == ptr::null_mut()) || (output_filename == ptr::null_mut()) {
        return Error::BadParameters.into();
    }

    println!("Opening!");

    let mut parameters = unsafe {
        std::slice::from_raw_parts_mut(std::mem::transmute(parameters), parameters_count as _)
    };

    let mut format = unsafe { std::slice::from_raw_parts_mut(format, format_count as _) };

    // Collect format names as &str and force to f32.
    let pixel_format = format
        .iter_mut()
        .enumerate()
        .map(|format| {
            // Ensure all channels are sent to us as 32bit float.
            format.1.type_ = ndspy_sys::PkDspyFloat32;
            unsafe { CStr::from_ptr(format.1.name).to_str().unwrap() }
        })
        .collect::<Vec<_>>();

    let mut display_data = Box::new(DisplayData {
        name: unsafe { CStr::from_ptr(output_filename).to_str().unwrap() },
        width: width as _,
        height: height as _,
        pixel_format,
        pixel_data: vec![1.0f32; width as usize * height as usize * format.len()],
        fn_open: get_parameter_triple_box::<FnOpen>("callback.open", b'p', 1, parameters),
        fn_write: get_parameter_triple_box::<FnWrite>("callback.write", b'p', 1, parameters),
        fn_query: get_parameter_triple_box::<FnQuery>("callback.query", b'p', 1, parameters),
        fn_finish: get_parameter_triple_box::<FnFinish>("callback.finish", b'p', 1, parameters),
    });

    let error = if let Some(ref mut fn_open) = display_data.fn_open {
        fn_open(
            display_data.name,
            width as _,
            height as _,
            display_data.pixel_format.as_slice(),
        )
    } else {
        Error::None
    };

    unsafe {
        *image_handle_ptr = Box::into_raw(display_data) as _;
    }

    error.into()
}

#[no_mangle]
pub extern "C" fn image_query(
    image_handle_ptr: ndspy_sys::PtDspyImageHandle,
    query_type: ndspy_sys::PtDspyQueryType,
    data_len: c_int,
    mut data: *mut c_void,
) -> ndspy_sys::PtDspyError {
    println!("Query: {:?}", query_type);
    Error::Unsupported.into()
}

#[no_mangle]
pub extern "C" fn image_data(
    image_handle_ptr: ndspy_sys::PtDspyImageHandle,
    x_min: c_int,
    x_max_plus_one: c_int,
    y_min: c_int,
    y_max_plus_one: c_int,
    _entry_size: c_int,
    data: *const u8,
) -> ndspy_sys::PtDspyError {
    let mut display_data = unsafe { Box::from_raw(image_handle_ptr as *mut DisplayData) };

    // _entry_size is pixel_length in u8s, we need pixel length in f32s.
    let pixel_length = display_data.pixel_format.len();
    let pixel_data = unsafe {
        std::slice::from_raw_parts(
            data as *const f32,
            pixel_length * ((x_max_plus_one - x_min) * (y_max_plus_one - y_min)) as usize,
        )
    };

    let bucket_width = pixel_length * (x_max_plus_one - x_min) as usize;

    //println!("{:?}", pixel_data);
    let mut source_index = 0;
    for y in y_min as usize..y_max_plus_one as _ {
        let dest_index = y * display_data.width * pixel_length;

        // copy_from_slice() uses memcpy() behind the scenes.
        (&mut display_data.pixel_data[dest_index..dest_index + bucket_width])
            .copy_from_slice(&pixel_data[source_index..source_index + bucket_width]);

        //println!("{} - {}", dest_index, bucket_width);

        source_index += bucket_width;
    }

    let error = if let Some(ref mut fn_write) = display_data.fn_write {
        fn_write(
            display_data.name,
            display_data.width,
            display_data.height,
            x_min as _,
            x_max_plus_one as _,
            y_min as _,
            y_max_plus_one as _,
            &display_data.pixel_format,
            &display_data.pixel_data,
        )
    } else {
        Error::None
    };

    Box::into_raw(display_data);

    error.into()
}

#[no_mangle]
pub extern "C" fn image_close(
    image_handle_ptr: ndspy_sys::PtDspyImageHandle,
) -> ndspy_sys::PtDspyError {
    println!("Closing!");
    let mut display_data = unsafe { Box::from_raw(image_handle_ptr as *mut DisplayData) };

    let error = if let Some(ref mut fn_finish) = display_data.fn_finish {
        fn_finish(
            display_data.name,
            display_data.width,
            display_data.height,
            &display_data.pixel_format,
            &display_data.pixel_data,
        )
    } else {
        Error::None
    };

    // These boxes somehow get deallocated twice if we don't suppress this here.
    // No idea why.
    if let Some(fn_open) = display_data.fn_open {
        Box::into_raw(fn_open);
    }
    if let Some(fn_write) = display_data.fn_write {
        Box::into_raw(fn_write);
    }
    if let Some(fn_query) = display_data.fn_query {
        Box::into_raw(fn_query);
    }
    if let Some(fn_finish) = display_data.fn_finish {
        Box::into_raw(fn_finish);
    }

    error.into()
}

/*
let data = FnDisplayDataType(Box::new(
    |xmin: usize,
     xmax_plus_one: usize,
     ymin: usize,
     ymax_plus_one: usize,
     entry_size: usize,
     data: &[f32]| {
        println!(
            "{},{}-{},{}",
            xmin,
            ymin,
            xmax_plus_one - 1,
            ymax_plus_one - 1
        );
        Error::None
    },
));
*/

fn test<'a>(open: FnOpenType<'a>) {
    {
        let ctx = nsi::Context::new(&[]).unwrap();

        ctx.create("foo_driver", nsi::NodeType::OutputDriver, &[]);

        ctx.set_attribute(
            "foo_driver",
            &[
                nsi::string!("drivername", "ferris"),
                nsi::reference!("callback.open", Some(&open)),
            ],
        );
    }
}

#[derive(Debug)]
struct Payload(Box<[u32; 10]>);

fn write_exr(name: &str, width: usize, height: usize, pixel_length: usize, pixel_data: &[f32]) {

    //println!("Writing EXR ... {:?}", pixel_data);

    let sample = |position: Vec2<usize>| {
        let index = pixel_length * (position.x() + position.y() * width);

        Pixel::rgba(
            pixel_data[index + 0],
            pixel_data[index + 1],
            pixel_data[index + 2],
            pixel_data[index + 3],
        )
    };

    let mut image_info = ImageInfo::rgba((width, height), SampleType::F32);

    // write it to a file with all cores in parallel
    image_info
        //.with_encoding(encoding)
        //.remove_excess()
        .write_pixels_to_file(
            name.clone(),
            // this will actually generate the pixels in parallel on all cores
            write_options::high(),
            &sample,
        )
        .unwrap();
}

pub fn main() {

    // so simply do not rely on borrow lifetimes for anything other than ensuring your own values' scope is limited when working with FFI
    unsafe {
        ndspy_sys::DspyRegisterDriver(
            b"ferris\0" as *const u8 as _,
            Some(image_open as _),
            Some(image_data as _),
            Some(image_close as _),
            Some(image_query as _),
        );
    }

    let mut width = 0usize;
    let mut height = 0usize;
    let mut pixel_len = 0usize;

    let mut final_pixel_data = Vec::<f32>::new();

    {
        let payload = Payload(Box::new([42; 10]));

        let open = FnOpenType::new(|_name: &str, w: usize, h: usize, format: &[&str]| {
            width = w;
            height = h;
            pixel_len = format.len();
            Error::None
        });

        let write = FnWriteType::new(
            |_name: &str,
             width: usize,
             height: usize,
             x_min: usize,
             x_max_plus_one: usize,
             y_min: usize,
             y_max_plus_one: usize,
             pixel_format: &[&str],
             pixel_data: &[f32]| {
                Error::None
            },
        );

        let finish = FnFinishType::new(
            |name: &str,
             width: usize,
             height: usize,
             pixel_format: &[&str],
             pixel_data: &[f32]| {
                write_exr(name, width, height, pixel_format.len(), pixel_data);
                Error::None
            },
        );

        {
            let mut polyhedron = p_ops::Polyhedron::tetrahedron();
            polyhedron.meta(None, None, None, false, true);
            polyhedron.normalize();
            polyhedron.gyro(Some(1. / 3.), Some(0.1), true);
            polyhedron.normalize();
            polyhedron.kis(Some(-0.2), None, true, true);
            polyhedron.normalize();

            //inspect_boxed_trait_object(&open);
            render::nsi_render(&polyhedron, &[0.0f64; 16], 1, false, open, finish, &payload);
        }

    }
}
