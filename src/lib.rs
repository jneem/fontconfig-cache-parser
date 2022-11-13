use bytemuck::AnyBitPattern;
use std::os::raw::{c_int, c_uint};

type Result<T> = std::result::Result<T, Error>;

// Many of fontconfig's values are stored as a tagged union. But because
// there's no layout guarantees for tagged unions in rust, we read them
// in the C layout, as a combination of `c_int` tag and an untagged union.
//
// This is the untagged union part, which we don't expose publicly.
#[repr(C)]
#[derive(AnyBitPattern, Copy, Clone)]
union ValueUnion {
    s: PtrOffset<u8>,
    i: c_int,
    b: c_int,
    d: f64,
    m: PtrOffset<()>, // TODO
    c: PtrOffset<CharSet>,
    f: PtrOffset<()>,
    l: PtrOffset<()>, // TODO
    r: PtrOffset<()>, // TODO
}

/// A wrapper around fontconfig's `FcValue` type.
#[derive(Clone, Debug)]
pub enum Value<'a> {
    Unknown,
    Void,
    Int(c_int),
    Float(f64),
    String(Ptr<'a, u8>),
    Bool(c_int),
    /// Not yet supported
    Matrix(Ptr<'a, ()>),
    CharSet(Ptr<'a, CharSet>),
    /// Not yet supported
    FtFace(Ptr<'a, ()>),
    /// Not yet supported
    LangSet(Ptr<'a, ()>),
    /// Not yet supported
    Range(Ptr<'a, ()>),
}

#[derive(AnyBitPattern, Copy, Clone)]
#[repr(C)]
pub struct ValueData {
    ty: c_int,
    val: ValueUnion,
}

impl<'a> Ptr<'a, ValueData> {
    /// Converts the raw
    pub fn to_value(&self) -> Result<Value<'a>> {
        use Value::*;
        let payload = self.payload()?;

        unsafe {
            Ok(match payload.ty {
                -1 => Unknown,
                0 => Void,
                1 => Int(payload.val.i),
                2 => Float(payload.val.d),
                3 => String(self.relative_offset(payload.val.s)?),
                4 => Bool(payload.val.b),
                5 => Matrix(self.relative_offset(payload.val.m)?),
                6 => CharSet(self.relative_offset(payload.val.c)?),
                7 => FtFace(self.relative_offset(payload.val.f)?),
                8 => LangSet(self.relative_offset(payload.val.l)?),
                9 => Range(self.relative_offset(payload.val.r)?),
                _ => return Err(Error::InvalidEnumTag(payload.ty)),
            })
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub enum Object {
    Invalid = 0,
    Family,
    FamilyLang,
    Style,
    StyleLang,
    FullName,
    FullNameLang,
    Slant,
    Weight,
    Width,
    Size,
    Aspect,
    PixelSize,
    Spacing,
    Foundry,
    AntiAlias,
    HintStyle,
    Hinting,
    VerticalLayout,
    AutoHint,
    GlobalAdvance,
    File,
    Index,
    Rasterizer,
    Outline,
    Scalable,
    Dpi,
    Rgba,
    Scale,
    MinSpace,
    CharWidth,
    CharHeight,
    Matrix,
    CharSet,
    Lang,
    FontVersion,
    Capability,
    FontFormat,
    Embolden,
    EmbeddedBitmap,
    Decorative,
    LcdFilter,
    NameLang,
    FontFeatures,
    PrgName,
    Hash,
    PostscriptName,
    Color,
    Symbol,
    FontVariations,
    Variable,
    FontHasHint,
    Order,
}

const MAX_OBJECT: c_int = Object::Order as c_int;

impl TryFrom<c_int> for Object {
    type Error = Error;

    fn try_from(value: c_int) -> Result<Self> {
        if value <= MAX_OBJECT {
            Ok(unsafe { std::mem::transmute(value) })
        } else {
            Err(Error::InvalidObjectTag(value))
        }
    }
}

/// A relative offset to another struct in the cache, which is encoded as a pointer in fontconfig.
///
/// See [`Offset`] for more on offsets in fontconfig and how we handle them in this crate.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PtrOffset<T: Copy>(isize, std::marker::PhantomData<T>);

unsafe impl<T: Copy> bytemuck::Zeroable for PtrOffset<T> {}
unsafe impl<T: Copy + 'static> bytemuck::Pod for PtrOffset<T> {}

/// This is basically equivalent to `TryInto<Offset<T>, Error=Error>`, but having this
/// alias makes type inference work better.
pub trait IntoOffset: AnyBitPattern + Copy {
    type Item: AnyBitPattern + Copy;

    fn into_offset(self) -> Result<Offset<Self::Item>>;
}

impl<T: AnyBitPattern + Copy> IntoOffset for PtrOffset<T> {
    type Item = T;

    fn into_offset(self) -> Result<Offset<T>> {
        if self.0 & 1 == 0 {
            Err(Error::BadPointer(self.0))
        } else {
            Ok(Offset(self.0 & !1, std::marker::PhantomData))
        }
    }
}

impl<T: AnyBitPattern + Copy> IntoOffset for Offset<T> {
    type Item = T;

    fn into_offset(self) -> Result<Offset<T>> {
        Ok(self)
    }
}

/// A relative offset to another struct in the cache.
///
/// Fontconfig's cache is stuctured as a collection of structs.
/// These structs are encoded in the cache by
/// writing their bytes into a cache file. (So the format is architecture-dependent.)
/// The structs reference each other using relative offsets, so for example in the struct
///
/// ```C
/// struct FcPattern {
///     int num;
///     int size;
///     intptr_t elts_offset;
///     int ref;
/// }
/// ```
///
/// the elements `num`, `size`, and `ref` are just plain old data, and the element `elts_offset`
/// says that there is some other struct (which happens to be an `FcPatternElt` in this case)
/// stored at the location `base_offset + elts_offset`, where `base_offset` is the offset
/// of the `FcPattern`. Note that `elts_offset` is signed: it can be negative.
///
/// We encode these offsets using `Offset`, so for example the struct above gets translated to
///
/// ```rust
/// struct Pattern {
///     num: c_int,
///     size: c_int,
///     elts_offset: Offset<PatternElt>,
///     ref: c_int,
/// }
/// ```
///
/// Sometimes, the structs in fontconfig contain pointers instead of offsets, like for example
///
/// ```C
/// struct FcPatternElt {
///     FcObject object;
///     FcValueList *values;
/// }
///
/// In this case, fontconfig actually handles two cases: if the lowest-order bit of `values` is 0
/// it's treated as a normal pointer, but if the lowest-order bit is 1 then that bit is set
/// to zero and `values` is treated as an offset. When the struct came from a cache file that
/// was serialized to disk (which we always are in this crate), it should always be in the "offset" case.
/// That is, these pointers get treated almost the same as offsets, except that we need to
/// sanity-check the low-order bit and then set it to zero. We encode these as `PtrOffset`,
/// so for example the struct above gets translated to
///
/// ```rust
/// struct PatternElt {
///     object: c_int,
///     values: PtrOffset<ValueList>,
/// }
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Offset<T: Copy>(isize, std::marker::PhantomData<T>);

fn offset<T: Copy>(off: isize) -> Offset<T> {
    Offset(off, std::marker::PhantomData)
}

unsafe impl<T: Copy> bytemuck::Zeroable for Offset<T> {}
unsafe impl<T: Copy + 'static> bytemuck::Pod for Offset<T> {}

impl std::fmt::Debug for ValueData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", self.ty))
        // TODO: write the rest
    }
}

#[derive(AnyBitPattern, Copy, Clone, Debug)]
#[repr(C)]
pub struct ValueList {
    next: PtrOffset<ValueList>,
    value: ValueData,
    binding: c_int,
}

impl<'a> Ptr<'a, ValueList> {
    fn value(&self) -> Result<Ptr<'a, ValueData>> {
        self.relative_offset(offset(std::mem::size_of_val(&self.payload()?.next) as isize))
    }
}

pub struct ValueListIter<'a> {
    next: Option<Result<Ptr<'a, ValueList>>>,
}

impl<'a> Iterator for ValueListIter<'a> {
    type Item = Result<Ptr<'a, ValueData>>;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.next.take();
        if let Some(Ok(next)) = next {
            match next.payload() {
                Ok(next_payload) => {
                    if next_payload.next.0 == 0 {
                        self.next = None;
                    } else {
                        self.next = Some(next.relative_offset(next_payload.next));
                    }
                }
                Err(e) => {
                    self.next = Some(Err(e));
                }
            }
            Some(next.value())
        } else if let Some(Err(e)) = next {
            Some(Err(e))
        } else {
            None
        }
    }
}

#[derive(AnyBitPattern, Copy, Clone, Debug)]
#[repr(C)]
pub struct Pattern {
    // The number of elements.
    pub num: c_int,
    // The capacity of the elements array. For serialized data, it's probably
    // the same as `num`.
    _size: c_int,
    pub elts_offset: Offset<PatternElt>,
    ref_count: c_int,
}

impl Ptr<'_, Pattern> {
    pub fn elts(&self) -> Result<impl Iterator<Item = Ptr<PatternElt>> + '_> {
        let payload = self.payload()?;
        let elts = self.relative_offset(payload.elts_offset)?;
        elts.array(payload.num)
    }
}

#[derive(AnyBitPattern, Copy, Clone, Debug)]
#[repr(C)]
pub struct PatternElt {
    pub object: c_int,
    pub values: PtrOffset<ValueList>,
}

impl Ptr<'_, PatternElt> {
    pub fn values(&self) -> Result<impl Iterator<Item = Result<Ptr<ValueData>>> + '_> {
        Ok(ValueListIter {
            next: Some(Ok(self.relative_offset(self.payload()?.values)?)),
        })
    }
}

#[derive(AnyBitPattern, Copy, Clone, Debug)]
#[repr(C)]
pub struct FontSet {
    pub nfont: c_int,
    // FIXME: what's this?
    pub sfont: c_int,
    pub fonts: PtrOffset<PtrOffset<Pattern>>,
}

impl Ptr<'_, FontSet> {
    pub fn fonts(&self) -> Result<impl Iterator<Item = Result<Ptr<Pattern>>> + '_> {
        let payload = self.payload()?;
        let fonts = self.relative_offset(payload.fonts)?.array(payload.nfont)?;
        let me = self.clone();
        Ok(fonts.map(move |font_offset| me.relative_offset(font_offset.payload()?)))
    }
}

#[derive(AnyBitPattern, Copy, Clone, Debug)]
#[repr(C)]
pub struct CharSet {
    pub ref_count: c_int,
    // Length of both of the following arrays
    pub num: c_int,
    // Array of offsets to leaves. Each offset is relative to the start of the array.
    pub leaves: Offset<Offset<CharSetLeaf>>,
    pub numbers: Offset<u16>,
}

impl<'buf> Ptr<'buf, CharSet> {
    pub fn leaves(&self) -> Result<impl Iterator<Item = Result<Ptr<'buf, CharSetLeaf>>> + 'buf> {
        let payload = self.payload()?;
        let leaf_array = self.relative_offset(payload.leaves)?;
        Ok(leaf_array
            .array(payload.num)?
            .map(move |leaf_offset| leaf_array.relative_offset(leaf_offset.payload()?)))
    }

    pub fn numbers(&self) -> Result<impl Iterator<Item = Ptr<'buf, u16>> + 'buf> {
        let payload = self.payload()?;
        self.relative_offset(payload.numbers)?.array(payload.num)
    }

    pub fn chunks(&self) -> Result<impl Iterator<Item = Result<CharSetChunk>> + 'buf> {
        Ok(self.leaves()?.zip(self.numbers()?).map(|(leaf, number)| {
            Ok(CharSetChunk {
                offset: number.payload()?,
                map: leaf?.payload()?.map,
            })
        }))
    }
}

#[derive(AnyBitPattern, Copy, Clone, Debug)]
#[repr(C)]
pub struct CharSetLeaf {
    map: [u32; 4],
}

#[derive(Copy, Clone, Debug)]
pub struct CharSetChunk {
    pub offset: u16,
    pub map: [u32; 4],
}

#[derive(AnyBitPattern, Copy, Clone, Debug)]
#[repr(C)]
pub struct StrSet {
    pub ref_count: c_int,
    pub num: c_int,
    pub size: c_int,
    pub strs: PtrOffset<PtrOffset<u8>>,
    pub control: c_uint,
}

#[derive(AnyBitPattern, Copy, Clone, Debug)]
#[repr(C)]
pub struct LangSet {
    extra: PtrOffset<StrSet>,
    map_size: u32,
}

#[derive(AnyBitPattern, Copy, Clone, Debug)]
#[repr(C)]
pub struct Cache {
    pub magic: c_uint,
    pub version: c_int,
    pub size: isize,
    pub dir: Offset<u8>,
    pub dirs: Offset<Offset<u8>>,
    pub dirs_count: c_int,
    pub set: Offset<FontSet>,
    pub checksum: c_int,
    pub checksum_nano: c_int,
}

#[derive(Clone)]
pub struct Ptr<'a, S> {
    pub offset: isize,
    pub buf: &'a [u8],
    marker: std::marker::PhantomData<S>,
}

impl<'a, S> std::fmt::Debug for Ptr<'a, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ptr").field("offset", &self.offset).finish()
    }
}

#[derive(Clone, Debug)]
struct DecodeIterator<'a, T> {
    buf: &'a [u8],
    offset: usize,
    remaining: isize,
    marker: std::marker::PhantomData<T>,
}

impl<'a, T> DecodeIterator<'a, T> {
    fn new(buf: &'a [u8], offset: isize, size: c_int) -> Result<Self> {
        let len = std::mem::size_of::<T>();
        let total_len = len
            .checked_mul(size as usize)
            .ok_or(Error::BadLength(size as isize))?;

        if offset < 0 {
            Err(Error::BadOffset(offset))
        } else {
            let end = (offset as usize)
                .checked_add(total_len)
                .ok_or(Error::BadLength(size as isize))?;
            if end > buf.len() {
                Err(Error::BadOffset(end as isize))
            } else {
                Ok(DecodeIterator {
                    buf,
                    offset: offset as usize,
                    remaining: size as isize,
                    marker: std::marker::PhantomData,
                })
            }
        }
    }
}

impl<'a, T: AnyBitPattern> Iterator for DecodeIterator<'a, T> {
    type Item = Ptr<'a, T>;

    fn next(&mut self) -> Option<Ptr<'a, T>> {
        if self.remaining <= 0 {
            None
        } else {
            let len = std::mem::size_of::<T>();
            let ret = Ptr {
                buf: self.buf,
                offset: self.offset as isize,
                marker: std::marker::PhantomData,
            };
            self.offset += len;
            self.remaining -= 1;
            Some(ret)
        }
    }
}

impl<'buf> Ptr<'buf, u8> {
    pub fn str(&self) -> Result<&'buf [u8]> {
        let offset = self.offset;
        if offset < 0 || offset > self.buf.len() as isize {
            Err(Error::BadOffset(offset))
        } else {
            let buf = &self.buf[(offset as usize)..];
            let null_offset = buf
                .iter()
                .position(|&c| c == 0)
                .ok_or(Error::UnterminatedString(offset))?;
            Ok(&buf[..null_offset])
        }
    }
}

impl<'a, S: AnyBitPattern> Ptr<'a, S> {
    pub fn relative_offset<Off: IntoOffset>(&self, offset: Off) -> Result<Ptr<'a, Off::Item>> {
        let offset = offset.into_offset()?;
        Ok(Ptr {
            buf: self.buf,
            offset: self
                .offset
                .checked_add(offset.0)
                .ok_or(Error::BadOffset(offset.0))?,
            marker: std::marker::PhantomData,
        })
    }

    pub fn payload(&self) -> Result<S> {
        let len = std::mem::size_of::<S>() as isize;
        if self.offset + len >= self.buf.len() as isize {
            Err(Error::BadOffset(self.offset))
        } else {
            // We checked at construction time that the buffer has enough elements for the payload,
            // so the slice will succeed.
            Ok(bytemuck::try_pod_read_unaligned(
                &self.buf[(self.offset as usize)..((self.offset + len) as usize)],
            )
            .expect("but we checked the length..."))
        }
    }

    fn array(&self, count: c_int) -> Result<impl Iterator<Item = Ptr<'a, S>> + 'a> {
        Ok(DecodeIterator::new(self.buf, self.offset, count)?)
    }
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    #[error("Invalid magic number {0:#x}")]
    BadMagic(c_uint),

    #[error("Unsupported version {0}")]
    UnsupportedVersion(c_int),

    #[error("Bad pointer {0}")]
    BadPointer(isize),

    #[error("Bad offset {0}")]
    BadOffset(isize),

    #[error("Bad length {0}")]
    BadLength(isize),

    #[error("Invalid enum tag {0}")]
    InvalidEnumTag(c_int),

    #[error("Invalid object tag {0}")]
    InvalidObjectTag(c_int),

    #[error("Unterminated string at {0}")]
    UnterminatedString(isize),

    #[error("Wrong size: header expects {expected} bytes, buffer is {actual} bytes")]
    WrongSize { expected: isize, actual: isize },
}

impl Cache {
    pub fn read(buf: &[u8]) -> Result<Ptr<'_, Cache>> {
        use Error::*;

        let len = std::mem::size_of::<Cache>();
        if buf.len() < len {
            Err(WrongSize {
                expected: len as isize,
                actual: buf.len() as isize,
            })
        } else {
            let cache: Cache = bytemuck::try_pod_read_unaligned(&buf[0..len])
                .expect("but we checked the length...");

            if cache.magic != 4228054020 {
                Err(BadMagic(cache.magic))
            } else if cache.version != 7 && cache.version != 8 {
                Err(UnsupportedVersion(cache.version))
            } else if cache.size != buf.len() as isize {
                Err(WrongSize {
                    expected: cache.size,
                    actual: buf.len() as isize,
                })
            } else {
                Ok(Ptr {
                    buf,
                    offset: 0,
                    marker: std::marker::PhantomData,
                })
            }
        }
    }
}

impl<'buf> Ptr<'buf, Cache> {
    pub fn set(&self) -> Result<Ptr<'buf, FontSet>> {
        self.relative_offset(self.payload()?.set)
    }
}
