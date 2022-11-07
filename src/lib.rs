use bytemuck::AnyBitPattern;
use std::os::raw::{c_int, c_uint};

type Result<T> = std::result::Result<T, CacheFormatError>;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub enum Type {
    Unknown = -1,
    Void,
    Integer,
    Double,
    String,
    Bool,
    Matrix,
    CharSet,
    FtFace,
    LangSet,
    Range,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub enum ValueBinding {
    Weak,
    Strong,
    Same,
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
    type Error = CacheFormatError;

    fn try_from(value: c_int) -> Result<Self> {
        if value <= MAX_OBJECT {
            Ok(unsafe { std::mem::transmute(value) })
        } else {
            Err(CacheFormatError::InvalidObjectTag(value))
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PtrOffset<T: Copy>(isize, std::marker::PhantomData<T>);

unsafe impl<T: Copy> bytemuck::Zeroable for PtrOffset<T> {}
unsafe impl<T: Copy + 'static> bytemuck::Pod for PtrOffset<T> {}

pub trait IntoOffset<T: Copy> {
    fn into_offset(self) -> Result<Offset<T>>;
}

impl<T: Copy> IntoOffset<T> for PtrOffset<T> {
    fn into_offset(self) -> Result<Offset<T>> {
        if self.0 & 1 == 0 {
            Err(CacheFormatError::BadPointer(self.0))
        } else {
            Ok(Offset(self.0 & !1, std::marker::PhantomData))
        }
    }
}

impl<T: Copy> IntoOffset<T> for Offset<T> {
    fn into_offset(self) -> Result<Offset<T>> {
        Ok(self)
    }
}

impl<T: Copy> IntoOffset<T> for usize {
    fn into_offset(self) -> Result<Offset<T>> {
        Ok(Offset(self as isize, std::marker::PhantomData))
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Offset<T: Copy>(isize, std::marker::PhantomData<T>);

unsafe impl<T: Copy> bytemuck::Zeroable for Offset<T> {}
unsafe impl<T: Copy + 'static> bytemuck::Pod for Offset<T> {}

impl<'a, T: Copy + AnyBitPattern> Encoded<'a, Offset<T>> {
    pub fn deref_offset(mut self, base_offset: isize) -> Result<Encoded<'a, T>> {
        self.offset = base_offset;
        self.decode(self.payload)
    }
}

impl<'a, T: Copy + AnyBitPattern> Encoded<'a, PtrOffset<T>> {
    pub fn deref_offset(mut self, base_offset: isize) -> Result<Encoded<'a, T>> {
        self.offset = base_offset;
        self.decode(self.payload)
    }
}

#[repr(C)]
#[derive(AnyBitPattern, Copy, Clone)]
pub union ValueUnion {
    s: PtrOffset<u8>,
    i: c_int,
    b: c_int,
    d: f64,
    m: PtrOffset<()>, // TODO
    c: PtrOffset<()>, // TODO
    f: PtrOffset<()>,
    l: PtrOffset<()>, // TODO
    r: PtrOffset<()>, // TODO
}

#[derive(Copy, Clone, Debug)]
pub enum ValueEnum {
    Unknown,
    Void,
    Int(c_int),
    Float(f64),
    String(PtrOffset<u8>),
    Bool(c_int),
    Matrix(PtrOffset<()>),
    CharSet(PtrOffset<()>),
    FtFace(PtrOffset<()>),
    LangSet(PtrOffset<()>),
    Range(PtrOffset<()>),
}

#[derive(AnyBitPattern, Copy, Clone)]
#[repr(C)]
pub struct Value {
    ty: c_int, // Actually a Type, but we need to check it (TODO)
    val: ValueUnion,
}

impl Value {
    pub fn to_enum(&self) -> Result<ValueEnum> {
        use ValueEnum::*;

        unsafe {
            Ok(match self.ty {
                -1 => Unknown,
                0 => Void,
                1 => Int(self.val.i),
                2 => Float(self.val.d),
                3 => String(self.val.s),
                4 => Bool(self.val.b),
                5 => Matrix(self.val.m),
                6 => CharSet(self.val.c),
                7 => FtFace(self.val.f),
                8 => LangSet(self.val.l),
                9 => Range(self.val.r),
                _ => return Err(CacheFormatError::InvalidEnumTag(self.ty)),
            })
        }
    }
}

impl<'a> Encoded<'a, Value> {
    pub fn to_enum(&self) -> Result<Encoded<'a, ValueEnum>> {
        Ok(Encoded {
            buf: self.buf,
            offset: self.offset,
            payload: self.payload.to_enum()?,
        })
    }
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", self.ty))
        // TODO: write the rest
    }
}

#[derive(AnyBitPattern, Copy, Clone, Debug)]
#[repr(C)]
pub struct ValueList {
    next: PtrOffset<ValueList>,
    value: Value,
    binding: c_int,
}

impl<'a> Encoded<'a, ValueList> {
    fn value(&self) -> Result<Encoded<'a, Value>> {
        self.decode(std::mem::size_of_val(&self.payload.next))
    }
}

pub struct ValueListIter<'a> {
    next: Option<Result<Encoded<'a, ValueList>>>,
}

impl<'a> Iterator for ValueListIter<'a> {
    type Item = Result<Encoded<'a, Value>>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(Ok(next)) = self.next.take() {
            if next.payload.next.0 == 0 {
                self.next = None;
            } else {
                self.next = Some(next.decode(next.payload.next));
            }
            Some(next.value())
        } else if let Some(Err(e)) = self.next.take() {
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

impl Encoded<'_, Pattern> {
    pub fn elts(&self) -> Result<impl Iterator<Item = Encoded<PatternElt>> + '_> {
        self.decode_array(self.payload.elts_offset, self.payload.num)
    }
}

#[derive(AnyBitPattern, Copy, Clone, Debug)]
#[repr(C)]
pub struct PatternElt {
    pub object: c_int,
    pub values: PtrOffset<ValueList>,
}

impl Encoded<'_, PatternElt> {
    pub fn values(&self) -> Result<impl Iterator<Item = Result<Encoded<Value>>> + '_> {
        Ok(ValueListIter {
            next: Some(Ok(self.decode(self.payload.values)?)),
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

impl Encoded<'_, FontSet> {
    pub fn fonts(&self) -> Result<impl Iterator<Item = Result<Encoded<Pattern>>> + '_> {
        let base_offset = self.offset;
        Ok(self
            .decode_array(self.payload.fonts, self.payload.nfont)?
            .map(move |font_offset| font_offset.deref_offset(base_offset)))
    }
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
pub struct Encoded<'a, S> {
    pub offset: isize,
    pub buf: &'a [u8],
    pub payload: S,
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
            .ok_or(CacheFormatError::BadLength(size as isize))?;

        if offset < 0 {
            Err(CacheFormatError::BadOffset(offset))
        } else {
            let end = (offset as usize)
                .checked_add(total_len)
                .ok_or(CacheFormatError::BadLength(size as isize))?;
            if end > buf.len() {
                Err(CacheFormatError::BadOffset(end as isize))
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
    type Item = Encoded<'a, T>;

    fn next(&mut self) -> Option<Encoded<'a, T>> {
        if self.remaining <= 0 {
            None
        } else {
            let len = std::mem::size_of::<T>();
            // We checked at construction time that the buffer has enough elements for the whole
            // iterator, so the slice will succeed.
            let payload =
                bytemuck::try_pod_read_unaligned(&self.buf[self.offset..(self.offset + len)])
                    .expect("but we checked the length...");
            let ret = Encoded {
                buf: self.buf,
                offset: self.offset as isize,
                payload,
            };
            self.offset += len;
            self.remaining -= 1;
            Some(ret)
        }
    }
}

impl<'a, S> Encoded<'a, S> {
    pub fn offset<T: AnyBitPattern>(&self, offset: impl IntoOffset<T>) -> Result<isize> {
        let offset = offset.into_offset()?;
        self.offset
            .checked_add(offset.0)
            .ok_or(CacheFormatError::BadOffset(offset.0))
    }

    pub fn decode<T: AnyBitPattern>(&self, offset: impl IntoOffset<T>) -> Result<Encoded<'a, T>> {
        let offset = self.offset(offset)?;
        let len = std::mem::size_of::<T>() as isize;
        if offset < 0 || len + offset > self.buf.len() as isize {
            Err(CacheFormatError::BadOffset(offset))
        } else {
            let payload = bytemuck::try_pod_read_unaligned(
                &self.buf[(offset as usize)..((offset + len) as usize)],
            )
            .expect("but we checked the length...");
            Ok(Encoded {
                buf: self.buf,
                offset,
                payload,
            })
        }
    }

    pub fn decode_str(&self, offset: impl IntoOffset<u8>) -> Result<&'a [u8]> {
        let offset = self.offset(offset)?;
        if offset < 0 || offset > self.buf.len() as isize {
            Err(CacheFormatError::BadOffset(offset))
        } else {
            let buf = &self.buf[(offset as usize)..];
            let null_offset = buf
                .iter()
                .position(|&c| c == 0)
                .ok_or(CacheFormatError::UnterminatedString(offset))?;
            Ok(&buf[..null_offset])
        }
    }

    fn decode_array<T: AnyBitPattern>(
        &self,
        offset: impl IntoOffset<T>,
        count: c_int,
    ) -> Result<impl Iterator<Item = Encoded<T>>> {
        let offset = self.offset(offset)?;
        Ok(DecodeIterator::new(self.buf, offset, count)?)
    }
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum CacheFormatError {
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
    pub fn read(buf: &[u8]) -> Result<Encoded<'_, Cache>> {
        use CacheFormatError::*;

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
                Ok(Encoded {
                    buf,
                    offset: 0,
                    payload: bytemuck::try_pod_read_unaligned(&buf[0..len])
                        .expect("but we checked the length..."),
                })
            }
        }
    }
}
