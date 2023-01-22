#![deny(missing_docs)]

//! A crate for parsing fontconfig cache files.
//!
//! The fontconfig cache format is a C-style binary format, containing a maze of twisty structs all alike,
//! with lots of pointers from one to another. This makes it pretty inefficient to parse the whole file,
//! especially if you're only interested in a few parts. The expected workflow of this crate is:
//!
//! 1. You read the cache file into memory (possibly using `mmap` if the file is large and performance is important).
//! 2. You construct a [`Cache`](crate::Cache::from_bytes), borrowing the memory chunk.
//! 3. You follow the various methods on `Cache` to get access to the information you want.
//!    As you follow those methods, the data will be read incrementally from the memory chunk you
//!    created in part 1.

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
    c: PtrOffset<CharSetData>,
    f: PtrOffset<()>,
    l: PtrOffset<()>, // TODO
    r: PtrOffset<()>, // TODO
}

/// A dynamically typed value.
///
/// This is a wrapper around fontconfig's `FcValue` type.
#[allow(missing_docs)]
#[derive(Clone, Debug)]
pub enum Value<'buf> {
    Unknown,
    Void,
    Int(c_int),
    Float(f64),
    String(Ptr<'buf, u8>),
    Bool(c_int),
    /// Not yet supported
    Matrix(Ptr<'buf, ()>),
    CharSet(CharSet<'buf>),
    /// Not yet supported
    FtFace(Ptr<'buf, ()>),
    /// Not yet supported
    LangSet(Ptr<'buf, ()>),
    /// Not yet supported
    Range(Ptr<'buf, ()>),
}

/// Fontconfig's `FcValue` data type, in the raw serialized format.
#[derive(AnyBitPattern, Copy, Clone)]
#[repr(C)]
pub struct ValueData {
    ty: c_int,
    val: ValueUnion,
}

impl<'buf> Ptr<'buf, ValueData> {
    /// Converts the raw C representation to an enum.
    pub fn to_value(&self) -> Result<Value<'buf>> {
        use Value::*;
        let payload = self.deref()?;

        unsafe {
            Ok(match payload.ty {
                -1 => Unknown,
                0 => Void,
                1 => Int(payload.val.i),
                2 => Float(payload.val.d),
                3 => String(self.relative_offset(payload.val.s)?),
                4 => Bool(payload.val.b),
                5 => Matrix(self.relative_offset(payload.val.m)?),
                6 => CharSet(crate::CharSet(self.relative_offset(payload.val.c)?)),
                7 => FtFace(self.relative_offset(payload.val.f)?),
                8 => LangSet(self.relative_offset(payload.val.l)?),
                9 => Range(self.relative_offset(payload.val.r)?),
                _ => return Err(Error::InvalidEnumTag(payload.ty)),
            })
        }
    }
}

/// All the different object types supported by fontconfig.
///
/// (We currently only actually handle a few of these.)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(missing_docs)]
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
    /// Into an offset of what type?
    type Item: AnyBitPattern + Copy;

    /// Turns `self` into an `Offset`.
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
/// # Implementation details
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
/// ```ignore
/// struct PatternData {
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
/// ```
///
/// In this case, fontconfig actually handles two cases: if the lowest-order bit of `values` is 0
/// it's treated as a normal pointer, but if the lowest-order bit is 1 then that bit is set
/// to zero and `values` is treated as an offset. When the struct came from a cache file that
/// was serialized to disk (which we always are in this crate), it should always be in the "offset" case.
/// That is, these pointers get treated almost the same as offsets, except that we need to
/// sanity-check the low-order bit and then set it to zero. We encode these as `PtrOffset`,
/// so for example the struct above gets translated to
///
/// ```ignore
/// struct PatternEltData {
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

/// A linked list of [`Value`]s, in the raw serialized format.
#[derive(AnyBitPattern, Copy, Clone, Debug)]
#[repr(C)]
pub struct ValueListData {
    next: PtrOffset<ValueListData>,
    value: ValueData,
    binding: c_int,
}

/// A linked list of [`Value`]s.
#[derive(Clone, Debug)]
struct ValueList<'buf>(pub Ptr<'buf, ValueListData>);

impl<'buf> ValueList<'buf> {
    fn value(&self) -> Result<Value<'buf>> {
        self.0
            .relative_offset(offset(std::mem::size_of_val(&self.0.deref()?.next) as isize))
            .and_then(|val_ptr| val_ptr.to_value())
    }
}

/// An iterator over [`Value`]s.
#[derive(Clone, Debug)]
struct ValueListIter<'buf> {
    next: Option<Result<ValueList<'buf>>>,
}

impl<'buf> Iterator for ValueListIter<'buf> {
    type Item = Result<Value<'buf>>;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.next.take();
        if let Some(Ok(next)) = next {
            match next.0.deref() {
                Ok(next_payload) => {
                    if next_payload.next.0 == 0 {
                        self.next = None;
                    } else {
                        self.next = Some(
                            next.0
                                .relative_offset(next_payload.next)
                                .map(|p| ValueList(p)),
                        );
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

/// The raw serialized format of a [`Pattern`].
#[derive(AnyBitPattern, Copy, Clone, Debug)]
#[repr(C)]
pub struct PatternData {
    /// The number of elements in this pattern.
    pub num: c_int,
    // The capacity of the elements array. For serialized data, it's probably
    // the same as `num`.
    _size: c_int,
    /// The offset of the element array.
    pub elts_offset: Offset<PatternEltData>,
    ref_count: c_int,
}

/// A list of properties, each one associated with a range of values.
#[derive(Clone, Debug)]
pub struct Pattern<'buf>(pub Ptr<'buf, PatternData>);

impl Pattern<'_> {
    /// Returns an iterator over the elements in this pattern.
    pub fn elts(&self) -> Result<impl Iterator<Item = PatternElt> + '_> {
        let payload = self.0.deref()?;
        let elts = self.0.relative_offset(payload.elts_offset)?;
        Ok(elts.array(payload.num)?.map(|ptr| PatternElt(ptr)))
    }

    /// The serialized pattern data, straight from the fontconfig cache.
    pub fn data(&self) -> Result<PatternData> {
        self.0.deref()
    }
}

/// A single element of a [`Pattern`], in the raw serialized format.
#[derive(AnyBitPattern, Copy, Clone, Debug)]
#[repr(C)]
pub struct PatternEltData {
    /// The object type tag.
    pub object: c_int,
    /// Offset of the linked list of values.
    pub values: PtrOffset<ValueListData>,
}

/// A single element of a [`Pattern`].
///
/// This consists of an [`Object`] type, and a range of values. By convention,
/// the values are all of the same [`Value`] variant (of a type determined by the object
/// tag), but this is not actually enforced.
pub struct PatternElt<'buf>(pub Ptr<'buf, PatternEltData>);

impl<'buf> PatternElt<'buf> {
    /// An iterator over the values in this `PatternElt`.
    pub fn values(&self) -> Result<impl Iterator<Item = Result<Value<'buf>>> + 'buf> {
        Ok(ValueListIter {
            next: Some(Ok(ValueList(
                self.0.relative_offset(self.0.deref()?.values)?,
            ))),
        })
    }

    /// The object tag, describing the font property that this `PatternElt` represents.
    pub fn object(&self) -> Result<Object> {
        self.0.deref()?.object.try_into()
    }

    /// The serialized pattern elt data, straight from the fontconfig cache.
    pub fn data(&self) -> Result<PatternEltData> {
        self.0.deref()
    }
}

/// A set of fonts, in the raw serialized format.
///
/// This struct is just the plain old data stored in the cache. To access
/// fonts in this set, look at [`FontSet`], which represents the font set
/// in the context of the cache file.
#[derive(AnyBitPattern, Copy, Clone, Debug)]
#[repr(C)]
pub struct FontSetData {
    /// The number of fonts in this set.
    pub nfont: c_int,
    // Capacity of the font array. Uninteresting for the serialized format.
    _sfont: c_int,
    /// Pointer to an array of fonts.
    ///
    /// All the offsets here (both outer and inner) are relative to this `FontSetData`.
    pub fonts: PtrOffset<PtrOffset<PatternData>>,
}

/// A set of fonts.
#[derive(Clone, Debug)]
pub struct FontSet<'buf>(pub Ptr<'buf, FontSetData>);

impl<'buf> FontSet<'buf> {
    /// Returns an iterator over the fonts in this set.
    pub fn fonts<'a>(&'a self) -> Result<impl Iterator<Item = Result<Pattern<'buf>>> + 'a> {
        let payload = self.0.deref()?;
        let fonts = self
            .0
            .relative_offset(payload.fonts)?
            .array(payload.nfont)?;
        let me = self.clone();
        Ok(fonts.map(move |font_offset| Ok(Pattern(me.0.relative_offset(font_offset.deref()?)?))))
    }

    /// The serialized font set data, straight from the fontconfig cache.
    pub fn data(&self) -> Result<FontSetData> {
        self.0.deref()
    }
}

/// A set of code points, in the raw serialized format.
///
/// This struct is just the plain old data stored in the cache. To access
/// this set, look at [`CharSet`], which represents the char set
/// in the context of the cache file.
///
/// # Implementation details
///
/// This charset is implemented as a bunch of bitsets. Each bitset (a [`CharSetLeaf`](crate::CharSetLeaf))
/// has 256 bits, and so it represents the least significant byte of the codepoint. Associated to each
/// leaf is a 16-bit offset, representing the next two least-significant bytes of the codepoint.
/// (In particular, this can represent any subset of the numbers `0` through `0x00FFFFFF`, which is
/// big enough for the unicode range.)
#[derive(AnyBitPattern, Copy, Clone, Debug)]
#[repr(C)]
pub struct CharSetData {
    // Reference count. Not interesting for us.
    ref_count: c_int,
    /// Length of both of the following arrays
    pub num: c_int,
    /// Array of offsets to leaves. Each offset is relative to the start of the array, not the
    /// start of this struct!
    pub leaves: Offset<Offset<CharSetLeaf>>,
    /// Array having the same length as `leaves`, and storing the 16-bit offsets of each leaf.
    pub numbers: Offset<u16>,
}

/// A set of code points.
#[derive(Clone, Debug)]
pub struct CharSet<'buf>(pub Ptr<'buf, CharSetData>);

impl<'buf> CharSet<'buf> {
    /// Returns an iterator over the leaf bitsets.
    pub fn leaves(&self) -> Result<impl Iterator<Item = Result<CharSetLeaf>> + 'buf> {
        let payload = self.0.deref()?;
        let leaf_array = self.0.relative_offset(payload.leaves)?;
        Ok(leaf_array.array(payload.num)?.map(move |leaf_offset| {
            leaf_array
                .relative_offset(leaf_offset.deref()?)
                .and_then(|leaf_ptr| leaf_ptr.deref())
        }))
    }

    /// Returns an iterator over the 16-bit leaf offsets.
    pub fn numbers(&self) -> Result<Array<'buf, u16>> {
        let payload = self.0.deref()?;
        self.0.relative_offset(payload.numbers)?.array(payload.num)
    }

    /// Creates an iterator over the codepoints in this charset.
    pub fn chars(&self) -> Result<impl Iterator<Item = Result<u32>> + 'buf> {
        // TODO: this iterator-mangling is super-grungy and shouldn't allocate.
        // This would be super easy to write using generators; the main issue is that
        // the early-return-on-decode errors make the control flow tricky to express
        // with combinators and closures.
        fn transpose_result_iter<T: 'static, I: Iterator<Item = T> + 'static>(
            res: Result<I>,
        ) -> impl Iterator<Item = Result<T>> {
            match res {
                Ok(iter) => Box::new(iter.map(|x| Ok(x))) as Box<dyn Iterator<Item = Result<T>>>,
                Err(e) => Box::new(Some(Err(e)).into_iter()) as Box<dyn Iterator<Item = Result<T>>>,
            }
        }

        let leaves = self.leaves()?;
        let numbers = self.numbers()?;
        Ok(leaves.zip(numbers).flat_map(|(leaf, number)| {
            let iter = (move || {
                let number = (number.deref()? as u32) << 8;
                Ok(leaf?.iter().map(move |x| x as u32 + number))
            })();
            transpose_result_iter(iter)
        }))
    }

    /// The `CharSetLeaf` at the given index, if there is one.
    pub fn leaf_at(&self, idx: usize) -> Result<Option<CharSetLeaf>> {
        let payload = self.0.deref()?;
        let leaf_array = self.0.relative_offset(payload.leaves)?;
        leaf_array
            .array(payload.num)?
            .get(idx)
            .map(|ptr| {
                leaf_array
                    .relative_offset(ptr.deref()?)
                    .and_then(|leaf_ptr| leaf_ptr.deref())
            })
            .transpose()
    }

    /// Checks whether this charset contains a given codepoint.
    pub fn contains(&self, ch: u32) -> Result<bool> {
        let hi = ((ch >> 8) & 0xffff) as u16;
        let lo = (ch & 0xff) as u8;
        match self.numbers()?.as_slice()?.binary_search(&hi) {
            // The unwrap will succeed because numbers and leaves have the same length.
            Ok(idx) => Ok(self.leaf_at(idx)?.unwrap().contains_byte(lo)),
            Err(_) => Ok(false),
        }
    }
}

/// A set of bytes, represented as a bitset.
#[derive(AnyBitPattern, Copy, Clone, Debug)]
#[repr(C)]
pub struct CharSetLeaf {
    /// The bits in the set, all 256 of them.
    pub map: [u32; 8],
}

impl CharSetLeaf {
    /// Checks whether this set contains the given byte.
    pub fn contains_byte(&self, byte: u8) -> bool {
        let map_idx = (byte >> 5) as usize;
        let bit_idx = (byte & 0x1f) as u32;

        (self.map[map_idx] >> bit_idx) & 1 != 0
    }

    /// Creates an iterator over bits in this set.
    pub fn iter(self) -> CharSetLeafIter {
        CharSetLeafIter {
            leaf: self,
            map_idx: 0,
        }
    }
}

impl IntoIterator for CharSetLeaf {
    type Item = u8;
    type IntoIter = CharSetLeafIter;
    fn into_iter(self) -> CharSetLeafIter {
        self.iter()
    }
}

/// An iterator over bits in a [`CharSetLeaf`](crate::CharSetLeaf),
/// created by [`CharSetLeaf::iter`](crate::CharSetLeaf::iter).
#[derive(Clone, Debug)]
pub struct CharSetLeafIter {
    leaf: CharSetLeaf,
    map_idx: u8,
}

impl Iterator for CharSetLeafIter {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        let len = self.leaf.map.len() as u8;
        if self.map_idx >= len {
            None
        } else {
            let bits = &mut self.leaf.map[self.map_idx as usize];
            if *bits != 0 {
                let ret = bits.trailing_zeros() as u8;
                *bits &= !(1 << ret);
                Some(ret + (self.map_idx << 5))
            } else {
                while self.map_idx < len && self.leaf.map[self.map_idx as usize] == 0 {
                    self.map_idx += 1;
                }
                self.next()
            }
        }
    }
}

/// The fontconfig cache header, in the raw serialized format.
///
/// This is just the plain old data from the fontconfig cache. If you want to
/// actually access any other parts of the cache file, you'll need to look at
/// [`Cache`], which represents the header in the context of the rest of the
/// cache.
#[derive(AnyBitPattern, Copy, Clone, Debug)]
#[repr(C)]
pub struct CacheData {
    /// The magic 4 bytes marking the data as a fontconfig cache.
    pub magic: c_uint,
    /// The cache format version. We support versions 7 and 8.
    pub version: c_int,
    /// The size of the cache.
    pub size: isize,
    /// This cache caches the data of all fonts in some directory.
    /// Here is (an offset to) the name of that directory.
    pub dir: Offset<u8>,
    /// Here is an offset to an array of offsets to the names of
    /// subdirectories.
    pub dirs: Offset<Offset<u8>>,
    /// How many subdirectories are there?
    pub dirs_count: c_int,
    /// An offset to the set of fonts in this cache.
    pub set: Offset<FontSetData>,
    /// A "checksum" of this cache (but really just a timestamp).
    pub checksum: c_int,
    /// Another "checksum" of this cache (but really just a more precise timestamp).
    pub checksum_nano: c_int,
}

/// A reference to a fontconfig struct that's been serialized in a buffer.
#[derive(Clone)]
pub struct Ptr<'buf, S> {
    /// We point at this `offset`, relative to the buffer.
    pub offset: isize,
    /// The buffer that we point into.
    pub buf: &'buf [u8],
    marker: std::marker::PhantomData<S>,
}

impl<'buf, S> std::fmt::Debug for Ptr<'buf, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ptr").field("offset", &self.offset).finish()
    }
}

/// A reference to an array of serialized fontconfig structs.
#[derive(Clone, Debug)]
pub struct Array<'buf, T> {
    buf: &'buf [u8],
    offset: usize,
    size: isize,
    marker: std::marker::PhantomData<T>,
}

impl<'buf, T: AnyBitPattern> Array<'buf, T> {
    fn new(buf: &'buf [u8], offset: isize, size: c_int) -> Result<Self> {
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
                Ok(Array {
                    buf,
                    offset: offset as usize,
                    size: size as isize,
                    marker: std::marker::PhantomData,
                })
            }
        }
    }

    /// The number of elements in this array.
    pub fn len(&self) -> usize {
        self.size as usize
    }

    /// Is this array empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Retrieve an element at a given index, if that index isn't too big.
    pub fn get(&self, idx: usize) -> Option<Ptr<'buf, T>> {
        if (idx as isize) < self.size {
            let len = std::mem::size_of::<T>() as isize;
            Some(Ptr {
                buf: self.buf,
                offset: self.offset as isize + (idx as isize) * len,
                marker: std::marker::PhantomData,
            })
        } else {
            None
        }
    }

    /// View this array as a rust slice.
    ///
    /// This conversion might fail if the alignment is wrong. That definitely won't happen if `T` has
    /// a two-byte alignment. It's *probably* fine in general, but don't blame me if it isn't.
    pub fn as_slice(&self) -> Result<&'buf [T]> {
        let len = std::mem::size_of::<T>() * self.size as usize;
        bytemuck::try_cast_slice(&self.buf[self.offset..(self.offset + len)]).map_err(|_| {
            Error::BadAlignment {
                offset: self.offset,
                expected_alignment: std::mem::align_of::<T>(),
            }
        })
    }
}

impl<'buf, T: AnyBitPattern> Iterator for Array<'buf, T> {
    type Item = Ptr<'buf, T>;

    fn next(&mut self) -> Option<Ptr<'buf, T>> {
        if self.size <= 0 {
            None
        } else {
            let len = std::mem::size_of::<T>();
            let ret = Ptr {
                buf: self.buf,
                offset: self.offset as isize,
                marker: std::marker::PhantomData,
            };
            self.offset += len;
            self.size -= 1;
            Some(ret)
        }
    }
}

impl<'buf> Ptr<'buf, u8> {
    /// Assuming that this `Ptr<u8>` is pointing to the beginning of a null-terminated string,
    /// return that string.
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

impl<'buf, S: AnyBitPattern> Ptr<'buf, S> {
    /// Turn `offset` into a pointer, assuming that it's an offset relative to this pointer.
    ///
    /// In order to be certain about which offsets are relative to what, you'll need to check
    /// the fontconfig source. But generally, offsets stored in a struct are relative to the
    /// base address of that struct. So for example, to access the `dir` field in
    /// [`Cache`](crate::Cache) you could call `cache.relative_offset(cache.deref()?.dir)?`.
    /// This will give you a `Ptr<u8>` pointing to the start of the directory name.
    pub fn relative_offset<Off: IntoOffset>(&self, offset: Off) -> Result<Ptr<'buf, Off::Item>> {
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

    /// "Dereference" this pointer, returning a plain struct.
    pub fn deref(&self) -> Result<S> {
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

    /// Treating this pointer as a reference to the start of an array of length `count`,
    /// return an iterator over that array.
    fn array(&self, count: c_int) -> Result<Array<'buf, S>> {
        Array::new(self.buf, self.offset, count)
    }
}

/// All the possible errors we can encounter when parsing the cache file.
#[derive(Clone, Debug, thiserror::Error)]
#[allow(missing_docs)]
pub enum Error {
    #[error("Invalid magic number {0:#x}")]
    BadMagic(c_uint),

    #[error("Unsupported version {0}")]
    UnsupportedVersion(c_int),

    #[error("Bad pointer {0}")]
    BadPointer(isize),

    #[error("Bad offset {0}")]
    BadOffset(isize),

    #[error("Bad alignment (expected {expected_alignment}) for offset {offset}")]
    BadAlignment {
        expected_alignment: usize,
        offset: usize,
    },

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

/// The fontconfig cache header.
#[derive(Clone, Debug)]
pub struct Cache<'buf>(Ptr<'buf, CacheData>);

impl<'buf> Cache<'buf> {
    /// Read a cache from a slice of bytes.
    pub fn from_bytes(buf: &'buf [u8]) -> Result<Self> {
        use Error::*;

        let len = std::mem::size_of::<CacheData>();
        if buf.len() < len {
            Err(WrongSize {
                expected: len as isize,
                actual: buf.len() as isize,
            })
        } else {
            let cache: CacheData = bytemuck::try_pod_read_unaligned(&buf[0..len])
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
                Ok(Cache(Ptr {
                    buf,
                    offset: 0,
                    marker: std::marker::PhantomData,
                }))
            }
        }
    }

    /// The [`FontSet`](crate::FontSet) stored in this cache.
    pub fn set(&self) -> Result<FontSet<'buf>> {
        Ok(FontSet(self.0.relative_offset(self.0.deref()?.set)?))
    }

    /// The serialized cache data, straight from the fontconfig cache.
    pub fn data(&self) -> Result<CacheData> {
        self.0.deref()
    }
}
