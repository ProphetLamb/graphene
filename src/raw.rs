use core::{
    alloc::{Allocator, Layout},
    mem,
    ops::{Index, IndexMut, Range},
    ptr::{self, NonNull},
    slice,
};

use alloc::{alloc::Global, borrow::ToOwned, raw_vec::RawVec, vec::Vec};

use self::Error::*;

/// https://arxiv.org/abs/1902.01961
#[macro_export]
macro_rules! fmm {
    ($d:expr) => {
        (u64::MAX / ($d as u32) as u64 + 1)
    };
}

/// Round up integer division
#[macro_export]
macro_rules! ru_int_div {
    ($x:expr, $d:expr) => {
        (if ($x) == 0 { 0 } else { (($x) - 1) / ($d) } + 1)
    };
}

/// Next greater multiple of
#[macro_export]
macro_rules! ng_mul_of {
    ($x:expr, $m:expr) => {
        (ru_int_div!(($x), ($m)) * ($m))
    };
}

/// Integer Log2
/// # Safely
/// The evaluated value may not exceed `isize::MAX`.
#[macro_export]
macro_rules! log2 {
    ($x:expr) => {{
        (num_bits::<isize>() - ($x as isize).leading_zeros() as usize - 1)
    }};
}

/// Same interface as `SliceIndex` but for collection.
pub unsafe trait UnsafeIndexer<Idx: Copy + ?Sized> {
    type Output: ?Sized;

    fn get(&self, index: Idx) -> Option<&Self::Output> {
        match self.guard_index(index) {
            Ok(_) => Some(unsafe { &*self.ptr_at(index) }),
            Err(_) => None,
        }
    }

    fn get_mut(&mut self, index: Idx) -> Option<&mut Self::Output> {
        match self.guard_index(index) {
            Ok(_) => Some(unsafe { &mut *self.ptr_at_mut(index) }),
            Err(_) => None,
        }
    }

    unsafe fn get_unchecked(&self, index: Idx) -> &Self::Output {
        &*self.ptr_at(index)
    }

    unsafe fn get_unchecked_mut(&mut self, index: Idx) -> &mut Self::Output {
        &mut *self.ptr_at_mut(index)
    }

    fn guard_index(&self, index: Idx) -> Result<(), ()>;

    unsafe fn ptr_at(&self, index: Idx) -> *const Self::Output;

    unsafe fn ptr_at_mut(&mut self, index: Idx) -> *mut Self::Output;
}

pub trait SetOperations {
    fn unify(&self, other: &Self) -> Self;

    fn encompasses(&self, other: &Self) -> bool;
}

impl<T: Copy + Eq + Ord> SetOperations for Range<T> {
    fn unify(&self, other: &Self) -> Self {
        let lo = self.start.min(other.start);
        let hi = self.end.max(other.end);
        Self { start: lo, end: hi }
    }

    fn encompasses(&self, other: &Self) -> bool {
        let unified = self.unify(other);
        unified.start == self.start && unified.end == self.end
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct HiLoSplit {
    raw: usize,
}

impl HiLoSplit {
    const HI_BSH: usize = log2!(mem::size_of::<usize>());
    const LO_MSK: usize = usize::MAX << (mem::size_of::<usize>() / 2);
    const MIN: usize = 0;
    const MAX: usize = Self::LO_MSK;

    fn from(lo: usize, hi: usize) -> HiLoSplit {
        HiLoSplit {
            raw: (hi << Self::HI_BSH) | (lo & Self::LO_MSK),
        }
    }

    #[inline(always)]
    fn hi(&self) -> usize {
        self.raw >> Self::HI_BSH
    }

    #[inline]
    fn hi_set(&mut self, v: usize) {
        debug_assert!(v <= Self::LO_MSK);
        self.raw = (v << Self::HI_BSH) | (self.raw & Self::LO_MSK)
    }

    #[inline]
    fn hi_set_off(&mut self, offset: isize) {
        debug_assert!(offset >= -(Self::LO_MSK as isize) && offset <= Self::LO_MSK as isize);
        self.raw =
            ((self.hi() as isize + offset) << Self::HI_BSH) as usize | (self.raw & Self::LO_MSK)
    }

    #[inline(always)]
    fn lo(&self) -> usize {
        self.raw & Self::LO_MSK
    }

    #[inline]
    fn lo_set(&mut self, v: usize) {
        debug_assert!(v <= Self::LO_MSK);
        self.raw = (self.raw & !Self::LO_MSK) | (v & Self::LO_MSK)
    }

    #[inline]
    fn lo_set_off(&mut self, offset: isize) {
        debug_assert!(offset >= -(Self::LO_MSK as isize) && offset <= Self::LO_MSK as isize);
        self.raw =
            (self.raw & !Self::LO_MSK) | ((self.lo() as isize + offset) as usize & Self::LO_MSK)
    }

    fn delta(&self) -> usize {
        self.hi() - self.lo()
    }

    fn sum(&self) -> usize {
        self.hi() + self.lo()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Insertion {
    insert_idx: usize,
    insert_cnt: usize,
    behind_off: usize,
}

impl Insertion {
    pub fn new(insert_idx: usize, insert_cnt: usize, behind_off: usize) -> Self {
        Self {
            insert_idx,
            insert_cnt,
            behind_off,
        }
    }
}

pub enum Error {
    KeyNotFound(usize),
    IteratorLengthOverflow(usize),
    RangeOutOfBounds(Range<usize>),
    NoVacantEntries(usize, usize),
}

pub struct SparseOrderedMappedRows<K: Copy + Ord, V, A: Clone + Allocator> {
    keys: RawVec<K, A>,
    keys_len: usize,
    values: RawVec<V, A>,
    values_len: usize,
    keys_values: RawVec<HiLoSplit, A>,
    occupied_values: BitMarker<A>,
}

unsafe impl<K: Copy + Ord, V, A: Clone + Allocator> UnsafeIndexer<usize>
    for SparseOrderedMappedRows<K, V, A>
{
    type Output = HiLoSplit;

    #[inline]
    fn guard_index(&self, index: usize) -> Result<(), ()> {
        if index < self.len() {
            Ok(())
        } else {
            Err(())
        }
    }

    #[inline]
    unsafe fn ptr_at(&self, index: usize) -> *const Self::Output {
        self.keys_values.ptr().add(index)
    }

    #[inline]
    unsafe fn ptr_at_mut(&mut self, index: usize) -> *mut Self::Output {
        self.keys_values.ptr().add(index)
    }
}

impl<K: Copy + Ord, V, A: Clone + Allocator> Drop for SparseOrderedMappedRows<K, V, A> {
    fn drop(&mut self) {
        if mem::size_of::<V>() != 0 && !self.values.capacity() == 0 {
            unsafe { ptr::drop_in_place(self.values.ptr()) };
            unsafe { ptr::drop_in_place(self.keys_values.ptr()) };
        }
        self.values.shrink_to_fit(0);
        self.keys_values.shrink_to_fit(0);
    }
}

impl<K: Copy + Ord, V, A: Clone + Allocator> SparseOrderedMappedRows<K, V, A> {
    const MIN_NZ_VALUE_CAP: usize = log2!(mem::size_of::<V>()) * 2;

    pub fn new_in(alloc: A) -> Self {
        Self {
            keys: RawVec::new_in(alloc.to_owned()),
            keys_values: RawVec::new_in(alloc.to_owned()),
            values: RawVec::new_in(alloc.to_owned()),
            occupied_values: BitMarker::new_in(alloc),
            values_len: 0,
            keys_len: 0,
        }
    }

    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        let value_capacity = capacity * (log2!(capacity as i32) as usize);
        Self::with_value_capacity_in(capacity, value_capacity, alloc)
    }

    pub fn with_value_capacity_in(capacity: usize, value_capacity: usize, alloc: A) -> Self {
        Self {
            keys: RawVec::with_capacity_in(capacity, alloc.to_owned()),
            keys_values: RawVec::with_capacity_in(capacity, alloc.to_owned()),
            values: RawVec::with_capacity_in(value_capacity, alloc.to_owned()),
            occupied_values: BitMarker::with_capacity_in(value_capacity, alloc),
            values_len: 0,
            keys_len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.keys_len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of values the map with the current number of keys can hold without moving or reallocating if all values are filled with no holes.
    ///
    /// This is proportional to but possibly greater then `self.len()*Self::MIN_NZ_VALUE_CAP`.
    pub fn head(&self) -> usize {
        if self.keys_len == 0 {
            Self::MIN_NZ_VALUE_CAP
        } else {
            let h = unsafe { self.get_unchecked(self.keys_len - 1) };
            ng_mul_of!(h.hi(), Self::MIN_NZ_VALUE_CAP)
        }
    }

    pub fn value_len(&self) -> usize {
        self.values_len
    }

    pub fn insert(&mut self, key: K) -> Result<usize, ()> {
        todo!()
    }

    pub fn remove(&mut self, key: K) -> Result<usize, ()> {
        todo!()
    }

    pub fn insert_value(&mut self, key_idx: usize, value: V) -> Result<Insertion, Error> {
        let off = self.guard_insert(key_idx, 1).ok_or(KeyNotFound(key_idx))?;
        unsafe {
            let key = *self.get_unchecked(key_idx);
            let key_max = key.lo() + unsafe { self.values_at_key_cap(key_idx, key) };
            let insertion = self.insert_value_unchecked(key_idx, key.lo(), key_max, value)?;
            self.get_unchecked_mut(key_idx).hi_set_off(1);
            self.values_len += 1;
            Ok(Insertion::new(insertion, 1, off))
        }
    }

    pub fn insert_values<I: ExactSizeIterator + IntoIterator<Item = V>>(
        &mut self,
        key_idx: usize,
        values: I,
    ) -> Result<Insertion, Error> {
        let len = values.len();
        let off = self
            .guard_insert(key_idx, len)
            .ok_or(KeyNotFound(key_idx))?;
        let key = unsafe { *self.get_unchecked(key_idx) };
        let key_max = key.lo() + unsafe { self.values_at_key_cap(key_idx, key) };
        let mut cnt = 0;
        let mut start_idx = key.lo();
        for value in values {
            if cnt >= len {
                return Err(IteratorLengthOverflow(cnt));
            }
            let insert_idx =
                unsafe { self.insert_value_unchecked(key_idx, start_idx, key_max, value)? };
            start_idx = usize::min(start_idx, insert_idx);
            cnt += 1;
        }
        unsafe { self.get_unchecked_mut(key_idx).hi_set_off(cnt as isize) };
        self.values_len += cnt;
        Ok(Insertion::new(start_idx, len, off))
    }

    unsafe fn insert_value_unchecked(
        &mut self,
        key_idx: usize,
        start: usize,
        end: usize,
        value: V,
    ) -> Result<usize, Error> {
        let mut i = start;
        while i < end {
            if !self.occupied_values.is_set_at(i) {
                *self.get_value_unchecked_mut(i) = value;
                return Ok(i);
            }
            i += 1;
        }
        Err(NoVacantEntries(start, end))
    }

    pub fn remove_value_range(&mut self, key_idx: usize, range: Range<usize>) -> Result<(), Error> {
        if range.start >= range.end {
            Err(KeyNotFound(key_idx))
        } else if let Some(key) = self.get(key_idx) {
            let key = *key;
            let key_cap = unsafe { self.values_at_key_cap(key_idx, key) };
            if !(0..key_cap).encompasses(&range) {
                return Err(RangeOutOfBounds(range));
            }
            let start = key.lo() + range.start;
            let end = key.lo() + range.end;
            unsafe {
                // Drop removed values.
                let ptr = self.values.ptr().add(key.lo());
                let slice = ptr::slice_from_raw_parts_mut(ptr, range.start);
                ptr::drop_in_place(slice);
                let ptr = self.values.ptr().add(end);
                let slice = ptr::slice_from_raw_parts_mut(ptr, key.hi() - end);
                ptr::drop_in_place(ptr);
                // Mutate index to exclude dropped values.
                *self.ptr_at_mut(key_idx) = HiLoSplit::from(start, end);
            }
            // Mark values as dropped.
            self.occupied_values.clear_wind(start, end);
            Ok(())
        } else {
            Err(KeyNotFound(key_idx))
        }
    }

    unsafe fn values_at_key_cap(&self, key_idx: usize, key: HiLoSplit) -> usize {
        if key_idx + 1 < self.len() {
            self.get_unchecked(key_idx + 1).lo()
        } else {
            self.values.capacity() - key.lo()
        }
    }

    fn guard_insert(&mut self, key_idx: usize, additional_values_at_key: usize) -> Option<usize> {
        if mem::size_of::<V>() == 0 {
            return None;
        }
        if let Some(key) = self.get(key_idx) {
            // The capacity of the values grows (`req_cap`) more aggressively then the values for the key (`new_key_cap`),
            // to run more often into the 2nd branch, where we move values instead of reallocating which is done in the 3rd branch.
            // This trade of has been made because graphs are often unbalanced
            // and just allocating the full capacity evenly throughout the keys would result is unreasonable growth.
            // A smarter algorithm, distributing memory based on the relative length of the values for each specific key
            // would ultimately resolve this issue, but increase the time for the resize and complexity of the function immensely,
            // And there is no guarantee that the prediction model would hold. Therefore we just leave a bit of overhead when resizing
            // and use up that overhead when needed.
            // One nasty side effect of the current model is that the last inserted vertex is naturally the cheapest to move,
            // as it requires zero effort. But the first vertex inserted into the graph is usually the most 'used'
            // and incidentally is inclined to own the most edges, but is also the most expensive to move.
            // TODO: When moving or reallocating the edges for a key, move them towards the end of the memory,
            // TODO: and shift to the left, to counteract the issue mentioned above. 
            // TODO: We must remodel the `key_values` to allow for randomized order of edges, by carrying the capacity.
            // TODO: This also speeds up copying because moving overlapping memory towards the right is much cheaper.
            let key = *key;
            let key_len = key.delta();
            let key_cap = unsafe { self.values_at_key_cap(key_idx, key) };
            let new_key_cap =
                key_len + ng_mul_of!(additional_values_at_key, Self::MIN_NZ_VALUE_CAP);
            let new_key_cap = usize::max(new_key_cap, key_cap * 3 / 2);
            let key_cap_inc = new_key_cap - key_cap;
            let req_cap = self.head() + key_cap_inc;
            if key_len + additional_values_at_key <= key_cap {
                // No need to resize
                Some(0)
            } else if req_cap <= self.values.capacity() {
                // Able to move values towards the end of the allocates memory.
                if key_idx + 1 == self.len() {
                    Some(0)
                } else {
                    unsafe { 
                        self.move_values_behind(key.lo() + key_cap, key_cap_inc);
                        self.offset_mappings_behind(key_idx, key_cap_inc as isize);
                    }
                    Some(key_cap_inc)
                }
            } else {
                // Must reallocate memory
                let req_cap = usize::max(req_cap, 2 * self.values.capacity());
                unsafe {
                    // If we need to offset the value mapping after the insertion, or not.
                    if key_idx + 1 == self.len() {
                        self.grow_values(req_cap)
                    } else {
                        self.grow_values_between(req_cap, key, new_key_cap);
                        self.offset_mappings_behind(key_idx, key_cap_inc as isize);
                    }
                }
                Some(key_cap_inc)
            }
        } else {
            None
        }
    }

    unsafe fn offset_mappings_behind(&mut self, mut key_idx: usize, offset: isize) {
        let len = self.len();
        while key_idx < len {
            unsafe { self.offset_mapping_unchecked(key_idx, offset) };
            key_idx += 1;
        }
    }

    unsafe fn offset_mapping_unchecked(&mut self, key_idx: usize, offset: isize) {
        let key = self.get_unchecked_mut(key_idx);
        key.lo_set_off(offset);
        key.hi_set_off(offset);
    }

    unsafe fn grow_values_between(&mut self, req_cap: usize, key: HiLoSplit, offset: usize) {
        let head = self.head();
        let req_cap = head + req_cap;
        split_move_to_new(&mut self.keys_values, req_cap, key.hi(), offset, head);
        split_move_to_new(&mut self.values, req_cap, key.hi(), offset, head);
        self.occupied_values.reserve(req_cap)
    }

    unsafe fn grow_values(&mut self, req_cap: usize) {
        let len = self.head();
        let additional = req_cap - len;
        self.keys_values.reserve(len, additional);
        self.values.reserve(len, additional);
        self.occupied_values.reserve(req_cap)
    }

    unsafe fn move_values_behind(&mut self, next_key_start: usize, offset: usize) {
        ptr::copy(
            self.values.ptr().add(next_key_start),
            self.values.ptr().add(next_key_start + offset),
            self.head() - next_key_start,
        );
    }

    pub fn iter(&self) -> SegIter<K, V, A> {
        SegIter {
            vec: self,
            key_idx: 0,
            value_idx: 0,
            size: self.value_len(),
            taken: 0,
        }
    }

    fn get_value(&self, key_idx: usize, value_idx: usize) -> Option<&V> {
        let key = *self.get(key_idx)?;
        if value_idx >= key.lo() && value_idx < key.hi() {
            Some(unsafe { &*self.get_value_unchecked(value_idx) })
        } else {
            None
        }
    }

    unsafe fn get_value_unchecked(&self, index: usize) -> *const V {
        self.values.ptr().add(index)
    }

    unsafe fn get_value_unchecked_mut(&mut self, index: usize) -> *mut V {
        self.values.ptr().add(index)
    }
}

pub struct SegIter<'a, K: Copy + Ord, V: 'a, A: Clone + Allocator> {
    vec: &'a SparseOrderedMappedRows<K, V, A>,
    key_idx: usize,
    value_idx: usize,
    size: usize,
    taken: usize,
}

impl<'a, K: Copy + Ord, V: 'a, A: Clone + Allocator> ExactSizeIterator for SegIter<'a, K, V, A> {}

impl<'a, K: Copy + Ord, V: 'a, A: Clone + Allocator> Iterator for SegIter<'a, K, V, A> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        if self.taken == self.size {
            return None;
        }
        if let Some(key) = self.vec.get(self.key_idx) {
            if self.value_idx < key.lo() {
                self.value_idx = key.lo();
            } else {
                self.value_idx += 1;
            }
            if self.value_idx >= key.hi() {
                self.key_idx += 1;
                return self.next();
            }
        }
        self.taken += 1;
        self.vec.get_value(self.key_idx, self.value_idx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.size - self.taken;
        (len, Some(len))
    }
}

impl<'a, K: Copy + Ord, V: 'a, A: Clone + Allocator> DoubleEndedIterator for SegIter<'a, K, V, A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.taken == 0 {
            // Disallow traversing behind the starting position.
            return None;
        }
        if let Some(key) = self.vec.get(self.key_idx) {
            if self.value_idx > key.hi() {
                self.value_idx = key.hi();
            } else {
                self.value_idx -= 1;
            }
            if self.value_idx < key.lo() {
                self.key_idx = unsafe { self.key_idx.unchecked_sub(1) };
                return self.next();
            }
        }
        self.taken -= 1;
        self.vec.get_value(self.key_idx, self.value_idx)
    }
}

struct Edge {
    vertex_id_flags: u32,
    segment_id: u32,
}

impl Edge {
    /// The amount by which to shift the flags in the `vertex_id_flags`.
    pub const FLAGS_BSH: u32 = ((mem::size_of::<u32>() - 1) * mem::size_of::<u8>()) as u32;
    /// The flag marking the edge as existed, if it is missing the edge is deleted. This is to make use of calloc.
    const FLAG_DEL: u32 = 0x01 << Self::FLAGS_BSH;
    /// The maximum valid id value of any `Vertex`.
    pub const VERTEX_ID_MAX: u32 = u32::MAX >> mem::size_of::<u8>();

    pub fn deleted() -> Edge {
        Edge {
            vertex_id_flags: Self::FLAG_DEL,
            segment_id: !0,
        }
    }

    #[inline]
    pub fn vertex_id(&self) -> u32 {
        self.vertex_id_flags & Self::VERTEX_ID_MAX
    }

    pub fn vertex_id_set(&mut self, vertex_id: u32) {
        debug_assert!(vertex_id & Self::VERTEX_ID_MAX == vertex_id);
        self.vertex_id_flags =
            (self.vertex_id_flags & !Self::VERTEX_ID_MAX) | (vertex_id & Self::VERTEX_ID_MAX);
    }

    #[inline]
    pub fn flags(&self) -> u32 {
        self.vertex_id_flags >> Self::FLAGS_BSH
    }

    #[inline]
    pub fn flags_set(&mut self, flags: u32) {
        debug_assert!(flags & (u8::MAX as u32) == flags);
        self.vertex_id_flags =
            (self.vertex_id_flags & Self::VERTEX_ID_MAX) | (flags << Self::FLAGS_BSH);
    }

    #[inline]
    pub fn is_del(&self) -> bool {
        self.vertex_id_flags & Self::FLAG_DEL == 0
    }

    pub fn del(&mut self) {
        if !self.is_del() {
            self.segment_id = !0;
            self.vertex_id_flags = 0;
        }
    }
}

struct Vertex<E, A: Allocator + Copy> {
    edges: RawVec<Edge, A>,
    edges_values: RawVec<E, A>,
    edges_init: BitMarker<A>,
    /// The total number of elements.
    len: usize,
    /// No free items exist before.
    first_free_hint: usize,
    /// No occupied items exists after.
    last_occupied_hint: usize,
}

impl<E, A: Allocator + Copy> Drop for Vertex<E, A> {
    fn drop(&mut self) {
        self.len = 0;
        self.first_free_hint = 0;
        self.last_occupied_hint = !0;
    }
}

impl<E, A: Allocator + Copy> Vertex<E, A> {
    pub fn new_in(alloc: A) -> Self {
        Self {
            edges: RawVec::new_in(alloc),
            edges_values: RawVec::new_in(alloc),
            edges_init: BitMarker::new_in(alloc),
            len: 0,
            first_free_hint: 0,
            last_occupied_hint: !0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn cap(&self) -> usize {
        self.edges.capacity()
    }

    pub fn insert(&mut self, edge: Edge, value: E) -> Result<usize, ()> {
        if self.len() == self.cap() {
            return Err(());
        }
        let next_free = self.guard_insert()?;
        // The element at which we insert must be deleted.
        debug_assert!(unsafe { &*self.ptr_at(next_free) }.is_del());
        unsafe { self.set_unchecked(next_free, edge, value) };
        self.len += 1;
        Ok(next_free)
    }

    pub fn remove_at(&mut self, index: usize) -> Result<(), ()> {
        self.guard_remove_at(index)?;
        let vertex = self.get_mut(index).ok_or(())?;
        vertex.del();
        Ok(())
    }

    fn guard_insert(&mut self) -> Result<usize, ()> {
        if self.len == self.cap() {
            return Err(());
        }
        // Last occupied is !0 when empty. We need to intentionally overflow in that case to reach zero.
        let after_last_occupied = unsafe { self.last_occupied_hint.unchecked_add(1) };
        if after_last_occupied == self.len {
            // There are no holes.
            self.last_occupied_hint = after_last_occupied;
            self.first_free_hint = after_last_occupied + 1;
            return Ok(after_last_occupied);
        }
        let mut first_free = self.first_free_hint;
        while first_free < after_last_occupied {
            if !self.edges_init.is_set_at(first_free) {
                self.first_free_hint = first_free + 1;
                return Ok(first_free);
            }
            first_free += 1;
        }

        // after_last_occupied must be within the capacity, because we still have free spots.
        debug_assert!(after_last_occupied < self.cap());
        self.last_occupied_hint = after_last_occupied + 1;
        self.first_free_hint = after_last_occupied;
        Ok(after_last_occupied)
    }

    fn guard_remove_at(&mut self, index: usize) -> Result<(), ()> {
        if self.len == self.cap() {
            return Err(());
        }
        let last_occupied_hint = self.last_occupied_hint;
        if last_occupied_hint > index {
            let mut last_occupied = index + 1;
            while last_occupied < last_occupied_hint {
                if self.edges_init.is_set_at(last_occupied) {
                    self.last_occupied_hint = last_occupied;
                    return Ok(());
                }
                last_occupied += 1
            }
            // No occupied items beyond index.
            self.last_occupied_hint = index - 1;
        }
        Ok(())
    }

    #[inline]
    fn get_value_unchecked(&self, index: usize) -> &E {
        unsafe { &*self.edges_values.ptr().add(index) }
    }

    #[inline]
    fn get_value_unchecked_mut(&mut self, index: usize) -> &mut E {
        unsafe { &mut *self.edges_values.ptr().add(index) }
    }

    #[inline(never)]
    unsafe fn set_unchecked(&mut self, index: usize, edge: Edge, value: E) {
        *self.ptr_at_mut(index) = edge;
        if mem::size_of::<E>() != 0 {
            *self.get_value_unchecked_mut(index) = value;
        }
    }
}

impl<E, A: Allocator + Copy> IndexMut<usize> for Vertex<E, A> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self.get_mut(index) {
            Some(edge) => edge,
            None => index_out_of_range(),
        }
    }
}

impl<E, A: Allocator + Copy> Index<usize> for Vertex<E, A> {
    type Output = Edge;

    fn index(&self, index: usize) -> &Self::Output {
        match self.get(index) {
            Some(edge) => edge,
            None => index_out_of_range(),
        }
    }
}

unsafe impl<E, A: Allocator + Copy> UnsafeIndexer<usize> for Vertex<E, A> {
    type Output = Edge;

    #[inline]
    fn guard_index(&self, index: usize) -> Result<(), ()> {
        if index < isize::MAX as usize {
            Ok(())
        } else {
            Err(())
        }
    }

    #[inline]
    unsafe fn ptr_at(&self, index: usize) -> *const Self::Output {
        self.edges.ptr().add(index)
    }

    #[inline]
    unsafe fn ptr_at_mut(&mut self, index: usize) -> *mut Self::Output {
        self.edges.ptr().add(index)
    }
}

struct Pack<V, E, A: Allocator + Copy> {
    vertices: RawVec<Vertex<E, A>, A>,
    vertices_values: RawVec<V, A>,
    vertices_init: BitMarker<A>,
    /// The total number of elements.
    len: usize,
    /// No free items exist before.
    first_free_hint: usize,
    /// No occupied items exists after.
    last_occupied_hint: usize,
}

impl<V, E, A: Allocator + Copy> Pack<V, E, A> {
    /// The maximum number of Vertices for a `Pack` to fit in one 4096byte memory page.
    pub const PAGE_PACK_VERTEX_NUM: usize =
        (4096 - mem::size_of::<Pack<V, E, Global>>()) / mem::size_of::<Vertex<E, Global>>();

    pub fn with_capacity_in(capacity: usize, alloc: A) -> Pack<V, E, A> {
        Pack {
            vertices: RawVec::with_capacity_zeroed_in(capacity, alloc),
            vertices_values: RawVec::with_capacity_in(capacity, alloc),
            vertices_init: BitMarker::with_capacity_in(capacity, alloc),
            len: 0,
            first_free_hint: 0,
            last_occupied_hint: !0,
        }
    }

    #[inline]
    pub fn cap(&self) -> usize {
        self.vertices.capacity()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn insert(&mut self, vertex: Vertex<E, A>, value: V) -> Result<usize, ()> {
        let next_free = self.guard_insert()?;
        // The element at which we insert must be deleted.
        if self.vertices_init.is_set_at(next_free) {
            return Err(());
        }
        self.vertices_init.set_at(next_free);
        unsafe { self.set_unchecked(next_free, vertex, value) };
        self.len += 1;
        Ok(next_free)
    }

    pub fn remove_at(&mut self, index: usize) -> Result<(), ()> {
        self.guard_remove_at(index)?;
        if !self.vertices_init.is_set_at(index) {
            return Err(());
        }
        self.vertices_init.clear_at(index);
        let vertex = self.get_mut(index).ok_or(())?;
        Ok(())
    }

    fn guard_insert(&mut self) -> Result<usize, ()> {
        if self.len == self.cap() {
            return Err(());
        }
        // Last occupied is !0 when empty. We need to intentionally overflow in that case to reach zero.
        let after_last_occupied = unsafe { self.last_occupied_hint.unchecked_add(1) };
        if after_last_occupied == self.len {
            // There are no holes.
            self.last_occupied_hint = after_last_occupied;
            self.first_free_hint = after_last_occupied + 1;
            return Ok(after_last_occupied);
        }
        let mut first_free = self.first_free_hint;
        while first_free < after_last_occupied {
            if !self.vertices_init.is_set_at(first_free) {
                self.first_free_hint = first_free + 1;
                return Ok(first_free);
            }
            first_free += 1;
        }

        // after_last_occupied must be within the capacity, because we still have free spots.
        debug_assert!(after_last_occupied < self.cap());
        self.last_occupied_hint = after_last_occupied + 1;
        self.first_free_hint = after_last_occupied;
        Ok(after_last_occupied)
    }

    fn guard_remove_at(&mut self, index: usize) -> Result<(), ()> {
        if self.len == self.cap() {
            return Err(());
        }
        let last_occupied_hint = self.last_occupied_hint;
        if last_occupied_hint > index {
            let mut last_occupied = index + 1;
            while last_occupied < last_occupied_hint {
                if self.vertices_init.is_set_at(last_occupied) {
                    self.last_occupied_hint = last_occupied;
                    return Ok(());
                }
                last_occupied += 1
            }
            // No occupied items beyond index.
            self.last_occupied_hint = index - 1;
        }
        Ok(())
    }

    #[inline]
    fn get_value_unchecked(&self, index: usize) -> &V {
        unsafe { &*self.vertices_values.ptr().add(index) }
    }

    #[inline]
    fn get_value_unchecked_mut(&mut self, index: usize) -> &mut V {
        unsafe { &mut *self.vertices_values.ptr().add(index) }
    }

    #[inline(never)]
    unsafe fn set_unchecked(&mut self, index: usize, vertex: Vertex<E, A>, value: V) {
        *self.ptr_at_mut(index) = vertex;
        if mem::size_of::<V>() != 0 {
            *self.get_value_unchecked_mut(index) = value;
        }
    }
}

unsafe impl<V, E, A: Allocator + Copy> UnsafeIndexer<usize> for Pack<V, E, A> {
    type Output = Vertex<E, A>;

    #[inline]
    fn guard_index(&self, index: usize) -> Result<(), ()> {
        if index < self.vertices.capacity() {
            Ok(())
        } else {
            Err(())
        }
    }

    #[inline]
    unsafe fn ptr_at(&self, index: usize) -> *const Self::Output {
        self.vertices.ptr().add(index)
    }

    #[inline]
    unsafe fn ptr_at_mut(&mut self, index: usize) -> *mut Self::Output {
        self.vertices.ptr().add(index)
    }
}

struct PackedSparseVector<V, E, A: Allocator + Copy> {
    packs: RawVec<Pack<V, E, A>>,
}

impl<V, E, A: Allocator + Copy> PackedSparseVector<V, E, A> {
    pub fn with_capacity_in(capacity: usize, alloc: A) {
        todo!()
    }
}

fn index_out_of_range() -> ! {
    panic!("The index is outside the range of valid values.")
}

struct BitMarker<A: Allocator> {
    raw: RawVec<usize, A>,
    len: usize,
}

impl<A: Allocator> BitMarker<A> {
    // Constants for fast mod multiplication.
    const FMM: u64 = fmm!(num_bits::<usize>() as u32);
    const DIV: u32 = num_bits::<usize>() as u32;
    const OFFSET_TO_ITEM_RSH: usize = log2!(mem::size_of::<usize>());

    fn bit_mod(offset: usize) -> u32 {
        fast_mod(offset as u32, Self::DIV, Self::FMM)
    }

    pub fn new_in(alloc: A) -> Self {
        Self {
            raw: RawVec::new_in(alloc),
            len: 0,
        }
    }

    pub fn with_capacity_in(bits: usize, alloc: A) -> Self {
        assert!(bits < u32::MAX as usize);
        let items = align_bits::<usize>(bits);
        Self {
            raw: RawVec::with_capacity_zeroed_in(items, alloc),
            len: items,
        }
    }

    pub fn reserve(&mut self, additional_bits: usize) {
        let aligned_bits = align_bits::<usize>(additional_bits);
        let additional = aligned_bits / mem::size_of::<usize>();
        self.raw.reserve(self.len, additional);
        self.len += aligned_bits;
    }

    pub fn set_at(&mut self, bit_offset: usize) {
        debug_assert!(bit_offset < i32::MAX as usize);
        let idx = bit_offset >> Self::OFFSET_TO_ITEM_RSH;
        let mask = 1 << Self::bit_mod(bit_offset);
        unsafe { *self.raw.ptr().add(idx) |= mask }
    }

    pub fn set_wind(&mut self, start_offset: usize, end_offset: usize) {
        debug_assert!(end_offset > start_offset);
        let first_idx = start_offset >> Self::OFFSET_TO_ITEM_RSH;
        let first_mask = usize::MAX << Self::bit_mod(start_offset);
        let last_idx = end_offset >> Self::OFFSET_TO_ITEM_RSH;
        let last_mask = usize::MAX >> Self::bit_mod(end_offset);
        unsafe {
            *self.raw.ptr().add(first_idx) |= !first_mask;
            let mut idx = first_idx + 1;
            while idx < last_idx {
                *self.raw.ptr().add(idx) = usize::MAX;
                idx += 1;
            }
            *self.raw.ptr().add(last_idx) |= !last_mask;
        }
    }

    pub fn clear_at(&mut self, bit_offset: usize) {
        debug_assert!(bit_offset < i32::MAX as usize);
        let items = bit_offset >> Self::OFFSET_TO_ITEM_RSH;
        let bytes = unsafe { &mut *self.raw.ptr().add(items) };
        let mask = 1 << Self::bit_mod(bit_offset);
        *bytes &= !mask;
    }

    pub fn clear_wind(&mut self, start_offset: usize, end_offset: usize) {
        debug_assert!(end_offset > start_offset);
        let first_idx = start_offset >> Self::OFFSET_TO_ITEM_RSH;
        let first_mask = usize::MAX << Self::bit_mod(start_offset);
        let last_idx = end_offset >> Self::OFFSET_TO_ITEM_RSH;
        let last_mask = usize::MAX >> Self::bit_mod(end_offset);
        unsafe {
            *self.raw.ptr().add(first_idx) &= first_mask;
            let mut idx = first_idx + 1;
            while idx < last_idx {
                *self.raw.ptr().add(idx) = 0;
                idx += 1;
            }
            *self.raw.ptr().add(last_idx) &= last_mask;
        }
    }

    pub fn is_set_at(&self, bit_offset: usize) -> bool {
        debug_assert!(bit_offset < i32::MAX as usize);
        let items = bit_offset >> Self::OFFSET_TO_ITEM_RSH;
        let bytes = unsafe { &*self.raw.ptr().add(items) };
        let mask = 1 << fast_mod(bit_offset as u32, Self::DIV, Self::FMM);
        (*bytes & mask) != 0
    }
}

unsafe fn split_move_to_new<T, A: Allocator + Clone>(
    vec: &mut RawVec<T, A>,
    req_cap: usize,
    split: usize,
    offset: usize,
    len: usize,
) {
    let new = RawVec::<T, A>::with_capacity_in(req_cap, vec.allocator().to_owned());

    ptr::copy(vec.ptr(), new.ptr(), split);
    ptr::copy(
        vec.ptr().add(split),
        new.ptr().add(split + offset),
        len - split,
    );
    ptr::drop_in_place(vec.ptr());
    vec.shrink_to_fit(0);
    *vec = new;
}

fn fast_mod(val: u32, div: u32, mul: u64) -> u32 {
    debug_assert!(div <= i32::MAX as u32);
    let hi = (((((mul * (val as u64)) >> 32) + 1) * (div as u64)) >> 32) as u32;
    debug_assert!(hi == val % div);
    hi
}

const fn num_bits<T>() -> usize {
    mem::size_of::<T>() * 8
}

fn align_bits<T>(bits: usize) -> usize {
    ng_mul_of!(bits, num_bits::<T>())
}
