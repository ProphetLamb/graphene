use core::{
    alloc::Allocator,
    mem,
    ops::{Index, IndexMut},
};

use alloc::{alloc::Global, raw_vec::RawVec};

/// Same interface as `SliceIndex` but for collection.
pub unsafe trait UnsafeIndex<Idx: Copy + ?Sized> {
    type Output: ?Sized;

    fn get(&self, index: Idx) -> Option<&Self::Output> {
        match self.guard_index(index) {
            Ok(_) => Some(unsafe { &*self.get_unchecked(index) }),
            Err(_) => None,
        }
    }

    fn get_mut(&mut self, index: Idx) -> Option<&mut Self::Output> {
        match self.guard_index(index) {
            Ok(_) => Some(unsafe { &mut *self.get_unchecked_mut(index) }),
            Err(_) => None,
        }
    }

    fn guard_index(&self, index: Idx) -> Result<(), ()>;

    unsafe fn get_unchecked(&self, index: Idx) -> *const Self::Output;

    unsafe fn get_unchecked_mut(&mut self, index: Idx) -> *mut Self::Output;
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
    edges_payloads: RawVec<E, A>,
    /// The total number of elements.
    len: usize,
    /// No free items exist before.
    first_free_hint: usize,
    /// No occupied items exists after.
    last_occupied_hint: usize,
}

impl<E, A: Allocator + Copy> Drop for Vertex<E, A> {
    fn drop(&mut self) {
        drop(self.edges_payloads);
        self.len = 0;
        self.first_free_hint = 0;
        self.last_occupied_hint = !0;
    }
}

impl<E, A: Allocator + Copy> Vertex<E, A> {
    pub fn new_in(alloc: A) -> Self {
        Self {
            edges: RawVec::new_in(alloc),
            edges_payloads: RawVec::new_in(alloc),
            edges_init: BitMarker::new(),
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

    pub fn insert(&mut self, edge: Edge, payload: E) -> Result<usize, ()> {
        if self.len() == self.cap() {
            return Err(());
        }
        let next_free = self.guard_insert()?;
        // The element at which we insert must be deleted.
        debug_assert!(unsafe { &*self.get_unchecked(next_free) }.is_del());
        unsafe { self.set_unchecked(next_free, edge, payload) };
        self.len += 1;
        Ok(next_free)
    }

    pub fn remove_at(&mut self, index: usize) -> Result<(), ()> {
        self.guard_remove(index)?;
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
    fn get_payload_unchecked(&self, index: usize) -> &E {
        unsafe { &*self.edges_payloads.ptr().add(index) }
    }

    #[inline]
    fn get_payload_unchecked_mut(&mut self, index: usize) -> &mut E {
        unsafe { &mut *self.edges_payloads.ptr().add(index) }
    }

    #[inline(never)]
    unsafe fn set_unchecked(&mut self, index: usize, edge: Edge, payload: E) {
        *self.get_unchecked_mut(index) = edge;
        if mem::size_of::<E>() != 0 {
            *self.get_payload_unchecked_mut(index) = payload;
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

unsafe impl<E, A: Allocator + Copy> UnsafeIndex<usize> for Vertex<E, A> {
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
    unsafe fn get_unchecked(&self, index: usize) -> *const Self::Output {
        self.edges.ptr().add(index)
    }

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> *mut Self::Output {
        self.edges.ptr().add(index)
    }
}

struct Pack<V, E, A: Allocator + Copy> {
    vertices: RawVec<Vertex<E, A>, A>,
    vertices_payloads: RawVec<V, A>,
    vertices_init: BitMarker,
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
            vertices_payloads: RawVec::with_capacity_in(capacity, alloc),
            vertices_init: BitMarker::new(capacity),
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

    pub fn insert(&mut self, vertex: Vertex<E, A>, payload: V) -> Result<usize, ()> {
        let next_free = self.guard_insert()?;
        // The element at which we insert must be deleted.
        if self.vertices_init.is_set_at(next_free) {
            return Err(());
        }
        self.vertices_init.mark_at(next_free);
        unsafe { self.set_unchecked(next_free, vertex, payload) };
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
    fn get_payload_unchecked(&self, index: usize) -> &V {
        unsafe { &*self.vertices_payloads.ptr().add(index) }
    }

    #[inline]
    fn get_payload_unchecked_mut(&mut self, index: usize) -> &mut V {
        unsafe { &mut *self.vertices_payloads.ptr().add(index) }
    }

    #[inline(never)]
    unsafe fn set_unchecked(&mut self, index: usize, vertex: Vertex<E, A>, payload: V) {
        *self.get_unchecked_mut(index) = vertex;
        if mem::size_of::<V>() != 0 {
            *self.get_payload_unchecked_mut(index) = payload;
        }
    }
}

unsafe impl<V, E, A: Allocator + Copy> UnsafeIndex<usize> for Pack<V, E, A> {
    type Output = Vertex<E, A>;

    fn guard_index(&self, index: usize) -> Result<(), ()> {
        if index < self.vertices.capacity() {
            Ok(())
        } else {
            Err(())
        }
    }

    unsafe fn get_unchecked(&self, index: usize) -> *const Self::Output {
        self.vertices.ptr().add(index)
    }

    unsafe fn get_unchecked_mut(&mut self, index: usize) -> *mut Self::Output {
        self.vertices.ptr().add(index)
    }
}

struct SparsePackedRows<V, E, A: Allocator + Copy> {
    packs: RawVec<Pack<V, E, A>>,
}

impl<V, E, A: Allocator + Copy> SparsePackedRows<V, E, A> {
    pub fn with_capacity_in(capacity: usize, alloc: A) {
        todo!()
    }
}

fn index_out_of_range() -> ! {
    panic!("The index is outside the range of valid values.")
}

struct BitMarker {
    raw: RawVec<usize>,
    len: usize,
    fmm: u64,
}

impl BitMarker {
    const OFFSET_TO_ITEM_RSH: usize = if mem::size_of::<usize>() == 2 {
        1
    } else if mem::size_of::<usize>() == 4 {
        2
    } else {
        4
    };

    pub fn new(bits: usize) -> BitMarker {
        assert!(bits < u32::MAX as usize);
        let items = align_bits::<usize>(bits);
        BitMarker {
            raw: RawVec::with_capacity_zeroed(items),
            len: items,
            fmm: get_fast_mod_mul(num_bits::<usize>() as u32),
        }
    }

    pub fn reserve(&mut self, additional_bits: usize) {
        self.raw
            .reserve(self.len, align_bits::<usize>(additional_bits));
    }

    pub fn mark_at(&mut self, bit_offset: usize) {
        debug_assert!(bit_offset < i32::MAX as usize);
        let items = bit_offset >> Self::OFFSET_TO_ITEM_RSH;
        let bytes = unsafe { &mut *self.raw.ptr().add(items) };
        let mask = 1 << fast_mod(bit_offset as u32, num_bits::<usize>() as u32, self.fmm);
        *bytes |= mask;
    }

    pub fn clear_at(&mut self, bit_offset: usize) {
        debug_assert!(bit_offset < i32::MAX as usize);
        let items = bit_offset >> Self::OFFSET_TO_ITEM_RSH;
        let bytes = unsafe { &mut *self.raw.ptr().add(items) };
        let mask = 1 << fast_mod(bit_offset as u32, num_bits::<usize>() as u32, self.fmm);
        *bytes &= !mask;
    }

    pub fn is_set_at(&self, bit_offset: usize) -> bool {
        debug_assert!(bit_offset < i32::MAX as usize);
        let items = bit_offset >> Self::OFFSET_TO_ITEM_RSH;
        let bytes = unsafe { &*self.raw.ptr().add(items) };
        let mask = 1 << fast_mod(bit_offset as u32, num_bits::<usize>() as u32, self.fmm);
        (*bytes & mask) != 0
    }
}

/// https://arxiv.org/abs/1902.01961
fn get_fast_mod_mul(div: u32) -> u64 {
    u64::MAX / div as u64 + 1
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
    (bits - 1) / num_bits::<T>() + 1
}

fn log_2(x: i32) -> u32 {
    assert!(x > 0);
    num_bits::<i32>() as u32 - x.leading_zeros() - 1
}
