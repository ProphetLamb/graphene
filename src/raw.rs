use core::{
    alloc::{Allocator, Layout},
    marker::PhantomData,
    mem::{self, MaybeUninit},
    ops::Index,
    ptr::{self, NonNull, Unique},
};

use alloc::{
    alloc::{handle_alloc_error, Global},
    raw_vec::RawVec,
    vec::Vec,
};

pub unsafe trait SliceIndex<Idx: Copy + Clone> {
    type Output;

    fn get(&self, index: Idx) -> Option<&Self::Output> {
        if self.guard_index(index) {
            Some(unsafe { self.get_unchecked(index) })
        } else {
            None
        }
    }

    fn get_mut(&mut self, index: Idx) -> Option<&mut Self::Output> {
        if self.guard_index(index) {
            Some(unsafe { self.get_mut_unchecked(index) })
        } else {
            None
        }
    }

    fn guard_index(&self, index: Idx) -> bool;

    unsafe fn get_unchecked(&self, index: Idx) -> &Self::Output;

    unsafe fn get_mut_unchecked(&mut self, index: Idx) -> &mut Self::Output;
}

/// The number of vertices in a pack.
/// Must always be smaller then `u16::MAX`.
///
/// The size of a `Pack` should is required to be as close as possible to a multiple of a page 4096 bytes when allocating.
/// By default `NUM_V_PAC` is chosen so that a `Pack` has a size of less then 2 pages.
pub const NUM_V_PAC: usize = 2 * 4096 / (4 * mem::size_of::<usize>()) - 8 * mem::size_of::<usize>();

/// The Edge represents a reference to a vertex in a graph.
/// The owner of the Edge is the origin of which the edge is outbound,
/// and the references vertex the target to which the edge is inbound.
///
/// The `size_of::<Edge>()` is 8 bytes which is the `sizeof::<usize>()` for 64bit operating systems,
/// and makes allocating easier when compared to the 7 bytes necessary to represent an `Edge`.
#[derive(Clone, Copy)]
pub struct Edge {
    /// Indicates whether the edge is marked as deleted or not.
    deleted_flag: u16,

    /// The index of the pack in the graph in which the vertex resides.
    pack_id: u32,

    /// The index of the vertex in the pack in which the vertex resides.
    /// Has a maximum value of `NUM_V_PAC`.
    vertex_id: u16,
}

/// Each `Vertex` represents a node in the graph and can have multiple outgoing edges.
/// Optionally each `Edge` may have a property value stored in `edge_properties` parallel to `adjacent_edges`
/// Managing and mutating vertices is the job of the `Pack` to which the respective vertex belongs.
pub struct Vertex<E> {
    /// The number of `adjacent_edges` and `edge_properties`.
    ///
    /// A value of :
    /// * `!0` indicates a deleted Vertex.
    /// * `1` indicates that `adjacent_edges` points directly to the adjacent vertex instead of a vector.
    len: usize,
    cap: usize,

    adjacent_edges: Unique<Edge>,

    edge_properties: Unique<E>,
}

impl<E> Vertex<E> {
    pub fn is_dropped(&self) -> bool {
        self.len == !0
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }
}

pub struct Pack<V, E, A: Allocator> {
    /// The `vertices` the pack contains.
    vertices: Unique<Vertex<E>>,

    /// The length of `vertices` of the pack.
    length: usize,

    /// The `vertex_properties` are parallel to `vertices`.
    vertex_properties: RawVec<V, A>,
}

impl<V, E, A: Allocator> Pack<V, E, A> {
    fn allocate_in(capacity: usize, alloc: A) -> Self {
        let (vertices, length) = alloc_array(capacity, &alloc);
        Self {
            vertices,
            length,
            vertex_properties: RawVec::new_in(alloc),
        }
    }

    fn allocate_with_props_in(capacity: usize, alloc: A) -> Self {
        let (vertices, length) = alloc_array(capacity, &alloc);
        Self {
            vertices,
            length,
            vertex_properties: RawVec::with_capacity_in(capacity, alloc),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.length
    }

    #[inline]
    pub fn allocator(&self) -> &A {
        self.vertex_properties.allocator()
    }

    pub fn insert(&mut self, vertex: Vertex<E>) {}

    fn get_prop(&self, index: usize) -> Option<&V> {
        if self.guard_index(index) {
            Some(unsafe { self.get_prop_unchecked(index) })
        } else {
            None
        }
    }

    fn get_prop_mut(&mut self, index: usize) -> Option<&mut V> {
        if self.guard_index(index) {
            Some(unsafe { self.get_prop_mut_unchecked(index) })
        } else {
            None
        }
    }

    #[inline]
    unsafe fn vertex_ptr_at_unchecked(&self, index: usize) -> *mut Vertex<E> {
        self.vertices.as_ptr().add(index)
    }

    #[inline]
    unsafe fn get_prop_unchecked(&self, index: usize) -> &V {
        &*self.prop_ptr_at_unchecked(index)
    }

    #[inline]
    unsafe fn get_prop_mut_unchecked(&mut self, index: usize) -> &mut V {
        &mut *self.prop_ptr_at_unchecked(index)
    }

    #[inline]
    unsafe fn prop_ptr_at_unchecked(&self, index: usize) -> *mut V {
        self.vertex_properties.ptr().add(index)
    }

    unsafe fn allocate_vertex_unchecked(&mut self, index: usize, capacity: usize) {
        let alloc = self.allocator();
        let (adjacent_edges, capacity) = alloc_array(capacity, alloc);
        let vertex = self.get_mut_unchecked(index);
        vertex.adjacent_edges = adjacent_edges;
        vertex.edge_properties = Unique::dangling();
        vertex.cap = capacity;
        vertex.len = 0;
    }

    unsafe fn allocate_vertex_with_props_unchecked(
        &mut self,
        vertex: &mut Vertex<E>,
        capacity: usize,
    ) {
        let alloc = self.allocator();
        let (adjacent_edges, capacity) = alloc_array(capacity, alloc);
        let (edge_properties, capacity) = alloc_array(capacity, alloc);
        vertex.adjacent_edges = adjacent_edges;
        vertex.edge_properties = edge_properties;
        vertex.cap = capacity;
        vertex.len = 0;
    }

    unsafe fn reserve_vertex_unchecked(&mut self, vertex: &mut Vertex<E>, additional: usize) {
        debug_assert!(additional != 0);
        let alloc = self.allocator();
        let len = vertex.len();
        let cap = vertex.cap;
        let required_cap = cap + additional;
        let (adjacent_edges, capacity) = alloc_array(required_cap, alloc);
        ptr::copy(vertex.adjacent_edges.as_ptr(), adjacent_edges.as_ptr(), len);
        self.drop_vertex_unchecked(vertex);
        vertex.adjacent_edges = adjacent_edges;
        vertex.edge_properties = Unique::dangling();
        vertex.cap = capacity;
        vertex.len = len;
    }

    unsafe fn reserve_vertex_with_props_unchecked(
        &mut self,
        vertex: &mut Vertex<E>,
        additional: usize,
    ) {
        debug_assert!(additional != 0);
        let alloc = self.allocator();
        let len = vertex.len();
        let cap = vertex.cap;
        let required_cap = cap + additional;
        let (adjacent_edges, capacity) = alloc_array(required_cap, alloc);
        ptr::copy(vertex.adjacent_edges.as_ptr(), adjacent_edges.as_ptr(), len);
        let (edge_properties, capacity) = alloc_array(required_cap, alloc);
        ptr::copy(vertex.adjacent_edges.as_ptr(), adjacent_edges.as_ptr(), len);
        self.drop_vertex_with_props_unchecked(vertex);
        vertex.adjacent_edges = adjacent_edges;
        vertex.edge_properties = edge_properties;
        vertex.cap = capacity;
        vertex.len = len;
    }

    unsafe fn drop_vertex_unchecked(&mut self, vertex: &mut Vertex<E>) {
        let (ptr, layout) = array_memory(vertex.adjacent_edges, vertex.cap);
        vertex.adjacent_edges = Unique::dangling();
        vertex.cap = 0;
        vertex.len = !0;
        self.allocator().deallocate(ptr, layout);
    }

    unsafe fn drop_vertex_with_props_unchecked(&mut self, vertex: &mut Vertex<E>) {
        let vertex = &mut *vertex;
        let (adj_ptr, adj_layout) = array_memory(vertex.adjacent_edges, vertex.cap);
        let (edge_ptr, edge_layout) = array_memory(vertex.edge_properties, vertex.cap);
        vertex.adjacent_edges = Unique::dangling();
        vertex.edge_properties = Unique::dangling();
        vertex.cap = 0;
        vertex.len = !0;
        self.allocator().deallocate(adj_ptr, adj_layout);
        self.allocator().deallocate(edge_ptr, edge_layout);
    }
}

unsafe impl<V, E, A: Allocator> SliceIndex<usize> for Pack<V, E, A> {
    type Output = Vertex<E>;

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> &Self::Output {
        &*self.vertex_ptr_at_unchecked(index)
    }

    #[inline]
    unsafe fn get_mut_unchecked(&mut self, index: usize) -> &mut Self::Output {
        &mut *self.vertex_ptr_at_unchecked(index)
    }

    #[inline]
    fn guard_index(&self, index: usize) -> bool {
        index < self.length && index < isize::MAX as usize
    }
}

impl<V, E, A: Allocator> Drop for Pack<V, E, A> {
    fn drop(&mut self) {
        let (ptr, layout) = array_memory(self.vertices, self.length);
        unsafe {
            self.allocator().deallocate(ptr, layout);
        };
    }
}

pub struct PackedSparseRow<V, E, A: Allocator = Global> {
    /// The vector of pointers to the packs.
    /// Reading an writing are atomic because `Unique` is a pointer.
    packs: Vec<Pack<V, E, A>, A>,
}

impl<V, E> PackedSparseRow<V, E> {
    #[inline]
    pub fn new() -> Self {
        Self::new_in(Global)
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
    }
}

impl<V, E, A: Allocator> PackedSparseRow<V, E, A> {
    #[inline]
    pub fn new_in(alloc: A) -> Self {
        Self {
            packs: Vec::new_in(alloc),
        }
    }

    #[inline]
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        Self {
            packs: Vec::with_capacity_in(capacity, alloc),
        }
    }

    /// Inserts a `Vertex` into the array, and returns the `Edge` referencing the inserted `Vertex`.
    pub fn insert(&mut self, vertex: Vertex<E>) -> Edge {
        todo!()
    }

    /// Deletes a `Vertex` from the array, by marking it as deleted.
    ///
    /// Returns `true` if the `Vertex` was deleted, otherwise; `false`.
    pub fn remove(&mut self, edge_to_vertex: Edge) -> bool {
        todo!()
    }

    pub fn get(&self, edge_to_vertex: Edge) -> Option<&Vertex<E>> {
        if self.check_edge(edge_to_vertex) {
            Some(unsafe { self.get_unchecked(edge_to_vertex) })
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, edge_to_vertex: Edge) -> Option<&Vertex<E>> {
        if self.check_edge(edge_to_vertex) {
            Some(unsafe { self.get_mut_unchecked(edge_to_vertex) })
        } else {
            None
        }
    }

    pub fn check_edge(&self, edge: Edge) -> bool {
        todo!()
    }

    pub unsafe fn get_mut_unchecked(&mut self, edge_to_vertex: Edge) -> &mut Vertex<E> {
        todo!()
    }

    pub unsafe fn get_unchecked(&self, edge_to_vertex: Edge) -> &Vertex<E> {
        todo!()
    }
}

fn alloc_array<T, A: Allocator>(capacity: usize, alloc: &A) -> (Unique<T>, usize) {
    debug_assert!(capacity != 0);
    debug_assert!(mem::size_of::<T>() != 0);

    let layout = match Layout::array::<T>(capacity) {
        Ok(layout) => layout,
        Err(_) => capacity_overflow(),
    };
    if !alloc_guard(layout.size()) {
        capacity_overflow();
    }
    let ptr = match alloc.allocate(layout) {
        Ok(ptr) => ptr,
        Err(_) => handle_alloc_error(layout),
    };

    (
        unsafe { Unique::new_unchecked(ptr.cast().as_ptr()) },
        capacity,
    )
}

fn array_memory<T>(ptr: Unique<T>, capacity: usize) -> (NonNull<u8>, Layout) {
    unsafe {
        (
            ptr.cast().into(),
            Layout::from_size_align_unchecked(mem::size_of::<T>() * capacity, mem::align_of::<T>()),
        )
    }
}

// We need to guarantee the following:
// * We don't ever allocate `> isize::MAX` byte-size objects.
// * We don't overflow `usize::MAX` and actually allocate too little.
//
// On 64-bit we just need to check for overflow since trying to allocate
// `> isize::MAX` bytes will surely fail. On 32-bit and 16-bit we need to add
// an extra guard for this in case we're running on a platform which can use
// all 4GB in user-space, e.g., PAE or x32.

#[inline]
fn alloc_guard(alloc_size: usize) -> bool {
    !(usize::BITS < 64 && alloc_size > isize::MAX as usize)
}

// One central function responsible for reporting capacity overflows. This'll
// ensure that the code generation related to these panics is minimal as there's
// only one location which panics rather than a bunch throughout the module.
#[cfg(not(no_global_oom_handling))]
fn capacity_overflow() -> ! {
    panic!("capacity overflow");
}
