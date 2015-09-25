"""
Keeps an inverted index in memory in a numpy array. Intersections are fast
and performed in compiled code.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""

from cpython cimport array
from array import array
from libc.stdlib cimport free
import struct
import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.int
ctypedef np.int_t DTYPE_t
ctypedef np.uint32_t DTYPE_uint32

@cython.boundscheck(False) # turn of bounds-checking for entire function
cdef read_posting_list(np.ndarray[DTYPE_uint32, ndim=1] index, int word_id,
                      np.ndarray[DTYPE_uint32, ndim=1] offsets,
                      np.ndarray[DTYPE_uint32, ndim=1] sizes):
    """Read the posting list from the memory kept index.

    :param index:
    :param word_id:
    :param offsets:
    :param sizes:
    :return:
    """
    cdef int offset = offsets[word_id]
    cdef int size = sizes[word_id]
    cdef np.ndarray[DTYPE_uint32, ndim=1] posting_list = index[offset:offset + size]
    return posting_list

@cython.boundscheck(False) # turn of bounds-checking for entire function
def compute_intersection(int id_a, int id_b,
                         np.ndarray[DTYPE_uint32, ndim=1] index,
                         np.ndarray[DTYPE_uint32, ndim=1] offsets,
                         np.ndarray[DTYPE_uint32, ndim=1] sizes):
    """Compute the intersection for two ids.

    This fetches two lists and intersects them.
    :param id_a:
    :param id_b:
    :param index:
    :param offsets:
    :param sizes:
    :return:
    """
    posting_list_a = read_posting_list(index, id_a, offsets, sizes)
    posting_list_b = read_posting_list(index, id_b, offsets, sizes)
    return intersect_postings_cython(posting_list_a, posting_list_b)

@cython.boundscheck(False) # turn of bounds-checking for entire function
def compute_intersection_fast(int id_a, int id_b,
                         np.ndarray[DTYPE_uint32, ndim=1] index,
                         np.ndarray[DTYPE_uint32, ndim=1] offsets,
                         np.ndarray[DTYPE_uint32, ndim=1] sizes):
    """Compute the intersection for two ids inplace (faster).

    This does not fetch the lists for the two ids but intersects directly
    from the index.

    :param id_a:
    :param id_b:
    :param index:
    :param offsets:
    :param sizes:
    :return:
    """
    cdef int start_a = offsets[id_a]
    cdef int size_a = sizes[id_a]
    cdef int start_b = offsets[id_b]
    cdef int size_b = sizes[id_b]
    intersection_result = []
    cdef int i_a = 0
    cdef int i_b = 0
    cdef DTYPE_t v_l_b
    cdef DTYPE_t v_r_a
    cdef DTYPE_t v_r_b
    cdef DTYPE_uint32 l_a
    cdef DTYPE_uint32 l_b
    cdef DTYPE_uint32 r_a
    cdef DTYPE_uint32 r_b
    while i_a < size_a and i_b < size_b:
        l_a = index[start_a + i_a]
        r_a = index[start_a + i_a + 1]
        l_b = index[start_b + i_b]
        r_b = index[start_b + i_b + 1]
        if l_a == l_b:
            v_l_b = l_b
            v_r_a = r_a
            v_r_b = r_b
            intersection_result.append((v_l_b, v_r_a, v_r_b))
            i_a += 2
            i_b += 2
        elif l_a < l_b:
            i_a += 2
        else:
            i_b += 2
    return intersection_result


def compute_intersection_for_list(ids,
                                  np.ndarray[DTYPE_uint32, ndim=1] index,
                                  np.ndarray[DTYPE_uint32, ndim=1] offsets,
                                  np.ndarray[DTYPE_uint32, ndim=1] sizes):
    """Intersect the lists for several ids.

    The first two ids are intersected in a fast way. Each subsequent
    intersection is slower.

    :param ids:
    :param index:
    :param offsets:
    :param sizes:
    :return:
    """
    # Sort the ids by increasing size of their poisting list.
    size_ids = sorted([(sizes[id], id) for id in ids], key= lambda x: x[0])
    # Intersect the two smallest.
    intersection = compute_intersection_fast(size_ids[0][1], size_ids[1][1],
                                             index, offsets, sizes)
    if not intersection:
        return []
    posting_lists = []
    for size, id in size_ids[2:]:
        posting_list = read_posting_list(index, id, offsets, sizes)
        posting_lists.append(posting_list)
        if posting_list.size == 0:
            return []
    #posting_lists = sorted(posting_lists,
    #                       key=lambda x: x.size)
    # Intersect the rest in python.
    if intersection:
        for posting_list in posting_lists:
            intersection = intersect_result_with_posting(intersection,
                                                         posting_list)
            if not intersection:
                break
    return intersection

@cython.boundscheck(False) # turn of bounds-checking for entire function
def intersect_result_with_posting(result,
                                  np.ndarray[DTYPE_uint32, ndim=1] postings_a):
    """Intersect existing result list with the posting list for entity id.

    Returns a list of tuples (mediator_id, rel_a, rel_b ...)
     where rel_x are the relation ids from the mediator to the respective entity.

    :param entity_id_a:
    :param entity_id_b:
    :return:
    """
    intersection_result = []
    cdef int i_a = 0
    cdef int i_b = 0
    cdef DTYPE_uint32 l_a
    cdef DTYPE_uint32 r_a
    while i_a < postings_a.size and i_b < len(result):
        l_a = postings_a[i_a]
        r_a = postings_a[i_a + 1]
        mediator_id = result[i_b][0]
        if l_a == mediator_id:
            intersection_result.append(tuple([x for x in result[i_b]] + [r_a]))
            i_a += 2
            i_b += 2
        elif l_a < mediator_id:
            i_a += 2
        else:
            i_b += 2
    return intersection_result


def read_index(index_prefix):
    """Read index from disk.

    :param index_prefix:
    :return:
    """
    vocabulary_words = []
    reverse_vocab = dict()
    cdef int value_id
    offsets = None
    cdef np.ndarray[DTYPE_uint32, ndim=1] index
    with open(index_prefix + ".vocabulary", "r") as f:
        for line in f:
            name = line.strip()
            vocabulary_words.append(name)
    with open(index_prefix + ".offsets", "rb") as f:
        offsets = np.fromstring(f.read(), dtype=np.uint32)
    with open(index_prefix + ".sizes", "rb") as f:
        sizes = np.fromstring(f.read(), dtype=np.uint32)
    with open(index_prefix + ".bin", "rb") as f:
        index = np.fromstring(f.read(), dtype=np.uint32)
    return index, vocabulary_words,\
           offsets, sizes


def write_index(index_prefix, vocabulary_words, postings):
    """Write index to disk.

    :param index_prefix:
    :param vocabulary_words:
    :param postings:
    :return:
    """
    cdef int value_id
    cdef int mediator_id
    cdef int relation_id
    cdef int offset = 0
    cdef int current_length = 0
    cdef int end
    cdef np.ndarray[DTYPE_uint32, ndim=1] index
    offsets_np = np.zeros(len(vocabulary_words), dtype=np.uint32)
    sizes_np = np.zeros(len(vocabulary_words), dtype=np.uint32)
    with open(index_prefix + ".bin", "wb") as f:
        for value_id in range(len(vocabulary_words)):
            offsets_np[value_id] = offset
            if value_id in postings:
                posting_list = postings[value_id]
                # Note: to string returns bytes, not an actual string.
                f.write(posting_list.tostring())
                current_length += len(posting_list)
            sizes_np[value_id] = current_length - offset
            offset = current_length
    with open(index_prefix + ".offsets", "w") as f:
        f.write(offsets_np.tostring())
    with open(index_prefix + ".sizes", "w") as f:
        f.write(sizes_np.tostring())
    with open(index_prefix + ".vocabulary", "w") as f:
        for word in vocabulary_words:
            f.write("%s\n" % word)
    # Re-read the index
    with open(index_prefix + ".bin", "rb") as f:
        index = np.fromstring(f.read(), dtype=np.uint32)
    return index, offsets_np, sizes_np



@cython.boundscheck(False) # turn of bounds-checking for entire function
def intersect_postings_cython(np.ndarray[DTYPE_uint32, ndim=1] postings_a,
                              np.ndarray[DTYPE_uint32, ndim=1] postings_b):
    """Straightforward intersection of two lists.

    :param postings_a:
    :param postings_b:
    :return:
    """
    intersection_result = []

    cdef int i_a = 0
    cdef int i_b = 0
    cdef DTYPE_t v_l_b
    cdef DTYPE_t v_r_a
    cdef DTYPE_t v_r_b
    cdef DTYPE_uint32 l_a
    cdef DTYPE_uint32 l_b
    cdef DTYPE_uint32 r_a
    cdef DTYPE_uint32 r_b
    while i_a < postings_a.size and i_b < postings_b.size:
        l_a = postings_a[i_a]
        r_a = postings_a[i_a + 1]
        l_b = postings_b[i_b]
        r_b = postings_b[i_b + 1]
        if l_a == l_b:
            v_l_b = l_b
            v_r_a = r_a
            v_r_b = r_b
            intersection_result.append((v_l_b, v_r_a, v_r_b))
            i_a += 2
            i_b += 2
        elif l_a < l_b:
            i_a += 2
        else:
            i_b += 2
    return intersection_result
