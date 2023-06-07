


#ifndef ANNOY_ANNOYLIB_H
#define ANNOY_ANNOYLIB_H

#include <chrono>

#include <stdio.h>
#include <sys/stat.h>
#ifndef _MSC_VER
#include <unistd.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <fcntl.h>
#include <stddef.h>



#if defined(_MSC_VER) && _MSC_VER == 1500
typedef unsigned char     uint8_t;
typedef signed __int32    int32_t;
typedef unsigned __int64  uint64_t;
typedef signed __int64    int64_t;
#else
#include <stdint.h>
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
 // a bit hacky, but override some definitions to support 64 bit
 #define off_t int64_t
 #define lseek_getsize(fd) _lseeki64(fd, 0, SEEK_END)
 #ifndef NOMINMAX
  #define NOMINMAX
 #endif
 #include "mman.h"
 #include <windows.h>
#else
 #include <sys/mman.h>
 #define lseek_getsize(fd) lseek(fd, 0, SEEK_END)
#endif

#include <cerrno>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <queue>
#include <list>
#include <limits>

#if __cplusplus >= 201103L
#include <type_traits>
#endif



#ifdef _MSC_VER
// Needed for Visual Studio to disable runtime checks for mempcy
#pragma runtime_checks("s", off)
#endif

// This allows others to supply their own logger / error printer without
// requiring Annoy to import their headers. See RcppAnnoy for a use case.
#ifndef __ERROR_PRINTER_OVERRIDE__
  #define annoylib_showUpdate(...) { fprintf(stderr, __VA_ARGS__ ); }
#else
  #define annoylib_showUpdate(...) { __ERROR_PRINTER_OVERRIDE__( __VA_ARGS__ ); }
#endif

// Portable alloc definition, cf Writing R Extensions, Section 1.6.4
#ifdef __GNUC__
  // Includes GCC, clang and Intel compilers
  # undef alloca
  # define alloca(x) __builtin_alloca((x))
#elif defined(__sun) || defined(_AIX)
  // this is necessary (and sufficient) for Solaris 10 and AIX 6:
  # include <alloca.h>
#endif

// We let the v array in the Node struct take whatever space is needed, so this is a mostly insignificant number.
// Compilers need *some* size defined for the v array, and some memory checking tools will flag for buffer overruns if this is set too low.
#define ANNOYLIB_V_ARRAY_SIZE 65536

#ifndef _MSC_VER
#define annoylib_popcount __builtin_popcountll
#else // See #293, #358
#define annoylib_popcount cole_popcount
#endif

#if !defined(NO_MANUAL_VECTORIZATION) && defined(__GNUC__) && (__GNUC__ >6) && defined(__AVX512F__)  // See #402
#define ANNOYLIB_USE_AVX512
#elif !defined(NO_MANUAL_VECTORIZATION) && defined(__AVX__) && defined (__SSE__) && defined(__SSE2__) && defined(__SSE3__)
#define ANNOYLIB_USE_AVX
#else
#endif

#if defined(ANNOYLIB_USE_AVX) || defined(ANNOYLIB_USE_AVX512)
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__)
#include <x86intrin.h>
#endif
#endif

#if !defined(__MINGW32__)
#define ANNOYLIB_FTRUNCATE_SIZE(x) static_cast<int64_t>(x)
#else
#define ANNOYLIB_FTRUNCATE_SIZE(x) (x)
#endif

namespace Annoy {

inline void set_error_from_errno(char **error, const char* msg) {
  annoylib_showUpdate("%s: %s (%d)\n", msg, strerror(errno), errno);
  if (error) {
    *error = (char *)malloc(256);  // TODO: win doesn't support snprintf
    snprintf(*error, 255, "%s: %s (%d)", msg, strerror(errno), errno);
  }
}

inline void set_error_from_string(char **error, const char* msg) {
  annoylib_showUpdate("%s\n", msg);
  if (error) {
    *error = (char *)malloc(strlen(msg) + 1);
    strcpy(*error, msg);
  }
}


using std::vector;
using std::pair;
using std::numeric_limits;
using std::make_pair;



// remap_memory_and_truncate(&_nodes, _fd, _s * _nodes_size, _s * new_nodes_size) 
inline bool remap_memory_and_truncate(void** _ptr, 
                    int _fd, size_t old_size, size_t new_size) {

#ifdef __linux__ // yes

    // new_size is getting larger and larger, eventually larger than physical memory size.
    *_ptr = mremap(*_ptr, old_size, new_size, MREMAP_MAYMOVE);

    // extend the file if new_size > current file size.
    bool ok = ftruncate(_fd, new_size) != -1; 

#else
    munmap(*_ptr, old_size);
    bool ok = ftruncate(_fd, ANNOYLIB_FTRUNCATE_SIZE(new_size)) != -1;

#ifdef MAP_POPULATE
    *_ptr = mmap(*_ptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, _fd, 0);
#else
    *_ptr = mmap(*_ptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, _fd, 0);
#endif
#endif

    return ok;
}




namespace {

template<typename S, typename Node>
inline Node* get_node_ptr(const void* _nodes, const size_t _s, const S i) {
  return (Node*)((uint8_t *)_nodes + (_s * i));
}


template<typename T>
__device__ __host__
inline T dot(const T* x, const T* y, int f) {
  T s = 0;
  for (int z = 0; z < f; z++) {
    s += (*x) * (*y);
    x++;
    y++;
  }
  return s;
}

template<typename T>
inline T get_norm(T* v, int f) {
  return sqrt(dot(v, v, f));
}

template<typename T, typename Random, typename Distance, typename Node>
inline void two_means(const vector<Node*>& nodes, int f, 
                        Random& random, bool cosine, Node* p, Node* q) {
  
  /*
    This algorithm is a huge heuristic. Empirically it works really well, but I
    can't motivate it well. The basic idea is to keep two centroids and assign
    points to either one of them. We weight each centroid by the number of points
    assigned to it, so to balance it. 
  */

  static int iteration_steps = 200;
  size_t count = nodes.size();

  size_t i = random.index(count);
  size_t j = random.index(count-1);
  j += (j >= i); // ensure that i != j

  Distance::template copy_node<T, Node>(p, nodes[i], f);
  Distance::template copy_node<T, Node>(q, nodes[j], f);

  if (cosine) { // yes
    Distance::template normalize<T, Node>(p, f); 
    Distance::template normalize<T, Node>(q, f);
  }

  Distance::init_node(p, f);
  Distance::init_node(q, f);


  int ic = 1, jc = 1;

  for (int l = 0; l < iteration_steps; l++) {
   
    size_t k = random.index(count);
 
    T di = ic * Distance::distance(p, nodes[k], f);
    T dj = jc * Distance::distance(q, nodes[k], f);
  
    T norm = cosine ? get_norm(nodes[k]->v, f) : 1;  // cosine == true
    
    if (!(norm > T(0))) {
      continue;
    }


    if (di < dj) {
      
      for (int z = 0; z < f; z++)
        p->v[z] = (p->v[z] * ic + nodes[k]->v[z] / norm) / (ic + 1);
     
      Distance::init_node(p, f);
      ic++;
    } 
    else if (dj < di) {
      
      for (int z = 0; z < f; z++)
        q->v[z] = (q->v[z] * jc + nodes[k]->v[z] / norm) / (jc + 1);
      
      Distance::init_node(q, f);
      jc++;
    }
  }
}


// template<typename S>
// void swap(vector<S>& arr, int id1, int id2){
//   S tmp = arr[id1];
//   arr[id1] = arr[id2];
//   arr[id2] = tmp;
// }

// template<typename T>
// void swap(T *arr, int id1, int id2){
//   T tmp = arr[id1];
//   arr[id1] = arr[id2];
//   arr[id2] = tmp;
// }

} // namespace




struct Base {
  template<typename T, typename S, typename Node>
  static inline void preprocess(void* nodes, size_t _s, const S node_count, const int f) {
    // Override this in specific metric structs below if you need to do any pre-processing
    // on the entire set of nodes passed into this index.
  }

  template<typename Node>
  static inline void zero_value(Node* dest) {
    // Initialize any fields that require sane defaults within this node.
  }

  template<typename T, typename Node>
  static inline void copy_node(Node* dest, const Node* source, const int f) {
    memcpy(dest->v, source->v, f * sizeof(T));
  }

  template<typename T, typename Node>
  static inline void normalize(Node* node, int f) {
    T norm = get_norm(node->v, f);
    if (norm > 0) {
      for (int z = 0; z < f; z++)
        node->v[z] /= norm;
    }
  }
};



struct Angular : Base {
  template<typename S, typename T>
  struct Node {
    /*
     * We store a binary tree where each node has two things
     * - A vector associated with it
     * - Two children
     * All nodes occupy the same amount of memory
     * All nodes with n_descendants == 1 are leaf nodes.
     * A memory optimization is that for nodes with 2 <= n_descendants <= K,
     * we skip the vector. Instead we store a list of all descendants. K is
     * determined by the number of items that fits in the space of the vector.
     * For nodes with n_descendants == 1 the vector is a data point.
     * For nodes with n_descendants > K the vector is the normal of the split plane.
     * Note that we can't really do sizeof(node<T>) because we cheat and allocate
     * more memory to be able to fit the vector outside
     */
    S n_descendants;
    union {
      S children[2]; // Will possibly store more than 2
      T norm;
    };
    T v[ANNOYLIB_V_ARRAY_SIZE];
  };


  template<typename S, typename T>
  static inline T distance(const Node<S, T>* x, const Node<S, T>* y, int f) {
    // want to calculate (a/|a| - b/|b|)^2
    // = a^2 / a^2 + b^2 / b^2 - 2ab/|a||b|
    // = 2 - 2cos
    T pp = x->norm ? x->norm : dot(x->v, x->v, f); // For backwards compatibility reasons, we need to fall back and compute the norm here
    T qq = y->norm ? y->norm : dot(y->v, y->v, f);
    T pq = dot(x->v, y->v, f);
    T ppqq = pp * qq;
    if (ppqq > 0) return 2.0 - 2.0 * pq / sqrt(ppqq);
    else return 2.0; // cos is 0
  }



  template<typename S, typename T>
  static inline T margin(const Node<S, T>* n, const T* y, int f) {
    return dot(n->v, y, f);
  }


  template<typename S, typename T, typename Random>
  static inline bool side(const Node<S, T>* n, const T* y, int f, Random& random) {

    // printf("----------------------- side()\n");

    T dot = margin(n, y, f);

    if (dot != 0)
      return (dot > 0);
    else
      return (bool)random.flip();
  }


  
  template<typename S, typename T, typename Random>
  static inline void create_split(const vector<Node<S, T>*>& nodes, 
              int f, size_t s, Random& random, Node<S, T>* n) {

    Node<S, T>* p = (Node<S, T>*)alloca(s);
    Node<S, T>* q = (Node<S, T>*)alloca(s);

    two_means<T, Random, Angular, Node<S, T> >(nodes, f, random, true, p, q);

    for (int z = 0; z < f; z++)
      n->v[z] = p->v[z] - q->v[z];

    Base::normalize<T, Node<S, T> >(n, f);
  }

  template<typename T>
  static inline T normalized_distance(T distance) {
    // Used when requesting distances from Python layer
    // Turns out sometimes the squared distance is -0.0
    // so we have to make sure it's a positive number.
    return sqrt(std::max(distance, T(0)));
  }


  template<typename T>
  static inline T pq_distance(T distance, T margin, int child_nr) {
    if (child_nr == 0)
      margin = -margin;
    return std::min(distance, margin);
  }


  template<typename T>
  static inline T pq_initial_value() {
    return numeric_limits<T>::infinity();
  }
  template<typename S, typename T>
  static inline void init_node(Node<S, T>* n, int f) {
    n->norm = dot(n->v, n->v, f);
  }

  static const char* name() {
    return "angular";
  }
};





template<typename S, typename T, typename R = uint64_t>
class AnnoyIndexInterface {
 public:
  // Note that the methods with an **error argument will allocate memory and write the pointer to that string if error is non-NULL
  virtual ~AnnoyIndexInterface() {};
  virtual bool add_item(S item, const T* w, char** error=NULL) = 0;
  virtual bool build(int q, int n_threads=-1, char** error=NULL) = 0;
  virtual bool unbuild(char** error=NULL) = 0;
  virtual bool save(const char* filename, bool prefault=false, char** error=NULL) = 0;
  virtual void unload() = 0;
  virtual bool load(const char* filename, bool prefault=false, char** error=NULL) = 0;
  virtual T get_distance(S i, S j) const = 0;
  virtual void get_nns_by_item(S item, size_t n, int search_k, vector<S>* result, vector<T>* distances) const = 0;
  virtual void get_nns_by_vector(const T* w, size_t n, int search_k, vector<S>* result, vector<T>* distances) const = 0;
  virtual S get_n_items() const = 0;
  virtual S get_n_trees() const = 0;
  virtual void verbose(bool v) = 0;
  virtual void get_item(S item, T* v) const = 0;
  virtual void set_seed(R q) = 0;
  virtual bool on_disk_build(const char* filename, char** error=NULL) = 0;
};






template<typename S, typename T, typename Distance, typename Random, class ThreadedBuildPolicy>
  class AnnoyIndex : public AnnoyIndexInterface<S, T, 
#if __cplusplus >= 201103L
    typename std::remove_const<decltype(Random::default_seed)>::type
#else
    typename Random::seed_type
#endif
    > {

public:

  typedef Distance D;
  typedef typename D::template Node<S, T> Node;

#if __cplusplus >= 201103L
  typedef typename std::remove_const<decltype(Random::default_seed)>::type R;
#else
  typedef typename Random::seed_type R;
#endif


  const int _f;
  size_t _s; // Size of each node in bytes.
  S _n_items;
  void* _nodes; // Could either be mmapped, or point to a memory buffer that we reallocate
  S _n_nodes;
  S _nodes_size;
  vector<S> _roots;
  S _K;
  R _seed;
  bool _loaded;
  bool _verbose;
  int _fd;
  bool _on_disk;
  bool _built;



   AnnoyIndex(int f) : _f(f), _seed(Random::default_seed) {
    _s = offsetof(Node, v) + _f * sizeof(T); // Size of each node
    _verbose = false;
    _built = false;

    // Max number of descendants to fit into node (space of children[2] + v[]).
    _K = (S) (((size_t) (_s - offsetof(Node, children))) / sizeof(S)); // 82

    // printf("_s: %d \n", (int)_s);
    // printf("offsetof(Node, children): %d \n", (int)offsetof(Node, children));
    // printf("sizeof(S): %d \n", (int)sizeof(S));

    reinitialize(); // Reset everything
  }


  ~AnnoyIndex() {
    unload();
  }

  int get_f() const {
    return _f;
  }


  bool add_item(S item, const T* w, char** error=NULL) {
    return add_item_impl(item, w, error);
  }



  template<typename W>
  bool add_item_impl(S item, const W& w, char** error=NULL) {
    if (_loaded) {
      set_error_from_string(error, "You can't add an item to a loaded index");
      return false;
    }

    _allocate_size(item + 1);
    Node* n = _get(item);

    D::zero_value(n);

    n->children[0] = 0;
    n->children[1] = 0;
    n->n_descendants = 1;

    for (int z = 0; z < _f; z++)
      n->v[z] = w[z];

    D::init_node(n, _f);

    if (item >= _n_items)
      _n_items = item + 1;

    return true;
  }
    

  // Prepares annoy to build the index in the specified file instead of RAM .
  // Execute before adding items, no need to save after build.
  bool on_disk_build(const char* file, char** error=NULL) {

    _on_disk = true;
    _fd = open(file, O_RDWR | O_CREAT | O_TRUNC, (int) 0600);

    if (_fd == -1) {
      set_error_from_errno(error, "Unable to open");
      _fd = 0;
      return false;
    }

    _nodes_size = 1;
    if (ftruncate(_fd, ANNOYLIB_FTRUNCATE_SIZE(_s) * ANNOYLIB_FTRUNCATE_SIZE(_nodes_size)) == -1) {
      set_error_from_errno(error, "Unable to truncate");
      return false;
    }


#ifdef MAP_POPULATE
    _nodes = (Node*) mmap(0, _s * _nodes_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, _fd, 0);
#else
    _nodes = (Node*) mmap(0, _s * _nodes_size, PROT_READ | PROT_WRITE, MAP_SHARED, _fd, 0);
#endif
    return true;
  }

  

  bool build(int q, int n_threads=-1, char** error=NULL) {

    if (_loaded) {
      set_error_from_string(error, "You can't build a loaded index");
      return false;
    }

    if (_built) {
      set_error_from_string(error, "You can't build a built index");
      return false;
    }

    D::template preprocess<T, S, Node>(_nodes, _s, _n_items, _f);

    _n_nodes = _n_items;

    ThreadedBuildPolicy::template build<S, T>(this, q, n_threads);


    // Also, copy the roots into the last segment of the array
    // This way we can load them faster without reading the whole file
    _allocate_size(_n_nodes + (S)_roots.size());

    for (size_t i = 0; i < _roots.size(); i++)
      memcpy(_get(_n_nodes + (S)i), _get(_roots[i]), _s);

    _n_nodes += _roots.size();


    if (_on_disk) {

      if (!remap_memory_and_truncate(&_nodes, _fd,
          static_cast<size_t>(_s) * static_cast<size_t>(_nodes_size),
          static_cast<size_t>(_s) * static_cast<size_t>(_n_nodes))) {
        
        // TODO: this probably creates an index in a corrupt state... not sure what to do
        set_error_from_errno(error, "Unable to truncate");
        return false;
      }
      _nodes_size = _n_nodes;
    }

    _built = true;

    return true;
  }



  
  bool unbuild(char** error=NULL) {
    if (_loaded) {
      set_error_from_string(error, "You can't unbuild a loaded index");
      return false;
    }

    _roots.clear();
    _n_nodes = _n_items;
    _built = false;

    return true;
  }

  bool save(const char* filename, bool prefault=false, char** error=NULL) {
    if (!_built) {
      set_error_from_string(error, "You can't save an index that hasn't been built");
      return false;
    }
    if (_on_disk) {
      return true;
    } else {
      // Delete file if it already exists (See issue #335)
      unlink(filename);

      FILE *f = fopen(filename, "wb");
      if (f == NULL) {
        set_error_from_errno(error, "Unable to open");
        return false;
      }

      if (fwrite(_nodes, _s, _n_nodes, f) != (size_t) _n_nodes) {
        set_error_from_errno(error, "Unable to write");
        return false;
      }

      if (fclose(f) == EOF) {
        set_error_from_errno(error, "Unable to close");
        return false;
      }

      unload();
      return load(filename, prefault, error);
    }
  }

  void reinitialize() {
    _fd = 0;
    _nodes = NULL;
    _loaded = false;
    _n_items = 0;
    _n_nodes = 0;
    _nodes_size = 0;
    _on_disk = false;
    _seed = Random::default_seed;
    _roots.clear();
  }

  void unload() {
    if (_on_disk && _fd) {
      close(_fd);
      munmap(_nodes, _s * _nodes_size);
    } else {
      if (_fd) {
        // we have mmapped data
        close(_fd);
        munmap(_nodes, _n_nodes * _s);
      } else if (_nodes) {
        // We have heap allocated data
        free(_nodes);
      }
    }
    reinitialize();
    if (_verbose) annoylib_showUpdate("unloaded\n");
  }

  bool load(const char* filename, bool prefault=false, char** error=NULL) {
    _fd = open(filename, O_RDONLY, (int)0400);
    if (_fd == -1) {
      set_error_from_errno(error, "Unable to open");
      _fd = 0;
      return false;
    }
    off_t size = lseek_getsize(_fd);
    if (size == -1) {
      set_error_from_errno(error, "Unable to get size");
      return false;
    } else if (size == 0) {
      set_error_from_errno(error, "Size of file is zero");
      return false;
    } else if (size % _s) {
      // Something is fishy with this index!
      set_error_from_errno(error, "Index size is not a multiple of vector size. Ensure you are opening using the same metric you used to create the index.");
      return false;
    }

    int flags = MAP_SHARED;
    if (prefault) {
#ifdef MAP_POPULATE
      flags |= MAP_POPULATE;
#else
      annoylib_showUpdate("prefault is set to true, but MAP_POPULATE is not defined on this platform");
#endif
    }
    _nodes = (Node*)mmap(0, size, PROT_READ, flags, _fd, 0);
    _n_nodes = (S)(size / _s);

    // Find the roots by scanning the end of the file and taking the nodes with most descendants
    _roots.clear();
    S m = -1;
    for (S i = _n_nodes - 1; i >= 0; i--) {
      S k = _get(i)->n_descendants;
      if (m == -1 || k == m) {
        _roots.push_back(i);
        m = k;
      } else {
        break;
      }
    }
    // hacky fix: since the last root precedes the copy of all roots, delete it
    if (_roots.size() > 1 && _get(_roots.front())->children[0] == _get(_roots.back())->children[0])
      _roots.pop_back();
    _loaded = true;
    _built = true;
    _n_items = m;
    if (_verbose) annoylib_showUpdate("found %lu roots with degree %d\n", _roots.size(), m);
    return true;
  }

  T get_distance(S i, S j) const {
    return D::normalized_distance(D::distance(_get(i), _get(j), _f));
  }


  // n: number of neighbors to return.
  // search_k: search `search_k` number of nodes to find neighbors. 
  void get_nns_by_item(S item, size_t n, int search_k, 
                vector<S>* result, vector<T>* distances) const {
    // TODO: handle OOB
    const Node* m = _get(item);
    _get_all_nns(m->v, n, search_k, result, distances);
  }



  void get_nns_by_vector(const T* w, size_t n, int search_k, 
                vector<S>* result, vector<T>* distances) const {
    _get_all_nns(w, n, search_k, result, distances);
  }

  S get_n_items() const {
    return _n_items;
  }

  S get_n_trees() const {
    return (S)_roots.size();
  }

  void verbose(bool v) {
    _verbose = v;
  }

  void get_item(S item, T* v) const {
    // TODO: handle OOB
    Node* m = _get(item);
    memcpy(v, m->v, (_f) * sizeof(T));
  }

  void set_seed(R seed) {
    _seed = seed;
  }

  



  void thread_build(int q, int thread_idx, ThreadedBuildPolicy& threaded_build_policy) {
    // Each thread needs its own seed, otherwise each thread would be building the same tree(s)
    Random _random(_seed + thread_idx);

    // S: int
    vector<S> thread_roots;


    // int tree_id = 0;
    int tree_count = 0;
    while (1) {

      if (q == -1) {
       
        threaded_build_policy.lock_n_nodes();
        
        if (_n_nodes >= 2 * _n_items) {
          threaded_build_policy.unlock_n_nodes();
          break;
        }
        threaded_build_policy.unlock_n_nodes();
      } 
      else {

        if (thread_roots.size() >= (size_t)q) {
          break;
        }
      }

      if (_verbose) annoylib_showUpdate("pass %zd...\n", thread_roots.size());


      // vector<S> indices;
      S *indices = new S[_n_items];

      threaded_build_policy.lock_shared_nodes();

      for (S i = 0; i < _n_items; i++) {
       
        if (_get(i)->n_descendants >= 1) { // Issue #223
          // indices.push_back(i);
          indices[i] = i;
        }
      }


 
      threaded_build_policy.unlock_shared_nodes();


      thread_roots.push_back(_make_tree(indices));
      tree_count++;
      printf("n trees built: %d / %d\n", tree_count + 1, q);
    }


    threaded_build_policy.lock_roots();
    _roots.insert(_roots.end(), thread_roots.begin(), thread_roots.end());
    threaded_build_policy.unlock_roots();
  }


  void _reallocate_nodes(S n) {

    const double reallocation_factor = 1.3;
    S new_nodes_size = std::max(n, (S) ((_nodes_size + 1) * reallocation_factor));
    // void *old = _nodes;
    
    if (_on_disk) {
      if (!remap_memory_and_truncate(&_nodes, _fd, 
                    static_cast<size_t>(_s) * static_cast<size_t>(_nodes_size), 
                    static_cast<size_t>(_s) * static_cast<size_t>(new_nodes_size)) && _verbose)
          annoylib_showUpdate("File truncation error\n");
    } 
    else {
      _nodes = realloc(_nodes, _s * new_nodes_size);
      memset((char *) _nodes + (_nodes_size * _s) / sizeof(char), 0, (new_nodes_size - _nodes_size) * _s);
    }
    
    _nodes_size = new_nodes_size;

    // if (_verbose) annoylib_showUpdate("Reallocating to %d nodes: old_address=%p, new_address=%p\n", new_nodes_size, old, _nodes);
  }


  void _allocate_size(S n, ThreadedBuildPolicy& threaded_build_policy) {
    if (n > _nodes_size) {
      // threaded_build_policy.lock_nodes();
      _reallocate_nodes(n);
      // threaded_build_policy.unlock_nodes();
    }
  }

  void _allocate_size(S n) {
    if (n > _nodes_size) {
      _reallocate_nodes(n);
    }
  }



  Node* _get(const S i) const {
    return get_node_ptr<S, Node>(_nodes, _s, i);
  }



  double _split_imbalance(int left_sz, int right_sz) {
    double ls = (float)left_sz;
    double rs = (float)right_sz;
    float f = ls / (ls + rs + 1e-9);  // Avoid 0/0
    return std::max(f, 1-f);
  }


  double _split_imbalance(const vector<S>& left_indices, const vector<S>& right_indices) {
    double ls = (float)left_indices.size();
    double rs = (float)right_indices.size();
    float f = ls / (ls + rs + 1e-9);  // Avoid 0/0
    return std::max(f, 1-f);
  }



  virtual S _make_tree(S *indices) = 0;


  void _get_all_nns(const T* v, size_t n, int search_k, vector<S>* result,
                                         vector<T>* distances) const {

    
    Node* v_node = (Node *)alloca(_s);
    D::template zero_value<Node>(v_node); // no effect.

    memcpy(v_node->v, v, sizeof(T) * _f);
    D::init_node(v_node, _f);

    std::priority_queue<pair<T, S> > q;

    if (search_k == -1) {
      search_k = n * _roots.size();
    }

    for (size_t i = 0; i < _roots.size(); i++) {
      // pq_initial_value() returns numeric_limits<T>::infinity().
      q.push(make_pair(Distance::template pq_initial_value<T>(), _roots[i]));
    }


    
    std::vector<S> nns;

    
    while (nns.size() < (size_t)search_k && !q.empty()) {
      
      const pair<T, S>& top = q.top();
      T d = top.first;
      S i = top.second;
      Node* nd = _get(i);
      q.pop();

      if (nd->n_descendants == 1 && i < _n_items) {
        nns.push_back(i);
      } 
      else if (nd->n_descendants <= _K) {
        const S* dst = nd->children;
        nns.insert(nns.end(), dst, &dst[nd->n_descendants]);
      } 
      else {
        
        T margin = D::margin(nd, v, _f); // dot of nd->v and v.

        // d is used by priority queue to sort. Larger ones are close to front.
        q.push(make_pair(D::pq_distance(d, margin, 1), static_cast<S>(nd->children[1])));
        q.push(make_pair(D::pq_distance(d, margin, 0), static_cast<S>(nd->children[0])));
      }
    }




    // Get distances for all items
    // To avoid calculating distance multiple times for any items, sort by id
    std::sort(nns.begin(), nns.end());
    vector<pair<T, S> > nns_dist;
    S last = -1;

    for (size_t i = 0; i < nns.size(); i++) {

      S j = nns[i]; 
      if (j == last) continue;

      last = j;

      if (_get(j)->n_descendants == 1)  // This is only to guard a really obscure case, #284
        nns_dist.push_back(make_pair(D::distance(v_node, _get(j), _f), j));
    }



    size_t m = nns_dist.size();
    size_t p = n < m ? n : m; // Return this many items
    std::partial_sort(nns_dist.begin(), nns_dist.begin() + p, nns_dist.end());

    for (size_t i = 0; i < p; i++) {

      if (distances) // no.
        distances->push_back(D::normalized_distance(nns_dist[i].first));

      result->push_back(nns_dist[i].second);
    }
  }


  
};



class AnnoyIndexSingleThreadedBuildPolicy {
public:

  template<typename S, typename T, typename D, typename Random>
  static void build(AnnoyIndex<S, T, D, Random, AnnoyIndexSingleThreadedBuildPolicy>* annoy, int q, int n_threads) {

    AnnoyIndexSingleThreadedBuildPolicy threaded_build_policy;
    annoy->thread_build(q, 0, threaded_build_policy);

  }



  void lock_n_nodes() {}
  void unlock_n_nodes() {}

  void lock_nodes() {}
  void unlock_nodes() {}

  void lock_shared_nodes() {}
  void unlock_shared_nodes() {}

  void lock_roots() {}
  void unlock_roots() {}
};




typedef unsigned char BYTE;
typedef unsigned int WORD;

  // copy_vec<T>(p, vecArray[indexArray[i] * f], f);

template<typename T>
__device__ 
void copy_vec(T *vec_dst, T *vec_src, int f){
  for(int i = 0; i < f; i++){
    vec_dst[i] = vec_src[i];
  }
}


// template<typename T>
// __device__ __host__
// T dot(const T* x, const T* y, int f) {
//   T s = 0;
//   for (int z = 0; z < f; z++) {
//     s += (*x) * (*y);
//     x++;
//     y++;
//     // s += x[z] * y[z];
//   }
//   return s;
// }


template<typename T>
__device__
T get_vec_norm(T* v, int f) {
  return sqrt(dot(v, v, f));
}


template<typename T>
__device__
void normalize(T* vec, int f) {
  T norm = get_vec_norm<T>(vec, f);
  if (norm > 0) {
    for (int z = 0; z < f; z++)
      vec[z] /= norm;
  }
}


// distance(p, vecArray + indexArray[k] * f, f);

template<typename T>
__device__
T distance(T *vec1, T *vec2, int f) {

  T pp = dot(vec1, vec1, f); 
  T qq = dot(vec2, vec2, f);
  T pq = dot(vec1, vec2, f);
  T ppqq = pp * qq;
  if (ppqq > 0) return 2.0 - 2.0 * pq / sqrt(ppqq);
  else return 2.0; // cos is 0
}




template<typename S, typename T, typename Random>
__device__
void two_means(S *indexArray, int sz, T *vecArray, int f, 
                        Random& random, bool cosine, T* p, T* q) {

  static int iteration_steps = 200;
  size_t count = sz;

  size_t i = random.index(count);
  size_t j = random.index(count-1);
  j += (j >= i); // ensure that i != j

  
  copy_vec<T>(p, vecArray + indexArray[i] * f, f);
  copy_vec<T>(q, vecArray + indexArray[j] * f, f);

  if (cosine) { // yes
    normalize<T>(p, f); 
    normalize<T>(q, f);
  }

  int ic = 1, jc = 1;

  for (int l = 0; l < iteration_steps; l++) {
   
    size_t k = random.index(count);
 
    T di = ic * distance(p, vecArray + indexArray[k] * f, f);
    T dj = jc * distance(q, vecArray + indexArray[k] * f, f);
  
    T norm = cosine ? get_vec_norm(vecArray + indexArray[k] * f, f) : 1;  // cosine == true
    
    if (!(norm > T(0))) {
      continue;
    }


    if (di < dj) {
      
      for (int z = 0; z < f; z++)
        p[z] = (p[z] * ic + vecArray[indexArray[k] * f + z] / norm) / (ic + 1);
     
      ic++;
    } 
    else if (dj < di) {
      
      for (int z = 0; z < f; z++)
        q[z] = (q[z] * jc + vecArray[indexArray[k] * f + z] / norm) / (jc + 1);
      
      jc++;
    }
  }

}


template<typename S, typename T, typename Random>
__device__
void create_split(S *indexArray, int sz, T *vecArray, 
              int f, Random& random, T *p, T *q, T* splitVec) {

  two_means<S, T, Random>(indexArray, sz, vecArray, f, random, true, p, q);

  for (int z = 0; z < f; z++)
    splitVec[z] = p[z] - q[z];

  normalize<T>(splitVec, f);
}


__device__
double _split_imbalance(int left_sz, int right_sz) {
  double ls = (float)left_sz;
  double rs = (float)right_sz;
  float f = ls / (ls + rs + 1e-9);  // Avoid 0/0
  // return std::max(f, 1-f);
  return max(f, 1-f);
}


template<typename T>
__device__
void swap(T *arr, int id1, int id2){
  T tmp = arr[id1];
  arr[id1] = arr[id2];
  arr[id2] = tmp;
}

template<typename S>
__device__
void group_moveSide(S* indexArray, int *sideArray, int sz){

  int n_right = 0 ,n_left = 0;
  while(n_left + n_right < sz){
    
    if(sideArray[n_left] == 1){ // right
      swap<int>(sideArray, n_left, sz - 1 - n_right);
      swap<S>(indexArray, n_left, sz - 1 - n_right);
      n_right++;
    }
    else{
      n_left++;
    }
  }
}



__device__
void group_getSideCount(int *sideArray, int sz, int *n_left, int *n_right){

  *n_left = 0, *n_right = 0;
  
  for(int i = 0; i < sz; i++){
    if(sideArray[i] == 1){
      (*n_right)++;
    }
    else{
      (*n_left)++;
    }
  }  
}


struct Group{
  int pos, sz;
};



template<typename S, typename T, typename Random>
__global__ void kernel_split(S *indexArray, T *vecArray, int f, S K, Group *groupArray, 
                  Group *groupArray_next, int n_group, T *splitVecArray,
                  int *sideArray, int *sideCountArray){


  int bid_x = blockIdx.x; 
  if(groupArray[bid_x].sz < K){

    groupArray_next[2 * bid_x + 0].pos = groupArray[bid_x].pos;
    groupArray_next[2 * bid_x + 0].sz = groupArray[bid_x].sz;

    groupArray_next[2 * bid_x + 1].pos = groupArray[bid_x].pos;
    groupArray_next[2 * bid_x + 1].sz = groupArray[bid_x].sz;

    sideCountArray[2 * bid_x + 0] = -1;
    sideCountArray[2 * bid_x + 1] = -1;

    return;
  } 

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  int offset = groupArray[bid_x].pos;
  int sz = groupArray[bid_x].sz;

  Random _random(Random::default_seed + gid + n_group);

  S *indexArray_local = indexArray + offset;
  int *sideArray_local = sideArray + offset;
  // T *vecArray_local = vecArray + f * offset;

  extern __shared__ T sm[];
  T *splitVec_sm = sm;
  T *p_sm = sm + f;
  T *q_sm = sm + 2 * f;

  int *isBalanced_sm = (int *)(sm + 3 * f);
  int *nLeft_sm = isBalanced_sm + 1;
  int *nRight_sm = nLeft_sm + 1;
  


  for (int attempt = 0; attempt < 3; attempt++){

    *isBalanced_sm = 0;
  
    if(tid == 0){
      create_split(indexArray_local, sz, vecArray, f, _random, p_sm, q_sm, splitVec_sm);
    }    

    __syncthreads();

    

    int idx = tid;

    while(idx < sz){

      S item = indexArray_local[idx];
      T dot = 0.;

      for(int i = 0; i < f; i++){
        
        dot += (vecArray + item * f)[i] * splitVec_sm[i];
      }

      if(dot != 0) sideArray_local[idx] = (int)(dot > 0);
      else sideArray_local[idx] = _random.flip();
      
      idx += blockDim.x;
    }

    __syncthreads();


    if(tid == 0){

      group_getSideCount(sideArray_local, sz, nLeft_sm, nRight_sm);

      if (_split_imbalance(*nLeft_sm, *nRight_sm) < 0.95) {
        *isBalanced_sm = 1;
      }      
    } 

    __syncthreads();

    if(*isBalanced_sm) {

      if(tid == 0){
        group_moveSide(indexArray_local, sideArray_local, sz);                
      }
      __syncthreads();
      
      break;
    }
  }


  if(tid == 0){

    if(_split_imbalance(*nLeft_sm, *nRight_sm) > 0.99){

      for (int z = 0; z < f; z++){
        splitVec_sm[z] = 0;
      }    

      while (_split_imbalance(*nLeft_sm, *nRight_sm) > 0.99) {

        for(int j = 0; j < sz; j++){
          sideArray_local[j] = _random.flip();
        }

        group_getSideCount(sideArray_local, sz, nLeft_sm, nRight_sm);
      }

      group_moveSide(indexArray_local, sideArray_local, sz);
    }




    // returns

    groupArray_next[2 * bid_x].pos = offset;
    groupArray_next[2 * bid_x].sz = *nLeft_sm;

    groupArray_next[2 * bid_x + 1].pos = offset + *nLeft_sm;
    groupArray_next[2 * bid_x + 1].sz = *nRight_sm;

    sideCountArray[2 * bid_x] = *nLeft_sm;
    sideCountArray[2 * bid_x + 1] = *nRight_sm;

    for (int z = 0; z < f; z++){
      splitVecArray[f * bid_x + z] = splitVec_sm[z];
    }    
    
  }
}




template<typename S, typename T, typename Distance, typename Random, class ThreadedBuildPolicy>
class AnnoyIndex_GPU: public AnnoyIndex<S, T, Distance, Random, ThreadedBuildPolicy>{

public:

  // typedef Distance D;
  // typedef typename D::template Node<S, T> Node;

  typedef Distance D;
  typedef typename D::template Node<S, T> Node;
  // using typename AnnoyIndex<S, T, Distance, Random, ThreadedBuildPolicy>::D;
  // using typename AnnoyIndex<S, T, Distance, Random, ThreadedBuildPolicy>::Node;
  using typename AnnoyIndex<S, T, Distance, Random, ThreadedBuildPolicy>::R;

  using AnnoyIndex<S, T, Distance, Random, ThreadedBuildPolicy>::_f;
  using AnnoyIndex<S, T, Distance, Random, ThreadedBuildPolicy>::_s; 
  using AnnoyIndex<S, T, Distance, Random, ThreadedBuildPolicy>::_n_items;
  using AnnoyIndex<S, T, Distance, Random, ThreadedBuildPolicy>::_nodes; 
  using AnnoyIndex<S, T, Distance, Random, ThreadedBuildPolicy>::_n_nodes;
  using AnnoyIndex<S, T, Distance, Random, ThreadedBuildPolicy>::_nodes_size;
  using AnnoyIndex<S, T, Distance, Random, ThreadedBuildPolicy>::_roots;
  using AnnoyIndex<S, T, Distance, Random, ThreadedBuildPolicy>::_K;
  using AnnoyIndex<S, T, Distance, Random, ThreadedBuildPolicy>::_seed;
  using AnnoyIndex<S, T, Distance, Random, ThreadedBuildPolicy>::_loaded;
  using AnnoyIndex<S, T, Distance, Random, ThreadedBuildPolicy>::_verbose;
  using AnnoyIndex<S, T, Distance, Random, ThreadedBuildPolicy>::_fd;
  using AnnoyIndex<S, T, Distance, Random, ThreadedBuildPolicy>::_on_disk;
  using AnnoyIndex<S, T, Distance, Random, ThreadedBuildPolicy>::_built;
  Random _random;





  AnnoyIndex_GPU(int f): AnnoyIndex<S, T, Distance, Random, ThreadedBuildPolicy>(f){
    // Random _random(_seed + thread_idx);
  }



  void find_split(vector<S>& indices, int offset, int sz, T *splitVec){

    vector<Node *> children;
    for (size_t i = offset; i < offset + sz; i++) {
      S j = indices[i];
      Node* n = this->_get(j);
      if (n) children.push_back(n);
    }

    Node *m = (Node*)alloca(_s);
    D::create_split(children, _f, _s, _random, m); 
    memcpy(splitVec, m->v, sizeof(T) * _f);
  }





  void hostPreKernel_split(S *indexArray, S *indexArray_dev, T *vecArray_dev, 
                                      int *sideArray_dev, Group *groupArray_dev){


    T *vecArray_host;
    cudaMalloc(&vecArray_dev, _n_items * _f * sizeof(T));
    vecArray_host = new T[_n_items * _f];
    int offset_batch = 0;
 
    for(int i = 0; i < _n_items; i++){

      Node *node = this->_get(indexArray[i]);
      
      for(int z = 0; z < _f; z++){
        vecArray_host[z] = node->v[z];
      }
    }
    cudaMemcpy((BYTE *)vecArray_dev, (BYTE *)vecArray_host, 
                    _n_items * _f * sizeof(T), cudaMemcpyHostToDevice);

    delete [] vecArray_host;

    // --------------------------------

    cudaMalloc(&indexArray_dev, _n_items * sizeof(S));
    cudaMemcpy((BYTE *)indexArray_dev, (BYTE *)indexArray, 
                _n_items * sizeof(S), cudaMemcpyHostToDevice);

    // --------------------------------

    cudaMalloc(&sideArray_dev, _n_items * sizeof(int));  

    // --------------------------------

    cudaMalloc(&groupArray_dev, 1 * sizeof(Group));
    Group groupArray[1];
    groupArray[0].pos = 0;
    groupArray[0].sz = _n_items;
    cudaMemcpy((BYTE *)groupArray_dev, (BYTE *)groupArray, 
                1 * sizeof(Group), cudaMemcpyHostToDevice);
  }



  void launchKernel_split(S *indexArray_dev, T* vecArray_dev, Group *&groupArray_dev, 
                            int n_group, T *&splitVecArray, int *sideArray_dev, 
                              int *&sideCountArray){

    T *splitVecArray_dev;
    cudaMalloc(&splitVecArray_dev, n_group * _f * sizeof(T));

    Group *groupArray_next_dev;
    cudaMalloc(&groupArray_next_dev, n_group * 2 * sizeof(Group));
    
    int *sideCountArray_dev;
    cudaMalloc(&sideCountArray_dev, n_group * 2 * sizeof(int));



    int n_blocks = n_group;
    int n_threads_per_block = 128;

    kernel_split<S, T, Random><<<n_blocks, n_threads_per_block, 
            3 * _f * sizeof(T) + 3 * sizeof(int)>>>(indexArray_dev, vecArray_dev,
             _f, _K, groupArray_dev, groupArray_next_dev,
              n_group, splitVecArray_dev, sideArray_dev, sideCountArray_dev);


    cudaMemcpy((BYTE *)splitVecArray, (BYTE *)splitVecArray_dev, 
                    n_group * _f * sizeof(T), cudaMemcpyDeviceToHost);

    cudaMemcpy((BYTE *)sideCountArray, (BYTE *)sideCountArray_dev, 
                    n_group * 2 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(groupArray_dev);
    cudaFree(splitVecArray_dev);
    cudaFree(sideCountArray_dev);

    groupArray_dev = groupArray_next_dev;
    
  }



  void hostPostKernel_split(S **&childPtrArray, T *splitVecArray, int *sideCountArray, 
                                        int n_group, bool& done){

    S **childPtrArray_next = new S*[4 * n_group];
    int done_count = 0;

    for(int group_i = 0; group_i < n_group; group_i++){

      for(int i = 0; i < 2; i++){

        // a leaf - Don't need to create group.
        if (sideCountArray[2 * group_i + i] <= (size_t)_K) {
          
          done_count++;

          childPtrArray_next[4 * group_i + 2 * i + 0] = childPtrArray[2 * group_i + i];
          childPtrArray_next[4 * group_i + 2 * i + 1] = childPtrArray[2 * group_i + i];

          continue;
        }


        // Need to create group.
        this->_allocate_size(_n_nodes + 1);
        S item = _n_nodes++;
        Node* m = this->_get(item);

        *(childPtrArray[2 * group_i + i]) = item; // children

        m->n_descendants = sideCountArray[2 * group_i + i]; // n_descendants
        
        childPtrArray_next[4 * group_i + 2 * i + 0] = &(m->children[0]);
        childPtrArray_next[4 * group_i + 2 * i + 1] = &(m->children[1]);        
      }


      Node *p = (Node *)(((BYTE *)childPtrArray[2 * group_i]) - offsetof(Node, children));
      if(sideCountArray[2 * group_i] >= 0){ 
        memcpy(p->v, splitVecArray + group_i * _f, _f * sizeof(T)); // v
      }

    }


    done = false;
    if(done_count == 2 * n_group){
      done = true;
      delete [] childPtrArray_next;  
      return;
    } 
    
    delete [] childPtrArray;
    childPtrArray = childPtrArray_next;
  }



  void copy_leaf_node(S *indexArray, Group *groupArray, S **childPtrArray, int n_group){

    for(int i = 0; i < n_group - 1;){
      
      while(childPtrArray[i + 1] == childPtrArray[i]) i++;

      int offset = groupArray[i].pos;
      int sz = groupArray[i].sz;

      if(sz == 1){
        *(childPtrArray[i]) = indexArray[offset];
        continue;
      }
      else{

        this->_allocate_size(_n_nodes + 1);
        S item = _n_nodes++;
        Node* m = this->_get(item);
        m->n_descendants = (S)sz;
        memcpy(m->children, &indexArray[offset], sz * sizeof(S));        
      }
    }
  }
                                                        


                                                        
  S _make_tree(S *indexArray) {

    this->_allocate_size(_n_nodes + 1);
    S item = _n_nodes++;
    Node* m = this->_get(item); 
    m->n_descendants = _n_items; 
    m->children[0] = 0;



    if (_n_items <= (size_t)_K && 
          ((size_t)_n_items <= (size_t)_K || _n_items == 1)) {
      
      if (_n_items != 0){
        memcpy(m->children, indexArray, _n_items * sizeof(S));
      }

      return item;
    }


    T *vecArray_dev;
    S *indexArray_dev;
    int *sideArray_dev;
    Group *groupArray_dev;
    hostPreKernel_split(indexArray, indexArray_dev, vecArray_dev, sideArray_dev, 
                              groupArray_dev);

    
    bool done = false;
    int n_group = 1;
    
    S **childPtrArray = new S*[2 * n_group];
    
    
    T *splitVecArray;  
    int *sideCountArray;
    
    while(1){

      splitVecArray = new T[_f * n_group];
      sideCountArray = new int[2 * n_group];

      launchKernel_split(indexArray_dev, vecArray_dev, groupArray_dev, n_group, 
                              splitVecArray, sideArray_dev, sideCountArray); 

      hostPostKernel_split(childPtrArray, splitVecArray, sideCountArray, n_group, done);

      delete [] splitVecArray;
      delete [] sideCountArray;

      if(!done) n_group = n_group * 2;
      else break;
    }

    cudaMemcpy((BYTE *)indexArray, (BYTE *)indexArray_dev, 
                _n_items * sizeof(S), cudaMemcpyDeviceToHost);

    Group *groupArray;
    cudaMemcpy((BYTE *)groupArray, (BYTE *)groupArray_dev, 
                n_group * sizeof(Group), cudaMemcpyDeviceToHost);

    copy_leaf_node(indexArray, groupArray, childPtrArray, n_group);


    cudaFree(vecArray_dev);
    cudaFree(indexArray_dev);
    cudaFree(sideArray_dev);
    cudaFree(groupArray_dev);
    
    return item;
  }
};



}

#endif
// vim: tabstop=2 shiftwidth=2

