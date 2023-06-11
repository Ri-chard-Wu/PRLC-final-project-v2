

# from annoy import AnnoyIndex
# import random

# f = 768  # Length of item vector that will be indexed

# t = AnnoyIndex(f, 'angular')

# print("adding items...")

# n = 100000
# for i in range(n):
#     if(i % 1000 == 0): print(f"{i} / {n}")
#     v = [random.gauss(0, 1) for z in range(f)]
#     t.add_item(i, v)

# t.build(5) # 10 trees
# t.save('test.ann')

# # ...

# u = AnnoyIndex(f, 'angular')
# u.load('test.ann') # super fast, will just mmap the file
# # print(u.get_nns_by_item(0, 1000)) # will find the 1000 nearest neighbors

# -------------------------------


# from __future__ import print_function
# import random, time

# from annoy import AnnoyIndex



# try:
#     xrange
# except NameError:
#     # Python 3 compat
#     xrange = range


# n, f = 100000, 786

# t = AnnoyIndex(f, 'angular')

# # for i in xrange(n):
# #     v = []
# #     for z in xrange(f):
# #         v.append(random.gauss(0, 1))
# #     t.add_item(i, v)

# t.load_items('test-1e6.tree')


# t.build(5)
# # t.save('test.tree')

# limits = [10, 100, 1000, 10000]
# k = 10
# prec_sum = {}
# prec_n = 10
# time_sum = {}

# for i in xrange(prec_n):
#     j = random.randrange(0, n)
        
#     closest = set(t.get_nns_by_item(j, k, n))
#     for limit in limits:
#         t0 = time.time()
#         toplist = t.get_nns_by_item(j, k, limit)
#         T = time.time() - t0
            
#         found = len(closest.intersection(toplist))
#         hitrate = 1.0 * found / k
#         prec_sum[limit] = prec_sum.get(limit, 0.0) + hitrate
#         time_sum[limit] = time_sum.get(limit, 0.0) + T

# for limit in limits:
#     print('limit: %-9d precision: %6.2f%% avg time: %.6fs'
#           % (limit, 100.0 * prec_sum[limit] / (i + 1),
#              time_sum[limit] / (i + 1)))






from __future__ import print_function
import random, time

from annoy import AnnoyIndex

try:
    xrange
except NameError:
    # Python 3 compat
    xrange = range



f = 786


def fill_items():

    n = 1000000

    t = AnnoyIndex(f, 'angular')
    
    t.fill_items('testPy-1e6.tree')

    for i in xrange(n):

        if(i%1000==0): print(f"{i} / {n}")

        v = []
        for z in xrange(f):
            v.append(random.gauss(0, 1))
        t.add_item(i, v)


    t.save_items()



def precision_test(t):
    limits = [10, 100, 1000, 10000]
    k = 10
    prec_sum = {}
    prec_n = 10
    time_sum = {}

    for i in xrange(prec_n):
        j = random.randrange(0, t.get_n_items())
            
        closest = set(t.get_nns_by_item(j, k, t.get_n_items()))
        for limit in limits:
            t0 = time.time()
            toplist = t.get_nns_by_item(j, k, limit)
            T = time.time() - t0
                
            found = len(closest.intersection(toplist))
            hitrate = 1.0 * found / k
            prec_sum[limit] = prec_sum.get(limit, 0.0) + hitrate
            time_sum[limit] = time_sum.get(limit, 0.0) + T

    for limit in limits:
        print('limit: %-9d precision: %6.2f%% avg time: %.6fs'
            % (limit, 100.0 * prec_sum[limit] / (i + 1),
                time_sum[limit] / (i + 1)))





# fill_items()

t = AnnoyIndex(f, 'angular')
t.load_items('testPy-1e6.tree')
t.build(5)
precision_test(t)
