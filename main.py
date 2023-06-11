

# # ======== CPU build (single thread) ========

# # n trees built: 1 / 5
# # n trees built: 2 / 5
# # n trees built: 3 / 5
# # n trees built: 4 / 5
# # n trees built: 5 / 5

# # Done building in 353 secs.
# # limit: 10        precision:  45.00% avg time: 0.001838s
# # limit: 100       precision:  45.00% avg time: 0.001726s
# # limit: 1000      precision:  58.00% avg time: 0.003799s
# # limit: 10000     precision:  93.00% avg time: 0.025689s


# from annoy import AnnoyIndex
# import numpy as np 
# import os 
# import random, time


# f = 768


# def fill_items():

#     t = AnnoyIndex(f, 'angular')
#     t.fill_items('TBrain-v2.tree')

#     dir = 'TBrain_data'

#     vec_list = []
#     for file in os.listdir(dir):
#         path = os.path.join(dir, file)
#         a = np.load(path)
#         print(path, a.shape)
#         vec_list += list(a)

#     for i, vec in enumerate(vec_list):
        
#         if(i % 1000 == 0): print(f"{i} / {len(vec_list)}")
#         t.add_item(i, list(vec))


#     t.save_items()



# def precision_test(t):

#     limits = [10, 100, 1000, 10000]
#     k = 10
#     prec_sum = {}
#     prec_n = 10
#     time_sum = {}

#     for i in range(prec_n):
#         j = random.randrange(0, t.get_n_items())
            
#         closest = set(t.get_nns_by_item(j, k, t.get_n_items()))
#         for limit in limits:
#             t0 = time.time()
#             toplist = t.get_nns_by_item(j, k, limit)
#             T = time.time() - t0
                
#             found = len(closest.intersection(toplist))
#             hitrate = 1.0 * found / k
#             prec_sum[limit] = prec_sum.get(limit, 0.0) + hitrate
#             time_sum[limit] = time_sum.get(limit, 0.0) + T

#     for limit in limits:
#         print('limit: %-9d precision: %6.2f%% avg time: %.6fs'
#             % (limit, 100.0 * prec_sum[limit] / (i + 1),
#                 time_sum[limit] / (i + 1)))



# # fill_items()

# t = AnnoyIndex(f, 'angular')
# t.load_items('TBrain-v2.tree')
# t.build(5)
# precision_test(t)



# -------------------------------




from __future__ import print_function
import random, time

from annoy import AnnoyIndex

try:
    xrange
except NameError:
    # Python 3 compat
    xrange = range



f = 768


def fill_items():

    n = 1000000

    t = AnnoyIndex(f, 'angular')
    
    t.fill_items('testPy-f768-n1e6.tree')

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
# t.load_items('testPy-f786-1e6.tree')
t.load_items('testPy-f768-n5e5.tree')
t.build(5)
precision_test(t)
