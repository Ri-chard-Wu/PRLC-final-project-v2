
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include "kissrandom.h"
#include "annoylib.h"
#include <chrono>
#include <algorithm>
#include <map>
#include <random>




int f = 786;
int n = 1000000;
int n_trees = 5;
char *filename = "AnnoyGPU-1e6.tree";

int search_multiplier = 20;
int GPU_BUILD_MAX_ITEM_NUM = 500000;






using namespace Annoy;


int fill_item(char *filename, int f=40, int n=1000000){

	AnnoyIndex<int, float, Angular, Kiss32Random, AnnoyIndexGPUBuildPolicy> t(f);

	std::default_random_engine generator;

	t.on_disk_build(filename);
	
	for(int i = 0; i < n; ++i){ // n: number of vectors

		float *vec = (float *) malloc( f * sizeof(float) );

		float mean = 0.0;
		float std = 1.0;		
		std::normal_distribution<float> distribution(mean, std);
		
		for(int z = 0; z < f; ++z){ // f: vector dim.
			vec[z] = (distribution(generator));
		}

		t.add_item(i, vec);

		if(i % 1024 == 0){
			std::cout << "Loading objects ...\t object: "
					<< i+1 
					<< "\tProgress:"
					<< std::fixed 
					<< std::setprecision(2) 
					<< (float) i / (float)(n + 1) * 100 
					<< "%\r";		
		}	  
	}
}


void load_item(AnnoyIndex<int, float, Angular, Kiss32Random, AnnoyIndexGPUBuildPolicy>& t, char *filename, int n){
	
	// t.load(filename);
	t.load_items(filename, n);

}


void build_index(AnnoyIndex<int, float, Angular, Kiss32Random, AnnoyIndexGPUBuildPolicy>& t, int n_trees){


	std::chrono::high_resolution_clock::time_point t_start, t_end;


	std::cout << std::endl;
	// std::cout << "Building index num_trees = 2 * num_features ...\n\n\n";

	t_start = std::chrono::high_resolution_clock::now();
	t.build(n_trees);
	t_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::\
				duration_cast<std::chrono::seconds>( t_end - t_start ).count();
	std::cout << " Done in "<< duration << " secs." << std::endl;

	// std::cout << "Saving index ...";
	// t.save("precision.tree");
	// std::cout << " Done" << std::endl;
}



int precision_test(AnnoyIndex<int, float, Angular, Kiss32Random, AnnoyIndexGPUBuildPolicy>& t, 
			int f=40, int n=1000000, int n_trees=80){

	std::chrono::high_resolution_clock::time_point t_start, t_end;

	std::vector<int> limits = {10, 100, 1000, 10000};
	int K=10;
	int prec_n = 10;

	std::map<int, double> prec_sum;
	std::map<int, double> time_sum;
	std::vector<int> closest;

	for(std::vector<int>::iterator it = limits.begin(); 
									it != limits.end(); ++it){
		prec_sum[(*it)] = 0.0;
		time_sum[(*it)] = 0.0;
	}


	for(int i = 0; i < prec_n; ++i){

		int j = rand() % n;

		t.get_nns_by_item(j, K, n, &closest, nullptr);

		std::vector<int> toplist;
		std::vector<int> intersection;

		for(std::vector<int>::iterator limit = limits.begin(); 
										limit!=limits.end(); ++limit){

			t_start = std::chrono::high_resolution_clock::now();

			//search_k defaults to "n_trees * (*limit)" (which is  << n) if not provided (pass -1).
			t.get_nns_by_item(j, (*limit), search_multiplier * n_trees * (*limit), &toplist, nullptr); 

			
			t_end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast\
					<std::chrono::milliseconds>( t_end - t_start ).count();

			std::sort(closest.begin(), closest.end(), std::less<int>());
			std::sort(toplist.begin(), toplist.end(), std::less<int>());

			intersection.resize(std::max(closest.size(), toplist.size()));
			
			std::vector<int>::iterator it_set = \
				std::set_intersection(closest.begin(), closest.end(), \
					toplist.begin(), toplist.end(), intersection.begin());

			intersection.resize(it_set-intersection.begin());

			int found = intersection.size();
			double hitrate = found / (double) K;
			prec_sum[(*limit)] += hitrate;
			time_sum[(*limit)] += duration;
 
			vector<int>().swap(intersection);
			vector<int>().swap(toplist);
		}


		closest.clear(); 
		vector<int>().swap(closest);
	}

	for(std::vector<int>::iterator limit = limits.begin(); limit!=limits.end(); ++limit){
		std::cout << "limit: " << (*limit) << "\tprecision: "<< std::fixed 
			<< std::setprecision(2) << (100.0 * prec_sum[(*limit)] / prec_n)
					<< "% \tavg. time: "<< std::fixed<< std::setprecision(6) 
					<< (time_sum[(*limit)] / prec_n) * 1e-04 << "s" << std::endl;
	}


	std::cout << "\nDone" << std::endl;
	return 0;
}



int precision(int f=40, int n=1000000, int n_trees=80){

	std::chrono::high_resolution_clock::time_point t_start, t_end;

	std::default_random_engine generator;

	AnnoyIndex<int, float, Angular, Kiss32Random, AnnoyIndexGPUBuildPolicy> t(f);


	char *filename = "test_disk_build.tree";
	t.on_disk_build(filename);

	// std::cout << "Building index ..." << std::endl;


	
	for(int i = 0; i < n; ++i){ // n: number of vectors

		float *vec = (float *) malloc( f * sizeof(float) );

		// double mean = (double)(rand() % 10);
		// double std = (double)(rand() % 5);
		float mean = 0.0;
		float std = 1.0;		
		std::normal_distribution<float> distribution(mean, std);
		
		for(int z = 0; z < f; ++z){ // f: vector dim.

			vec[z] = (distribution(generator));
		}

		t.add_item(i, vec);

		// std::cout << "Loading objects ...\t object: "
		// 		  << i+1 
		// 		  << "\tProgress:"
		// 		  << std::fixed 
		// 		  << std::setprecision(2) 
		// 		  << (float) i / (float)(n + 1) * 100 
		// 		  << "%\r";
		
						  
	}


	std::cout << std::endl;
	// std::cout << "Building index num_trees = 2 * num_features ...\n\n\n";

	t_start = std::chrono::high_resolution_clock::now();
	t.build(n_trees);
	t_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>( t_end - t_start ).count();
	std::cout << " Done in "<< duration << " secs." << std::endl;

	// std::cout << "Saving index ...";
	// t.save("precision.tree");
	// std::cout << " Done" << std::endl;


	//******************************************************


	std::vector<int> limits = {10, 100, 1000, 10000};
	int K=10;
	int prec_n = 10;

	std::map<int, double> prec_sum;
	std::map<int, double> time_sum;
	std::vector<int> closest;

	//init precision and timers map
	for(std::vector<int>::iterator it = limits.begin(); 
									it != limits.end(); ++it){
		prec_sum[(*it)] = 0.0;
		time_sum[(*it)] = 0.0;
	}


	// test precision with `prec_n` random number.
	for(int i = 0; i < prec_n; ++i){

		// select a random node
		int j = rand() % n;

		// std::cout << "finding nbs for " << j << std::endl;

		// getting the K closest
		// search all n nodes, very slow but most accurate achievable.
		t.get_nns_by_item(j, K, n, &closest, nullptr);

		std::vector<int> toplist;
		std::vector<int> intersection;

		for(std::vector<int>::iterator limit = limits.begin(); 
										limit!=limits.end(); ++limit){

			t_start = std::chrono::high_resolution_clock::now();

			//search_k defaults to "n_trees * (*limit)" 
				// (which is  << n) if not provided (pass -1).
			t.get_nns_by_item(j, (*limit), (size_t) -1, &toplist, nullptr); 

			
			t_end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast\
					<std::chrono::milliseconds>( t_end - t_start ).count();

			std::sort(closest.begin(), closest.end(), std::less<int>());
			std::sort(toplist.begin(), toplist.end(), std::less<int>());

			intersection.resize(std::max(closest.size(), toplist.size()));
			
			std::vector<int>::iterator it_set = \
				std::set_intersection(closest.begin(), closest.end(), \
					toplist.begin(), toplist.end(), intersection.begin());

			intersection.resize(it_set-intersection.begin());

			int found = intersection.size();
			double hitrate = found / (double) K;
			prec_sum[(*limit)] += hitrate;
			time_sum[(*limit)] += duration;
 
			vector<int>().swap(intersection);
			vector<int>().swap(toplist);
		}


		closest.clear(); 
		vector<int>().swap(closest);
	}

	for(std::vector<int>::iterator limit = limits.begin(); limit!=limits.end(); ++limit){
		std::cout << "limit: " << (*limit) << "\tprecision: "<< std::fixed 
			<< std::setprecision(2) << (100.0 * prec_sum[(*limit)] / prec_n)
					<< "% \tavg. time: "<< std::fixed<< std::setprecision(6) 
					<< (time_sum[(*limit)] / prec_n) * 1e-04 << "s" << std::endl;
	}


	std::cout << "\nDone" << std::endl;
	return 0;
}



// #############################################################
/*
gpu, 1e6, 5, no partition  -------------------------------------------

 Done in 20 secs.
limit: 10       precision: 11.00%       avg. time: 0.000100s
limit: 100      precision: 11.00%       avg. time: 0.000210s
limit: 1000     precision: 12.00%       avg. time: 0.001930s
limit: 10000    precision: 24.00%       avg. time: 0.019520s


gpu, 1e6, 5, partition into 4 -------------------------------------------

 Done in 19 secs.
limit: 10       precision: 4.00%        avg. time: 0.000130s
limit: 100      precision: 5.00%        avg. time: 0.000300s
limit: 1000     precision: 11.00%       avg. time: 0.001580s
limit: 10000    precision: 21.00%       avg. time: 0.016340s

gpu, 1e6, 5, partition into 4, search 5x -------------------------------------------

 Done in 19 secs.
limit: 10       precision: 3.00%        avg. time: 0.000140s
limit: 100      precision: 6.00%        avg. time: 0.000640s
limit: 1000     precision: 15.00%       avg. time: 0.006720s
limit: 10000    precision: 45.00%       avg. time: 0.064340s

gpu, 1e6, 5, partition into 4, search 20x -------------------------------------------

 Done in 19 secs.
limit: 10       precision: 9.00%        avg. time: 0.000330s
limit: 100      precision: 16.00%       avg. time: 0.002510s
limit: 1000     precision: 28.00%       avg. time: 0.026500s
limit: 10000    precision: 100.00%      avg. time: 0.232380s

gpu, 1e6, 5, no partition, search 20x -------------------------------------------

 Done in 20 secs.
limit: 10       precision: 11.00%       avg. time: 0.000360s
limit: 100      precision: 15.00%       avg. time: 0.003350s
limit: 1000     precision: 32.00%       avg. time: 0.031870s
limit: 10000    precision: 100.00%      avg. time: 0.255850s
*/

// #############################################################

/*

cpu, 5e6, 5 -------------------------------------------

 Done in 1070 secs.
limit: 10       precision: 10.00%       avg. time: 0.000160s
limit: 100      precision: 10.00%       avg. time: 0.000210s
limit: 1000     precision: 11.00%       avg. time: 0.002200s
limit: 10000    precision: 17.00%       avg. time: 0.022570s


gpu, 5e6, 5, no partition -------------------------------------

 Done in 125 secs.
limit: 10       precision: 1.00%        avg. time: 0.000180s
limit: 100      precision: 7.00%        avg. time: 0.000700s
limit: 1000     precision: 12.00%       avg. time: 0.006390s
limit: 10000    precision: 23.00%       avg. time: 0.066370s


gpu, 5e6, 5, partition into 5, search 20x -------------------------------------------

 Done in 112 secs.
limit: 10       precision: 2.00%        avg. time: 0.000410s
limit: 100      precision: 11.00%       avg. time: 0.003170s
limit: 1000     precision: 16.00%       avg. time: 0.031250s
limit: 10000    precision: 39.00%       avg. time: 0.309640s

*/







// -------------------------------------------
// gpu, create_split() no parallel, 1e6 nodes, 5 trees: Done in 25 secs.
// gpu, create_split() with parallel, 1e6 nodes, 5 trees: Done in 22 secs.

int main(int argc, char **argv) {


	// fill_item("AnnoyGPU-5e6.tree", f, n);
	
	AnnoyIndex<int, float, Angular, Kiss32Random, AnnoyIndexGPUBuildPolicy> t(f);
	load_item(t, filename, n);

	t.GPU_BUILD_MAX_ITEM_NUM = GPU_BUILD_MAX_ITEM_NUM;

	build_index(t, n_trees);
	precision_test(t, f, n, n_trees);




	return EXIT_SUCCESS;
}