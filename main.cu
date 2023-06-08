
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include "kissrandom.h"
#include "annoylib.h"
#include <chrono>
#include <algorithm>
#include <map>
#include <random>


using namespace Annoy;


int fill_item(char *filename, int f=40, int n=1000000, int n_trees=80){

	AnnoyIndex_GPU<int, float, Angular, Kiss32Random> t(f);

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

		std::cout << "Loading objects ...\t object: "
				  << i+1 
				  << "\tProgress:"
				  << std::fixed 
				  << std::setprecision(2) 
				  << (float) i / (float)(n + 1) * 100 
				  << "%\r";			  
	}
}


void load_item(AnnoyIndex_GPU<int, float, Angular, Kiss32Random>& t, char *filename, int n){
	
	// t.load(filename);
	t.load_items(filename, n);

}


void build_index(AnnoyIndex_GPU<int, float, Angular, Kiss32Random>& t, int n_trees){


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



int precision_test(AnnoyIndex_GPU<int, float, Angular, Kiss32Random>& t, 
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



int precision(int f=40, int n=1000000, int n_trees=80){

	std::chrono::high_resolution_clock::time_point t_start, t_end;

	std::default_random_engine generator;

	AnnoyIndex_GPU<int, float, Angular, Kiss32Random> t(f);


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

			//search_k defaults to "n_trees * (*limit)" (which is  << n) if not provided (pass -1).
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



// f = 786;
// n = 100000;
// n_trees = 5;
// >>>
//  Done in 13 secs.
// limit: 10       precision: 11.00%       avg. time: 0.000160s
// limit: 100      precision: 11.00%       avg. time: 0.000240s
// limit: 1000     precision: 19.00%       avg. time: 0.001910s
// limit: 10000    precision: 67.00%       avg. time: 0.017350s


int main(int argc, char **argv) {


	int f, n, n_trees;


	f = 786;
	n = 100000;
	n_trees = 5;

	

	// precision(f, n, n_trees);



	// fill_item("AnnoyGPU.tree", f, n, n_trees);
	
	AnnoyIndex_GPU<int, float, Angular, Kiss32Random> t(f);
	load_item(t, "AnnoyGPU.tree", n);

	build_index(t, n_trees);
	precision_test(t, f, n, n_trees);

	return EXIT_SUCCESS;
}