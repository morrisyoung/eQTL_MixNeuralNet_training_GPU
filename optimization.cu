// the main optimization routine

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <string>
#include <vector>
#include "basic.h"
#include <forward_list>
#include <utility>
#include "genotype.h"
#include "expression.h"
#include "optimization.h"
#include "global.h"
#include "main.h"  // typedef struct tuple_long
#include <math.h>       /* exp */
#include "opt_subroutine.h"
#include "opt_para_save.h"
#include "opt_debugger.h"
#include "lib_matrix.h"
#include "opt_hierarchy.h"
// includes CUDA runtime
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

#include <sys/time.h>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */




using namespace std;




//====================================== local global variables ========================================
// these variables are specially designed for this routine -- optimization
// need to initialize some local containers:
array<float *, NUM_CHR> snp_dosage_list;
float * gene_rpkm_exp;  // with length "num_gene"
float * cellenv_hidden_var;  // with length "num_cellenv"
float * batch_var;  // with length "num_batch"
float * batch_hidden_var;  // with length "num_batch_hidden"


// parameter derivative containers:
vector<Matrix_imcomp> cube_para_dev_cis_gene;
Matrix matrix_para_dev_snp_cellenv;
vector<Matrix> cube_para_dev_cellenv_gene;
Matrix matrix_para_dev_batch_batch_hidden;
Matrix matrix_para_dev_batch_hidden_gene;




// GPU global variables (whenever it comes to d_xxx (GPU device memory), it's an array other than matrix)
//matrix_para_dev_snp_cellenv --> float * d_para_dev_snp_cellenv
float * d_para_snp_cellenv;
//vector<Matrix> cube_para_dev_cellenv_gene --> float * d_para_dev_cellenv_gene
vector<float *> d_list_para_cellenv_gene;
//Matrix matrix_para_dev_batch_batch_hidden --> float * d_para_dev_batch_batch_hidden
float * d_para_batch_batch_hidden;
//Matrix matrix_para_dev_batch_hidden_gene --> float * d_para_dev_batch_hidden_gene
float * d_para_batch_hidden_gene;


//matrix_para_dev_snp_cellenv --> float * d_para_dev_snp_cellenv
float * d_para_dev_snp_cellenv;
//vector<Matrix> cube_para_dev_cellenv_gene --> float * d_para_dev_cellenv_gene
vector<float *> d_list_para_dev_cellenv_gene;
//Matrix matrix_para_dev_batch_batch_hidden --> float * d_para_dev_batch_batch_hidden
float * d_para_dev_batch_batch_hidden;
//Matrix matrix_para_dev_batch_hidden_gene --> float * d_para_dev_batch_hidden_gene
float * d_para_dev_batch_hidden_gene;


// temp space for Matrix Multiplication
//matrix_para_dev_snp_cellenv --> float * d_para_dev_snp_cellenv
float * d_temp_snp_cellenv;
//vector<Matrix> cube_para_dev_cellenv_gene --> float * d_para_dev_cellenv_gene
float * d_temp_cellenv_gene;
//Matrix matrix_para_dev_batch_batch_hidden --> float * d_para_dev_batch_batch_hidden
float * d_temp_batch_batch_hidden;
//Matrix matrix_para_dev_batch_hidden_gene --> float * d_para_dev_batch_hidden_gene
float * d_temp_batch_hidden_gene;



// intermediate variables
int * d_etissue_index_p;
float * d_snp;						// real data
float * d_expr;						// real data
float * d_gene_rpkm_exp;  			// with length "num_gene"
float * d_gene_rpkm_exp_cellenv;  	// with length "num_gene"
float * d_gene_rpkm_exp_batch;  	// with length "num_gene"
float * d_error_list;				// with length "num_gene"
float * d_cellenv_hidden_var;  		// with length "num_cellenv"
float * d_batch_var;  				// with length "num_batch"
float * d_batch_hidden_var;  		// with length "num_batch_hidden"






//=====================================================
//********************** hierarchy ********************
//=====================================================
// containers:
vector<Matrix_imcomp> cube_para_cis_gene_parent;
vector<Matrix> cube_para_cellenv_gene_parent;


// the hierarchy:
// what to build for the hierarchy computing (prepared and renewed):
//		[2]. hashing all the leaves to their parents (in order to retrieve the parental variable array)
//		[4]. hashing all the internal nodes to their children and parent (with length to them, or the variance), in order to build the computational matrices
//		[5]. having a bi-directional list for the internal nodes (in order to build and fill in computtional matrices)
//		(6). building the tissue distance list (to its parent) based on etissue_list, to be used by the actual regularization
unordered_map<string, hierarchy_neighbor> hash_leaf_parent;		// --> [2]
unordered_map<string, vector< hierarchy_neighbor >> hash_internode_neighbor;
																// --> [4]
vector<string> internode_list;									// --> [5]
unordered_map<string, int> internode_index_map;					// --> [5]
int num_internode;
vector<float> etissue_dis_par_list;								// --> (6)





// some assistant components:
// the prior number for each un-pruned snp for regularization (from pruned snps and chromatin states); per etissue, per chromosome, for each snp
// TODO: we also still need to integrate distance prior later on with the following prior information
vector<vector<vector<float>>> prior_tissue_vector;
// pairwise phylogenetic distance between etissues
//vector<vector<float>> tissue_hierarchical_pairwise;		--> changed to the new variables above; as we prepare the hierarchy as what we need them to be




// learning control parameters:
int iter_learn_out = 1;  // iteration across all tissues
//int iter_learn_in = 200;  // iteration across all samples from one tissue 			--> (Mar.8, 2016) this is probably too much (we used 3:40 for running only one tissue; maybe 50 is good enough, as the changing speed is much slower)
int iter_learn_in = 50;
int batch_size = 20;  // better be 20												--> (Jan.27) testing mode


// test different learning rate
//float rate_learner = 1.0;  // the learning rate; this doesn't work
//float rate_learner = 0.1;  // the learning rate; this doesn't work
//float rate_learner = 0.01;  // the learning rate; this doesn't work
//float rate_learner = 0.001;  // the learning rate; works!!!; bench#3
float rate_learner = 0.0001;  // the learning rate; works!!!; bench#4
//float rate_learner = 0.00001;  // the learning rate; works!!!; bench#5				--> (Jan.27) the latest one
//float rate_learner = 0.000001;  // the learning rate




//========================================
// the following is the debugging routine
//========================================
/*
Feb.3:
Finally we cleaned all the bugs (if there is no more).
I'm now working on tuning the learn_rate parameter. It looks like large learn_rate will lead to the program go wild:
###########################
try different learning rate parameters:
2(1): all the parameters tend to be quite large, and there appears “Nan” at iter#8 in snp_cellenv
3(0.01): iter#29 “Nan” in snp_cellenv
4(0.0001): no “Nan”, and all the parameter values seem to be normal
5(0.000001): no “Nan”, and all the parameter values seem to be normal
###########################
"0.001" and "0.0001" seem good, yet it looks like they are still wilding the program, with different speeds.
I will try these two parameters with more iterations, to see where do they end up
###########################
I have run 200 iterations (1h12min) for 0.001 and 0.0001, and the results are in workbench#2 (0.001) and workbench#3 (0.0001)
I will run 1000 iterations (6h) for 0.001 and 0.0001, to see how do they converge; they are in workbench#4 and workbench#5
...
...
...
###########################



Feb.4:
something I planned to do, but finally give up as there is no need to do that:
1. add the stochastic module (the current method won't affect too much);
2. add errors to the true parameters (however, even we are in the true parameters, the model seem to diverse, so no meaning to do that);
3. learning several iterations in training set, predict on the testing set (directly predict other than saving the parameters first) (however, as the learning is diverging, this seems not so meaningful)



Feb.8 (Feb.12):
I will test the following this week:
(done) 1. add errors to the parameters (N(0, 1), as all the parameters are drawn from N(0, 1), so errors with that magtitude is acceptable; we can also try other error magtitude later on);
2. tune the prior (sparsity) strength, to see whether we have a better converging trend;
3. output the likelihood for the tensor, with and without errors (on parameters) added, and check its trend; we can do this on either training set or testing set;
4. it's the time now to make the program truly stochastic (for gradient descent)


*/
//======================================================================================================







// TODO: we need to think more about this, say, how to use the data from other epigenomics projects
// load all the cis- snp prior information (tissue specific) from prepared file outside
// fill in the following: vector<vector<vector<float>>> prior_tissue_vector
void opt_snp_prior_load()
{

	for(int i=0; i<num_etissue; i++)
	{
		vector<vector<float>> matrix;
		prior_tissue_vector.push_back(matrix);
	}


	/*
	// TODO: there should always be prior information in the repo
	// TODO: in the simulating data, we don't have this prior, so temporarily stop this

	// get the eTissue-index map
	unordered_map<string, string> index_map;  // for temporary usage
	char filename[100] = "../prior.score.unpruned/prior_tissue_index.txt";
	FILE * file_in = fopen(filename, "r");
	if(file_in == NULL)
	{
		fputs("File error\n", stderr); exit (1);
	}
	int input_length = 1000;
	char input[input_length];
	while(fgets(input, input_length, file_in) != NULL)
	{
		trim(input);

		const char * sep = "\t";
		char * p;
		p = strtok(input, sep);
		string eTissue = p;

		int count = 0;
		while(p)
		{
			count++;
			if(count == 1)  // this is the eTissue
			{
				p = strtok(NULL, sep);
				continue;
			}
			if(count == 2)  // this is the index
			{
				string index = p;
				index_map[eTissue] = index;
				break;
			}
		}
	}
	fclose (file_in);

	// get the prior score for each eTissue, on all chromosomes
	for( auto it = index_map.begin(); it != index_map.end(); ++it )
	{
		string eTissue = it->first;
		string index = it->second;
		vector<vector<float>> vec;
		prior_tissue_rep[eTissue] = vec;

		int i;
		for(i=0; i<NUM_CHR; i++)
		{
			int chr = i+1;
			vector<float> vec;
			prior_tissue_rep[eTissue].push_back(vec);

			//======== get all SNPs with their snp_info (count, position) ========
			char filename[100] = "../prior.score.unpruned/etissue";
			char temp[10];
			StrToCharSeq(temp, index);
			strcat(filename, temp);
			strcat(filename, "/chr");
			sprintf(temp, "%d", chr);
			strcat(filename, temp);
			strcat(filename, ".score");
			//puts("the current file worked on is: ");
			//puts(filename);

			FILE * file_in = fopen(filename, "r");
			if(file_in == NULL)
			{
				fputs("File error\n", stderr); exit (1);
			}

			int input_length = 100;
			char input[input_length];
			while(fgets(input, input_length, file_in) != NULL)
			{
				trim(input);

				float prior = stof(input);
				prior_tissue_rep[eTissue][i].push_back(prior);
			}
			fclose(file_in);
			//======================================
		}
	}
	*/

}





// load the pairwise tissue hierarchy from prepared file outside
// TODO: maybe we should check whether this makes the results better
void opt_tissue_hierarchy_load()
{

	/*	Mar.27: change the hierarchy data structures

	// target: vector<vector<float>> tissue_hierarchical_pairwise;
	// init
	for(int i=0; i<num_etissue; i++)
	{
		vector<float> vec;
		for(int j=0; j<num_etissue; j++)
		{
			vec.push_back(0);
		}
		tissue_hierarchical_pairwise.push_back(vec);
	}

	// load from data source
	char filename[100] = "../tissue_hierarchy_normalized.txt";
	FILE * file_in = fopen(filename, "r");
	if(file_in == NULL)
	{
		fputs("File error\n", stderr); exit (1);
	}
	int input_length = 100000;
	char input[input_length];
	while(fgets(input, input_length, file_in) != NULL)
	{
		trim(input);

		const char * sep = "\t";
		char * p;
		p = strtok(input, sep);
		string eTissue1 = p;
		int index1 = etissue_index_map[eTissue1];
		int index2 = 0;

		int count = 0;
		while(p)
		{
			count++;
			if(count == 1)  // this is the eTissue1
			{
				p = strtok(NULL, sep);
				continue;
			}
			if(count == 2)  // this is the eTissue2
			{
				string eTissue2 = p;
				int index2 = etissue_index_map[eTissue2];

				p = strtok(NULL, sep);
				continue;
			}
			if(count == 3)
			{
				float dist = stof(p);
				tissue_hierarchical_pairwise[index1][index2] = dist;
				tissue_hierarchical_pairwise[index2][index1] = dist;
				break;
			}
		}

	}
	fclose(file_in);


	*/



	//=====================================================
	//********************** hierarchy ********************
	//=====================================================

	cout << "now loading the tissue hierarchy..." << endl;



	// the hierarchy:
	// what to build for the hierarchy computing (prepared and renewed):
	//		[2]. hashing all the leaves to their parents (in order to retrieve the parental variable array)
	//		[4]. hashing all the internal nodes to their children and parent (with length to them, or the variance), in order to build the computational matrices
	//		[5]. having a bi-directional list for the internal nodes (in order to build and fill in computtional matrices)
	//		(6). building the tissue distance list (to its parent) based on etissue_list, to be used by the actual regularization
	
	//unordered_map<string, hierarchy_neighbor> hash_leaf_parent;		// --> [2]
	//unordered_map<string, vector< hierarchy_neighbor >> hash_internode_neighbor;
																		// --> [4]
	//vector<string> internode_list;									// --> [5]
	//unordered_map<string, int> internode_index_map;					// --> [5]
	//int num_internode;
	//vector<float> etissue_dis_par_list;								// --> (6)





	// TODO (Mar.29): load the following two:
	//unordered_map<string, hierarchy_neighbor> hash_leaf_parent;		// --> [2]
	//unordered_map<string, vector< hierarchy_neighbor >> hash_internode_neighbor;
																		// --> [4]
	while(1)
	{
		//
		//
		//
		//
		//

		string leaf;
		string parent;
		float branch;

		hierarchy_neighbor tuple;
		tuple.node = parent;
		tuple.branch = branch;
		hash_leaf_parent.emplace(leaf, tuple);
	}
	while(1)
	{
		//
		//
		//
		//
		//

		string internode;
		string neighbor1;		// child1 node
		float branch1;
		string neighbor2;		// child2 node
		float branch2;
		string neighbor3;		// parent node
		float branch3;

		vector<hierarchy_neighbor> vec;

		hierarchy_neighbor tuple1;
		tuple1.node = neighbor1;
		tuple1.branch = branch1;
		vec.push_back(tuple1);

		hierarchy_neighbor tuple2;
		tuple2.node = neighbor2;
		tuple2.branch = branch2;
		vec.push_back(tuple2);

		hierarchy_neighbor tuple3;
		tuple3.node = neighbor3;
		tuple3.branch = branch3;
		vec.push_back(tuple3);


		hash_internode_neighbor.emplace(internode, vec);
	}



	//vector<string> internode_list;									// --> [5]
	//unordered_map<string, int> internode_index_map;					// --> [5]
	//int num_internode;
	int count = 0;
	for( auto it = hash_internode_neighbor.begin(); it != hash_internode_neighbor.end(); ++it )
	{
		string internode = it->first;
		internode_list.push_back(internode);
		internode_index_map[internode] = count;
		count += 1;
	}
	num_internode = count;

	//vector<float> etissue_dis_par_list;								// --> (6)
	for(int i=0; i<num_etissue; i++)
	{
		etissue_dis_par_list.push_back(0);
	}
	for(auto it = hash_leaf_parent.begin(); it != hash_leaf_parent.end(); ++it )
	{
		string leaf = it->first;
		int etissue_index = etissue_index_map[leaf];
		etissue_dis_par_list[etissue_index] = (it->second).branch;
	}


	return;
}





void opt_para_init()
{
	puts("opt_para_init..");

	//=============== snp_dosage_list ===============
	for(int i=0; i<NUM_CHR; i++)
	{
		long num_temp = snp_name_list[i].size();
		float * p = (float *)calloc( num_temp, sizeof(float) );
		snp_dosage_list[i] = p;
	}

	//=============== gene_rpkm_exp ===============
	gene_rpkm_exp = (float *)calloc( num_gene, sizeof(float) );

	//=============== cellenv_hidden_var ===============
	cellenv_hidden_var = (float *)calloc( num_cellenv, sizeof(float) );

	//=============== batch_var ===============
	batch_var = (float *)calloc( num_batch, sizeof(float) );

	//=============== batch_hidden_var ===============
	batch_hidden_var = (float *)calloc( num_batch_hidden, sizeof(float) );



	//=============== cube_para_dev_cis_gene ===============
	for(int j=0; j<num_etissue; j++)
	{
		Matrix_imcomp matrix_imcomp;
		matrix_imcomp.init(num_gene);
		for(long int i=0; i<num_gene; i++)
		{
			string gene = gene_list[i];
			unordered_map<string, int>::const_iterator got = gene_xymt_rep.find(gene);
			if ( got != gene_xymt_rep.end() )
			{
				continue;
			}
			else
			{
				long int first = gene_cis_index[gene].first;  // index
				long int second = gene_cis_index[gene].second;  // index
				long int amount = second - first + 1;
				matrix_imcomp.init_element(i, amount + 1);

				// assing the chr and the tss:
				matrix_imcomp.init_assign_chr(i, gene_tss[gene].chr);
				//matrix_imcomp.init_assign_sst(i, gene_tss[gene].tss);		// Here is a BUG: sst != tss
																			// sst: the start index of cis SNPs for one gene; tss: transcription start site
				matrix_imcomp.init_assign_sst(i, gene_cis_index[gene].first);		// Here is a BUG: sst != tss
			}
		}
		cube_para_dev_cis_gene.push_back(matrix_imcomp);
	}

	//=============== matrix_para_dev_snp_cellenv ===============
	matrix_para_dev_snp_cellenv.init(num_cellenv, num_snp + 1);		// we do have the intercept term here

	//=============== cube_para_dev_cellenv_gene ===============
	for(int j=0; j<num_etissue; j++)
	{
		Matrix matrix;
		matrix.init(num_gene, num_cellenv + 1);						// we do have the intercept term here
		cube_para_dev_cellenv_gene.push_back(matrix);
	}

	//=============== matrix_para_dev_batch_batch_hidden ===============
	matrix_para_dev_batch_batch_hidden.init(num_batch_hidden, num_batch + 1);

	//=============== matrix_para_dev_batch_hidden_gene ===============
	matrix_para_dev_batch_hidden_gene.init(num_gene, num_batch_hidden + 1);



	//=====================================================
	//********************** hierarchy ********************
	//=====================================================
	cout << "initializing the hierarchy prior containers" << endl;

	//=============== cube_para_cis_gene_parent ===============
	for(int j=0; j<num_etissue; j++)
	{
		Matrix_imcomp matrix_imcomp;
		matrix_imcomp.init(num_gene);
		for(long int i=0; i<num_gene; i++)
		{
			int dimension = cube_para_dev_cis_gene[j].get_dimension2(i);
			matrix_imcomp.init_element(i, dimension);
		}
		cube_para_cis_gene_parent.push_back(matrix_imcomp);
	}

	//=============== cube_para_cellenv_gene_parent ===============
	for(int j=0; j<num_etissue; j++)
	{
		Matrix matrix;
		matrix.init(num_gene, num_cellenv + 1);						// we do have the intercept term here
		cube_para_cellenv_gene_parent.push_back(matrix);
	}


}




void opt_para_release()
{
	//=============== snp_dosage_list ===============
	for(int i=0; i<NUM_CHR; i++)
	{
		free(snp_dosage_list[i]);
	}

	//=============== gene_rpkm_exp ===============
	free(gene_rpkm_exp);

	//=============== cellenv_hidden_var ===============
	free(cellenv_hidden_var);

	//=============== batch_var ===============
	free(batch_var);

	//=============== batch_hidden_var ===============
	free(batch_hidden_var);



	//=============== cube_para_dev_cis_gene ===============
	for(int j=0; j<num_etissue; j++)
	{
		cube_para_dev_cis_gene[j].release();
	}


	//=============== matrix_para_dev_snp_cellenv ===============
	matrix_para_dev_snp_cellenv.release();


	//=============== cube_para_dev_cellenv_gene ===============
	for(int j=0; j<num_etissue; j++)
	{
		cube_para_dev_cellenv_gene[j].release();
	}


	//=============== matrix_para_dev_batch_batch_hidden ===============
	matrix_para_dev_batch_batch_hidden.release();


	//=============== matrix_para_dev_batch_hidden_gene ===============
	matrix_para_dev_batch_hidden_gene.release();


	//=====================================================
	//********************** hierarchy ********************
	//=====================================================
	cout << "releasing the hierarchy prior containers" << endl;

	//=============== cube_para_cis_gene_parent ===============
	for(int j=0; j<num_etissue; j++)
	{
		cube_para_cis_gene_parent[j].release();
	}

	//=============== cube_para_cellenv_gene_parent ===============
	for(int j=0; j<num_etissue; j++)
	{
		cube_para_cellenv_gene_parent[j].release();
	}


}




void GPU_init()
{
	int deviceID = 0;
    checkCudaErrors(cudaSetDevice(deviceID));

	//=====================================================
	//================ GPU memory build-up ================
	//=====================================================
	//==== para_dev
	//matrix_para_dev_snp_cellenv --> float * d_para_dev_snp_cellenv
	long int dimension1 = matrix_para_dev_snp_cellenv.get_dimension1();
	long int dimension2 = matrix_para_dev_snp_cellenv.get_dimension2();
    //float * d_para_dev_snp_cellenv;			// whenever it comes to d_xxx (GPU device memory), it's an array other than matrix
    checkCudaErrors(cudaMalloc(&d_para_dev_snp_cellenv, (dimension1*dimension2)*sizeof(float)));
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_dev_snp_cellenv.get_list(i);
    	long int pos_start = i * dimension2;
    	checkCudaErrors(cudaMemcpy( (d_para_dev_snp_cellenv + pos_start), x, dimension2*sizeof(float), cudaMemcpyHostToDevice));
    }
	//vector<Matrix> cube_para_dev_cellenv_gene --> vector<float *> d_list_para_dev_cellenv_gene
	dimension1 = cube_para_dev_cellenv_gene[0].get_dimension1();
	dimension2 = cube_para_dev_cellenv_gene[0].get_dimension2();
    //vector<float *> d_list_para_dev_cellenv_gene;			// whenever it comes to d_xxx (GPU device memory), it's an array other than matrix
    for(int j=0; j<num_etissue; j++)
    {
		float * d_para_dev_cellenv_gene;
	    checkCudaErrors(cudaMalloc(&d_para_dev_cellenv_gene, (dimension1*dimension2)*sizeof(float)));
	    for(long int i=0; i<dimension1; i++)
	    {
	    	float * x = cube_para_dev_cellenv_gene[j].get_list(i);
	    	long int pos_start = i * dimension2;
	    	checkCudaErrors(cudaMemcpy( (d_para_dev_cellenv_gene + pos_start), x, dimension2*sizeof(float), cudaMemcpyHostToDevice));
	    }
	    d_list_para_dev_cellenv_gene.push_back(d_para_dev_cellenv_gene);
    }
	//Matrix matrix_para_dev_batch_batch_hidden --> float * d_para_dev_batch_batch_hidden
	dimension1 = matrix_para_dev_batch_batch_hidden.get_dimension1();
	dimension2 = matrix_para_dev_batch_batch_hidden.get_dimension2();
    //float * d_para_dev_batch_batch_hidden;			// whenever it comes to d_xxx (GPU device memory), it's an array other than matrix
    checkCudaErrors(cudaMalloc(&d_para_dev_batch_batch_hidden, (dimension1*dimension2)*sizeof(float)));
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_dev_batch_batch_hidden.get_list(i);
    	long int pos_start = i * dimension2;
    	checkCudaErrors(cudaMemcpy( (d_para_dev_batch_batch_hidden + pos_start), x, dimension2*sizeof(float), cudaMemcpyHostToDevice));
    }
	//Matrix matrix_para_dev_batch_hidden_gene --> float * d_para_dev_batch_hidden_gene
	dimension1 = matrix_para_dev_batch_hidden_gene.get_dimension1();
	dimension2 = matrix_para_dev_batch_hidden_gene.get_dimension2();
    //float * d_para_dev_batch_hidden_gene;			// whenever it comes to d_xxx (GPU device memory), it's an array other than matrix
    checkCudaErrors(cudaMalloc(&d_para_dev_batch_hidden_gene, (dimension1*dimension2)*sizeof(float)));
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_dev_batch_hidden_gene.get_list(i);
    	long int pos_start = i * dimension2;
    	checkCudaErrors(cudaMemcpy( (d_para_dev_batch_hidden_gene + pos_start), x, dimension2*sizeof(float), cudaMemcpyHostToDevice));
    }



    //==== para
	//matrix_para_snp_cellenv --> float * d_para_snp_cellenv
	dimension1 = matrix_para_snp_cellenv.get_dimension1();
	dimension2 = matrix_para_snp_cellenv.get_dimension2();
    //float * d_para_snp_cellenv;			// whenever it comes to d_xxx (GPU device memory), it's an array other than matrix
	checkCudaErrors(cudaMalloc(&d_para_snp_cellenv, (dimension1*dimension2)*sizeof(float)));
	for(long int i=0; i<dimension1; i++)
	{
		float * x = matrix_para_snp_cellenv.get_list(i);
		long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy( (d_para_snp_cellenv + pos_start), x, dimension2*sizeof(float), cudaMemcpyHostToDevice));
    }
	//vector<Matrix> cube_para_cellenv_gene --> vector<float *> d_list_para_cellenv_gene
	dimension1 = cube_para_cellenv_gene[0].get_dimension1();
	dimension2 = cube_para_cellenv_gene[0].get_dimension2();
    //vector<float *> d_list_para_cellenv_gene;			// whenever it comes to d_xxx (GPU device memory), it's an array other than matrix
    for(int j=0; j<num_etissue; j++)
    {
		float * d_para_cellenv_gene;
	    checkCudaErrors(cudaMalloc(&d_para_cellenv_gene, (dimension1*dimension2)*sizeof(float)));
	    for(long int i=0; i<dimension1; i++)
	    {
	    	float * x = cube_para_cellenv_gene[j].get_list(i);
	    	long int pos_start = i * dimension2;
	    	checkCudaErrors(cudaMemcpy( (d_para_cellenv_gene + pos_start), x, dimension2*sizeof(float), cudaMemcpyHostToDevice));
	    }
	    d_list_para_cellenv_gene.push_back(d_para_cellenv_gene);
    }
	//Matrix matrix_para_batch_batch_hidden --> float * d_para_batch_batch_hidden
	dimension1 = matrix_para_batch_batch_hidden.get_dimension1();
	dimension2 = matrix_para_batch_batch_hidden.get_dimension2();
    //float * d_para_batch_batch_hidden;			// whenever it comes to d_xxx (GPU device memory), it's an array other than matrix
    checkCudaErrors(cudaMalloc(&d_para_batch_batch_hidden, (dimension1*dimension2)*sizeof(float)));
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_batch_batch_hidden.get_list(i);
    	long int pos_start = i * dimension2;
    	checkCudaErrors(cudaMemcpy( (d_para_batch_batch_hidden + pos_start), x, dimension2*sizeof(float), cudaMemcpyHostToDevice));
    }
	//Matrix matrix_para_batch_hidden_gene --> float * d_para_batch_hidden_gene
	dimension1 = matrix_para_batch_hidden_gene.get_dimension1();
	dimension2 = matrix_para_batch_hidden_gene.get_dimension2();
    //float * d_para_batch_hidden_gene;			// whenever it comes to d_xxx (GPU device memory), it's an array other than matrix
    checkCudaErrors(cudaMalloc(&d_para_batch_hidden_gene, (dimension1*dimension2)*sizeof(float)));
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_batch_hidden_gene.get_list(i);
    	long int pos_start = i * dimension2;
    	checkCudaErrors(cudaMemcpy( (d_para_batch_hidden_gene + pos_start), x, dimension2*sizeof(float), cudaMemcpyHostToDevice));
    }



    //==== temp
    // temp space for Matrix Multiplication
	//float * d_temp_snp_cellenv;
	dimension1 = matrix_para_snp_cellenv.get_dimension1();
	dimension2 = matrix_para_snp_cellenv.get_dimension2();
	checkCudaErrors(cudaMalloc(&d_temp_snp_cellenv, (dimension1*dimension2)*sizeof(float)));

	//float * d_temp_cellenv_gene;
	dimension1 = cube_para_cellenv_gene[0].get_dimension1();
	dimension2 = cube_para_cellenv_gene[0].get_dimension2();
	checkCudaErrors(cudaMalloc(&d_temp_cellenv_gene, (dimension1*dimension2)*sizeof(float)));

	//float * d_temp_batch_batch_hidden;
	dimension1 = matrix_para_batch_batch_hidden.get_dimension1();
	dimension2 = matrix_para_batch_batch_hidden.get_dimension2();
    checkCudaErrors(cudaMalloc(&d_temp_batch_batch_hidden, (dimension1*dimension2)*sizeof(float)));

	//float * d_temp_batch_hidden_gene;
	dimension1 = matrix_para_batch_hidden_gene.get_dimension1();
	dimension2 = matrix_para_batch_hidden_gene.get_dimension2();
    checkCudaErrors(cudaMalloc(&d_temp_batch_hidden_gene, (dimension1*dimension2)*sizeof(float)));





    //==== temp (intermediate) variables
	checkCudaErrors(cudaMalloc(&d_etissue_index_p, 1*sizeof(int)));

	//==== float * d_snp
	checkCudaErrors(cudaMalloc(&d_snp, num_snp*sizeof(float)));

	//==== float * d_expr
	checkCudaErrors(cudaMalloc(&d_expr, num_gene*sizeof(float)));

	//==== float * d_gene_rpkm_exp
	checkCudaErrors(cudaMalloc(&d_gene_rpkm_exp, num_gene*sizeof(float)));

	//==== float * d_gene_rpkm_exp_cellenv
	checkCudaErrors(cudaMalloc(&d_gene_rpkm_exp_cellenv, num_gene*sizeof(float)));

	//==== float * d_gene_rpkm_exp_batch
	checkCudaErrors(cudaMalloc(&d_gene_rpkm_exp_batch, num_gene*sizeof(float)));

	//==== float * d_error_list
	checkCudaErrors(cudaMalloc(&d_error_list, num_gene*sizeof(float)));

	//==== float * d_cellenv_hidden_var
	checkCudaErrors(cudaMalloc(&d_cellenv_hidden_var, num_cellenv*sizeof(float)));

	//==== float * d_batch_var
	checkCudaErrors(cudaMalloc(&d_batch_var, num_batch*sizeof(float)));

	//==== float * d_batch_hidden_var
	checkCudaErrors(cudaMalloc(&d_batch_hidden_var, num_batch_hidden*sizeof(float)));

}




void GPU_release()
{

	//====================================================
	//================ GPU data retrieval ================
	//====================================================
	//==== para_dev
	//matrix_para_dev_snp_cellenv --> float * d_para_dev_snp_cellenv
	long int dimension1 = matrix_para_dev_snp_cellenv.get_dimension1();
	long int dimension2 = matrix_para_dev_snp_cellenv.get_dimension2();
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_dev_snp_cellenv.get_list(i);
    	long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy(x, (d_para_dev_snp_cellenv + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
    }
    checkCudaErrors(cudaFree(d_para_dev_snp_cellenv));
	//vector<Matrix> cube_para_dev_cellenv_gene --> <float *> d_list_para_dev_cellenv_gene
	dimension1 = cube_para_dev_cellenv_gene[0].get_dimension1();
	dimension2 = cube_para_dev_cellenv_gene[0].get_dimension2();
	for(int j=0; j<num_etissue; j++)
	{
		float * d_para_dev_cellenv_gene = d_list_para_dev_cellenv_gene[j];
	    for(long int i=0; i<dimension1; i++)
	    {
	    	float * x = cube_para_dev_cellenv_gene[j].get_list(i);
	    	long int pos_start = i * dimension2;
			checkCudaErrors(cudaMemcpy(x, (d_para_dev_cellenv_gene + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
	    }
	    checkCudaErrors(cudaFree(d_para_dev_cellenv_gene));
	}
	//Matrix matrix_para_dev_batch_batch_hidden --> float * d_para_dev_batch_batch_hidden
	dimension1 = matrix_para_dev_batch_batch_hidden.get_dimension1();
	dimension2 = matrix_para_dev_batch_batch_hidden.get_dimension2();
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_dev_batch_batch_hidden.get_list(i);
    	long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy(x, (d_para_dev_batch_batch_hidden + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
    }
    checkCudaErrors(cudaFree(d_para_dev_batch_batch_hidden));
	//Matrix matrix_para_dev_batch_hidden_gene --> float * d_para_dev_batch_hidden_gene
	dimension1 = matrix_para_dev_batch_hidden_gene.get_dimension1();
	dimension2 = matrix_para_dev_batch_hidden_gene.get_dimension2();
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_dev_batch_hidden_gene.get_list(i);
    	long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy(x, (d_para_dev_batch_hidden_gene + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
    }
    checkCudaErrors(cudaFree(d_para_dev_batch_hidden_gene));


    //==== para
	//matrix_para_snp_cellenv --> float * d_para_snp_cellenv
	dimension1 = matrix_para_snp_cellenv.get_dimension1();
	dimension2 = matrix_para_snp_cellenv.get_dimension2();
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_snp_cellenv.get_list(i);
    	long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy(x, (d_para_snp_cellenv + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
    }
    checkCudaErrors(cudaFree(d_para_snp_cellenv));
	//vector<Matrix> cube_para_cellenv_gene --> <float *> d_list_para_cellenv_gene
	dimension1 = cube_para_cellenv_gene[0].get_dimension1();
	dimension2 = cube_para_cellenv_gene[0].get_dimension2();
	for(int j=0; j<num_etissue; j++)
	{
		float * d_para_cellenv_gene = d_list_para_cellenv_gene[j];
	    for(long int i=0; i<dimension1; i++)
	    {
	    	float * x = cube_para_cellenv_gene[j].get_list(i);
	    	long int pos_start = i * dimension2;
			checkCudaErrors(cudaMemcpy(x, (d_para_cellenv_gene + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
	    }
	    checkCudaErrors(cudaFree(d_para_cellenv_gene));
	}
	//Matrix matrix_para_batch_batch_hidden --> float * d_para_batch_batch_hidden
	dimension1 = matrix_para_batch_batch_hidden.get_dimension1();
	dimension2 = matrix_para_batch_batch_hidden.get_dimension2();
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_batch_batch_hidden.get_list(i);
    	long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy(x, (d_para_batch_batch_hidden + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
    }
    checkCudaErrors(cudaFree(d_para_batch_batch_hidden));
	//Matrix matrix_para_batch_hidden_gene --> float * d_para_batch_hidden_gene
	dimension1 = matrix_para_batch_hidden_gene.get_dimension1();
	dimension2 = matrix_para_batch_hidden_gene.get_dimension2();
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_batch_hidden_gene.get_list(i);
    	long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy(x, (d_para_batch_hidden_gene + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
    }
    checkCudaErrors(cudaFree(d_para_batch_hidden_gene));




    //==== para temp
    // temp space for Matrix Multiplication
	//float * d_temp_snp_cellenv;
    checkCudaErrors(cudaFree(d_temp_snp_cellenv));

	//float * d_temp_cellenv_gene;
    checkCudaErrors(cudaFree(d_temp_cellenv_gene));

	//float * d_temp_batch_batch_hidden;
    checkCudaErrors(cudaFree(d_temp_batch_batch_hidden));

	//float * d_temp_batch_hidden_gene;
    checkCudaErrors(cudaFree(d_temp_batch_hidden_gene));




    //==== temp (intermediate) variables
    //==== int * d_etissue_index_p
    checkCudaErrors(cudaFree(d_etissue_index_p));

	//==== float * d_snp
    checkCudaErrors(cudaFree(d_snp));

	//==== float * d_expr
    checkCudaErrors(cudaFree(d_expr));

	//==== float * d_gene_rpkm_exp
    checkCudaErrors(cudaFree(d_gene_rpkm_exp));

	//==== float * d_gene_rpkm_exp_cellenv
    checkCudaErrors(cudaFree(d_gene_rpkm_exp_cellenv));

    //==== float * d_gene_rpkm_exp_batch
    checkCudaErrors(cudaFree(d_gene_rpkm_exp_batch));

    //==== float * d_error_list
    checkCudaErrors(cudaFree(d_error_list));

	//==== float * d_cellenv_hidden_var
    checkCudaErrors(cudaFree(d_cellenv_hidden_var));

	//==== float * d_batch_var
    checkCudaErrors(cudaFree(d_batch_var));

	//==== float * d_batch_hidden_var
    checkCudaErrors(cudaFree(d_batch_hidden_var));






    checkCudaErrors(cudaDeviceReset());
}






//function: mini-batches gradient; gradient descent
void optimize()
{
	puts("============== entering the optimization routine...");




	// Mar.30 DEBUG
	// the hierarchy code has not yet been finished; so I comment this routine here
	//puts("[xx] loading the tissue hierarchy...");
	//opt_tissue_hierarchy_load();




	puts("[xx] initializing the parameter space in this optimization routine...");
	opt_para_init();
	puts("[xx] loading the prior information for cis- snps...");
	opt_snp_prior_load();




	// TODO:
	// to define and initialze the parent parameter space (only for cis- regulator, and for cellular regulator)
	// we can use the similar data structure with the original two:
	//		vector<Matrix_imcomp> cube_para_cis_gene;
	//		vector<Matrix> cube_para_cellenv_gene;
	// make the variables visible to the hierarchical clustering sub-routine





	//======== likelihood ========
	// save the loglikelihood along the way
	char filename[100] = "../result/loglike.txt";
	FILE * file_out_loglike = fopen(filename, "w+");
	if(file_out_loglike == NULL)
	{
	    fputs("File error\n", stderr); exit(1);
	}


	//======== testing error (predictive error) ========
	sprintf(filename, "%s", "../result/test_error.txt");
	FILE * file_out_testerror = fopen(filename, "w+");
	if(file_out_testerror == NULL)
	{
	    fputs("File error\n", stderr); exit(1);
	}





	//======== GPU global variable init
	GPU_init();





	//============== timing starts ================
    struct timeval time_start;
    struct timeval time_end;
    double diff;
    gettimeofday(&time_start, NULL);




	for(int count1=0; count1<iter_learn_out; count1++)  // one count1 is for iteration across all tissues
	{

		for(int count2=0; count2<num_etissue; count2++)  // one count2 is for one tissue
		{
			string etissue = etissue_list[count2];
			int num_esample = eQTL_tissue_rep[etissue].size();



			//======== likelihood ========
			// indicating the current tissue
			char buf[100];
			sprintf(buf, "%s\t", etissue.c_str());
			fwrite(buf, sizeof(char), strlen(buf), file_out_loglike);



			//======== testing error (predictive error) ========
			//char buf[100];
			sprintf(buf, "%s\t", etissue.c_str());
			fwrite(buf, sizeof(char), strlen(buf), file_out_testerror);





			// entering this tissue
			for(int count3=0; count3<iter_learn_in; count3++)  // one count3 is for a batch_size mini-batch in current tissue
			{


				//
				// TODO: change this module to the real stochastic one (other than rounding over all the samples)
				//
				// QUESTION: can we shuffle the sample list?
				//
				// ANS: not urgent to do that, as the current setting is perceptron
				//


				int pos_start = (batch_size * count3) % (num_esample);
				printf("[@@@] now we are working on %d iter_out (%d total), eTissue #%d (%d total) -- %s (%d training samples in), #%d mini-batch (%d batch size, rounding all samples).\n", count1+1, iter_learn_out, count2+1, num_etissue, etissue.c_str(), num_esample, count3+1, batch_size);
				forward_backward_prop_batch(etissue, pos_start, num_esample);
				// leaving this mini-batch



				// // DEBUG:
				// // we do only one sample (or one mini-batch) for the current tissue
				// break;


				/*
				//======== parameter check ========
				// check "nan" after this mini-batch
				int flag = para_check_nan(etissue);
				if(flag == 1)
				{
					//
					cout << "we get nan..." << endl;
					cout << "the # of mini-batch we are in is ";
					cout << count3 + 1 << endl;
					break;
				}




				//=========================================================================================================
				//****************************************** loglike or testerror *****************************************
				//=========================================================================================================
				int num_check_every = 5;

				//======== likelihood ========
				// (Feb.14) after we finish this mini-batch, we'll need to check the log-likelihood of the model (for the current tissue); or maybe check every several mini-batches
				if(count3 % num_check_every == 0)
				{
					float loglike;
					loglike = cal_loglike(etissue);

					char buf[1024];
					sprintf(buf, "%f\t", loglike);
					fwrite(buf, sizeof(char), strlen(buf), file_out_loglike);
				}

				//======== testing error (predictive error) ========
				// we can check every several mini-batches
				if(count3 % num_check_every == 0)
				{
					float testerror;
					testerror = cal_testerror(etissue);

					//char buf[1024];
					sprintf(buf, "%f\t", testerror);
					fwrite(buf, sizeof(char), strlen(buf), file_out_testerror);
				}
				*/





				// DEBUG
				// Mar.30: DEBUG: run only mini-batch to see the functionality
				break;






			}
			// leaving this etissue


			//======== likelihood ========
			// finish this line in the likelihood file
			fwrite("\n", sizeof(char), 1, file_out_loglike);


			//======== testing error (predictive error) ========
			fwrite("\n", sizeof(char), 1, file_out_testerror);






			// DEBUG: won't consider other tissues
			break;







			// DEBUG: we do think all the tissues
			//DEBUG: won't consider other tissues; only consider the current tissue
			//break;



		}
		//
		// whenever we finish one iteration across all tissues, we should save the learned parameters
		//
		//para_inter_save(count1);
		//
		//


		// (Mar.22, 2016) TODO: now we do the tissue hierarchical prior
		// we only care about the following two:
		//		vector<Matrix_imcomp> cube_para_cis_gene;
		//		vector<Matrix> cube_para_cellenv_gene;
		// we want do things more general




		//hierarchy();




	}// leave the current outer iteration




	//============== timing ends ================
	gettimeofday(&time_end, NULL);
	diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("Time used totally is %f seconds.\n", diff);



	//======== GPU global variable release
	GPU_release();






	//======== likelihood ========
	// finish the likelihood file
	fclose(file_out_loglike);


	//======== testing error (predictive error) ========
	fclose(file_out_testerror);








	opt_para_release();
	puts("============== leaving the optimization routine...");
}


