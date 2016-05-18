// optimization.h
// function: the main optimization routine

#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H


#include <string>
#include "global.h"
#include "lib_matrix.h"


using namespace std;






extern array<float *, NUM_CHR> snp_dosage_list;
extern float * gene_rpkm_exp;  // with length "num_gene"
extern float * cellenv_hidden_var;  // with length "num_cellenv"
extern float * batch_var;  // with length "num_batch"
extern float * batch_hidden_var;  // with length "num_batch_hidden"


extern vector<Matrix_imcomp> cube_para_dev_cis_gene;
extern Matrix matrix_para_dev_snp_cellenv;
extern vector<Matrix> cube_para_dev_cellenv_gene;
extern Matrix matrix_para_dev_batch_batch_hidden;
extern Matrix matrix_para_dev_batch_hidden_gene;



// para
//vector<Matrix_imcomp> cube_para_dev_cis_gene--> vector<float *> d_list_para_cis_gene;
extern vector<float *> d_list_para_cis_gene;
//Matrix matrix_para_dev_snp_cellenv --> float * d_para_dev_snp_cellenv
extern float * d_para_snp_cellenv;
//vector<Matrix> cube_para_dev_cellenv_gene --> float * d_para_dev_cellenv_gene
extern vector<float *> d_list_para_cellenv_gene;
//Matrix matrix_para_dev_batch_batch_hidden --> float * d_para_dev_batch_batch_hidden
extern float * d_para_batch_batch_hidden;
//Matrix matrix_para_dev_batch_hidden_gene --> float * d_para_dev_batch_hidden_gene
extern float * d_para_batch_hidden_gene;


// para dev
//vector<Matrix_imcomp> cube_para_dev_cis_gene--> vector<float *> d_list_para_cis_gene;
extern vector<float *> d_list_para_dev_cis_gene;
//Matrix matrix_para_dev_snp_cellenv --> float * d_para_dev_snp_cellenv
extern float * d_para_dev_snp_cellenv;
//vector<Matrix> cube_para_dev_cellenv_gene --> float * d_para_dev_cellenv_gene
extern vector<float *> d_list_para_dev_cellenv_gene;
//Matrix matrix_para_dev_batch_batch_hidden --> float * d_para_dev_batch_batch_hidden
extern float * d_para_dev_batch_batch_hidden;
//Matrix matrix_para_dev_batch_hidden_gene --> float * d_para_dev_batch_hidden_gene
extern float * d_para_dev_batch_hidden_gene;


// temp
//vector<Matrix_imcomp> cube_para_dev_cis_gene--> vector<float *> d_list_para_cis_gene;
extern float * d_temp_cis_gene;
//matrix_para_dev_snp_cellenv --> float * d_para_dev_snp_cellenv
extern float * d_temp_snp_cellenv;
//vector<Matrix> cube_para_dev_cellenv_gene --> float * d_para_dev_cellenv_gene
extern float * d_temp_cellenv_gene;
//Matrix matrix_para_dev_batch_batch_hidden --> float * d_para_dev_batch_batch_hidden
extern float * d_temp_batch_batch_hidden;
//Matrix matrix_para_dev_batch_hidden_gene --> float * d_para_dev_batch_hidden_gene
extern float * d_temp_batch_hidden_gene;


// intermediate variables
extern int * d_etissue_index_p;
extern float * d_snp;			// real data
extern float * d_expr;			// real data
extern float * d_gene_rpkm_exp;  // with length "num_gene"
extern float * d_gene_rpkm_exp_cis;		// with length "num_gene"
extern float * d_gene_rpkm_exp_cellenv;  	// with length "num_gene"
extern float * d_gene_rpkm_exp_batch;  	// with length "num_gene"
extern float * d_error_list;				// with length "num_gene"
extern float * d_cellenv_hidden_var;  // with length "num_cellenv"
extern float * d_cellenv_hidden_var_backerror;  // with length "num_cellenv"
extern float * d_batch_var;  // with length "num_batch"
extern float * d_batch_hidden_var;  // with length "num_batch_hidden"
extern float * d_batch_hidden_var_backerror;  		// with length "num_batch_hidden"
// for cis- range query
extern long int * d_cis_para_start;				// with length "num_gene", start pos in para (dev) list of this gene
extern long int * d_cis_snp_start;				// with length "num_gene", start pos in snp list of this gene
extern long int * d_cis_para_amount;			// with length "num_gene", amount of cis parameters of this gene
extern long int * d_cis_para_index1;			// with length num_para_cis (below)
extern long int num_para_cis;				// total amount of cis- parameters (across all genes)


extern unordered_map<string, int> d_snp_index_map;
extern vector<float *> d_snp_list;

extern unordered_map<string, int> d_sample_index_map;
extern vector<float *> d_sample_list;

extern unordered_map<string, int> d_batch_index_map;
extern vector<float *> d_batch_list;

extern unordered_map<string, int> d_sample_test_index_map;
extern vector<float *> d_sample_test_list;





typedef struct hierarchy_neighbor
{
	string node;
    float branch;
}hierarchy_neighbor;


extern vector<Matrix_imcomp> cube_para_cis_gene_parent;
extern vector<Matrix> cube_para_cellenv_gene_parent;


extern unordered_map<string, hierarchy_neighbor> hash_leaf_parent;
extern unordered_map<string, vector< hierarchy_neighbor >> hash_internode_neighbor;
extern vector<string> internode_list;
extern unordered_map<string, int> internode_index_map;
extern int num_internode;
extern vector<float> etissue_dis_par_list;




// some assistant components:
// the prior number for each un-pruned snp for regularization (from pruned snps and chromatin states); per etissue, per chromosome, for each snp
// TODO: we also still need to integrate distance prior later on with the following prior information
extern vector<vector<vector<float>>> prior_tissue_vector;
// pairwise phylogenetic distance between etissues
extern vector<vector<float>> tissue_hierarchical_pairwise;


// learning control parameters:
extern int iter_learn_out;  // iteration across all tissues
extern int iter_learn_in;  // iteration across all samples from one tissue
extern int batch_size;
extern float rate_learner;  // the learning rate




// load all the cis- snp prior information (tissue specific) from file outside
void opt_snp_prior_load();


// load the pairwise tissue hierarchy from prepared file outside
void opt_tissue_hierarchy_load();


// initialize some local parameter containers
void opt_para_init();


// release the memory for some dynamically allocated space (if there is)
void opt_para_release();


// GPU
void GPU_init();
void GPU_releae();


// main entrance
void optimize();




#endif

// end of optimization.h