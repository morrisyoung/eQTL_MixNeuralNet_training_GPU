// global.h
// function: global variables

#ifndef GLOBAL_H
#define GLOBAL_H


#include "expression.h"
#include "lib_matrix.h"
#include <array>
#include <unordered_map>



using namespace std;


// number of chromosomes (in real data, this should be 22; and in simu data, this should be 1)
#define NUM_CHR 1



// number of cell environment variables; default value is 400
extern int num_cellenv;
// number of un-pruned snps
extern long int num_snp;
// number of genes
extern long int num_gene;
// number of eQTL tissues
extern int num_etissue;
// number of original batch variables
extern int num_batch;
// number of hidden batch variables
extern int num_batch_hidden;
// number of individuals
extern int num_individual;




// genotype relevant:
extern array<vector<string>, NUM_CHR> snp_name_list;
extern array<vector<long>, NUM_CHR> snp_pos_list;
extern unordered_map<string, vector<vector<float>>> snp_dosage_rep;




// expression relevant:
extern unordered_map<string, unordered_map<string, vector<float>>> eQTL_tissue_rep;  // hashing all eTissues to their actual rep, in which all sample from that tissue is hashed to their rpkm array
extern unordered_map<string, string> eQTL_samples;  // hashing all eQTL samples to their tissues
extern vector<string> gene_list;  // all genes from the source file
extern unordered_map<string, int> gene_index_map;  // re-map those genes into their order (reversed hashing of above)
extern vector<string> etissue_list;  // eTissues in order
extern unordered_map<string, int> etissue_index_map;  // re-map those etissues into their order (reversed hashing above)
extern unordered_map<string, vector<string>> esample_tissue_rep;  // esample lists of all etissues
extern unordered_map<string, gene_pos> gene_tss;  // TSS for all genes (including those pruned genes)
extern unordered_map<string, int> gene_xymt_rep;  // map all the X, Y, MT genes

// expression (testing set):
extern unordered_map<string, unordered_map<string, vector<float>>> eQTL_tissue_rep_test;	// hashing all eTissues to their actual rep, in which all sample from that tissue is hashed to their rpkm array
extern unordered_map<string, string> eQTL_samples_test;	// hashing all eQTL samples to their tissues
extern unordered_map<string, vector<string>> esample_tissue_rep_test;	// esample lists of all etissues








// batch variables:
// batch variables are per genotype per sample (one individual may produce several RNA samples)
extern unordered_map<string, vector<float>> batch_individual;
extern unordered_map<string, vector<float>> batch_sample;




//// parameter space:
// cis parameter table
extern vector<vector<float *>> para_cis_gene;
// snp to cell env variable parameter container
extern vector<float *> para_snp_cellenv;
// cell env variable to gene parameter container
extern vector<vector<float *>> para_cellenv_gene;
// original batch to hidden batch
extern vector<float *> para_batch_batch_hidden;
// hidden batch to genes
extern vector<float *> para_batch_hidden_gene;
// mapping the gene to cis snp indices (start position and end position)
extern unordered_map<string, tuple_long> gene_cis_index;
//// re-format into the new class:
extern vector<Matrix_imcomp> cube_para_cis_gene;
extern Matrix matrix_para_snp_cellenv;
extern vector<Matrix> cube_para_cellenv_gene;
extern Matrix matrix_para_batch_batch_hidden;
extern Matrix matrix_para_batch_hidden_gene;





//// file name space
// (note: if we standadize the source data format and name, we only need the upper folder name)
extern char filename_data_source[];
extern char file_para_init[];



//// indicator of whether this is working on real dataset
extern int indicator_real;




#endif

// end of global.h