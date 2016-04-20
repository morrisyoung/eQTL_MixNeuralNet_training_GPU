// expression.h
// function: read the expression matrix (rpkm values) into the specified data structure

#ifndef EXPRESSION_H
#define EXPRESSION_H


using namespace std;


// the position of a gene
typedef struct gene_pos
{
	int chr;
	long tss;
}gene_pos;


// the cis- region (SNP list range) of a gene
typedef struct tuple_long
{
	long int first;
	long int second;
}tuple_long;


// load the expression matrix into memory (training dataset)
long int gene_train_load(char *, char *);  // fill in: eQTL_samples; gene_list; eQTL_tissue_rep

// load the testing dataset
void gene_test_load(char *, char *);  // fill in: eQTL_tissue_rep_test, eQTL_samples_test, esample_tissue_rep_test



// load tss for all genes, from the annotation file
void gene_tss_load();


// map all the X, Y, MT genes to unordered_map<string, int> gene_xymt_rep
void gene_xymt_load();



#endif

// end of expression.h