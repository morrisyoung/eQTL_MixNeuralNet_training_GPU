// genotype.h
// function: read dosage chunk (for one individual on one chromosome), according to the information from pruning

#ifndef GENOTYPE_H
#define GENOTYPE_H

#include <unordered_map>
#include <string>
#include <vector>
#include <forward_list>
#include <utility>
#include "global.h"

using namespace std;


// this can also be substituted by pair<long, long>
typedef struct snp_info
{
	long count;
    long position;
}snp_info;


// read snp_info into the specified hashtable
long int snp_info_read();


// read dosage chunk (for one individual on all chromosomes)
void snp_dosage_load(array<float *, NUM_CHR> *, string);


// load all the dosage data for all individuals on all chromosomes
void dosage_load();



#endif

// end of genotype.h