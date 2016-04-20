// function: processing genotype relevant data (dosage data)

#include "genotype.h"
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
#include "global.h"
#include "main.h"
#include <math.h>



using namespace std;



long int snp_info_read()
{
	long int num = 0;

	int i;
	for(i=0; i<NUM_CHR; i++)
	{
		int chr = i+1;
		vector<string> vec1;
		vector<long> vec2;
		snp_name_list[i] = vec1;
		snp_pos_list[i] = vec2;

		//======== get all SNPs with their snp_info (count, position) ========
		char filename[100];
		filename[0] = '\0';
		strcat(filename, filename_data_source);
		strcat(filename, "genotype/chr");
		char chrom[10];
		sprintf(chrom, "%d", chr);
		strcat(filename, chrom);
		strcat(filename, "/SNP_info.txt");
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
			// the target from this code section: snp (string); position (long)
			char input2[input_length];
			strcpy(input2, input);
			char * pos = strstr(input2, " ");
			pos++;
			string snp = strtok(input, " ");
			long position = strtol(pos, NULL, 10);

			snp_name_list[i].push_back(snp);
			snp_pos_list[i].push_back(position);
		}

		fclose(file_in);
		//======================================

		num += snp_name_list[i].size();
	}

	return num;
}



void snp_dosage_load(array<float *, NUM_CHR> * array_pointer, string individual)
{
	/*
	int i;
	for(i=0; i<NUM_CHR; i++)
	{
		int chr = i+1;

		//======== get all SNPs with their snp_info (count, position) ========
		char filename[100];
		strcat(filename, filename_data_source);
		strcat(filename, "genotype/chr");
		char chrom[10];
		sprintf(chrom, "%d", chr);
		strcat(filename, chrom);
		strcat(filename, "/SNP_dosage_");
		char individual1[20];
		StrToCharSeq(individual1, individual);
		strcat(filename, individual1);
		strcat(filename, ".txt");
		//puts("the current file worked on is: ");
		//puts(filename);

		FILE * file_in = fopen(filename, "r");
		if(file_in == NULL)
		{
			fputs("File error\n", stderr); exit (1);
		}

		int input_length = 100;
		char input[input_length];
		int j = 0;
		while(fgets(input, input_length, file_in) != NULL)
		{
			trim(input);

			float dosage = stof(input);
			(* array_pointer)[i][j] = dosage;
			j ++;
		}

		fclose(file_in);
		//======================================
	}
	*/

	// we need another routine to load the snps from rep in memory
	// fill in this: unordered_map<string, vector<vector<float>>> snp_dosage_rep;
	// with these: array<vector<float>, NUM_CHR> * array_pointer, string individual
	for(int i=0; i<NUM_CHR; i++)
	{
		for(long j=0; j<snp_name_list[i].size(); j++)
		{
			float dosage = snp_dosage_rep[individual][i][j];
			(* array_pointer)[i][j] = dosage;
		}
	}


	return;
}




void dosage_load()
{
	// fill in: int num_individual; unordered_map<string, vector<vector<float>>> snp_dosage_rep;
	// int num_individual;
	char filename[100];
	filename[0] = '\0';
	strcat(filename, filename_data_source);
	strcat(filename, "list_individuals.txt");
	//puts("the current file worked on is: ");
	//puts(filename);

	FILE * file_in = fopen(filename, "r");
	if(file_in == NULL)
	{
		fputs("File error\n", stderr); exit (1);
	}
	int input_length = 100;
	char input[input_length];
	int count = 0;
	while(fgets(input, input_length, file_in) != NULL)
	{
		trim(input);

		string individual = input;
		vector<vector<float>> vec;
		snp_dosage_rep[individual] = vec;
		count ++;
	}
	fclose(file_in);
	num_individual = count;

	// unordered_map<string, vector<vector<float>>> snp_dosage_rep;
	for(auto it = snp_dosage_rep.begin(); it != snp_dosage_rep.end(); ++it)
	{
		string individual = it->first;
		for(int i=0; i<NUM_CHR; i++)
		{
			int chr = i+1;
			vector<float> vec;
			snp_dosage_rep[individual].push_back(vec);
			// read the dosage file for this individual on this chromosome
			char filename[100];
			filename[0] = '\0';
			strcat(filename, filename_data_source);
			strcat(filename, "genotype/chr");
			char chrom[10];
			sprintf(chrom, "%d", chr);
			strcat(filename, chrom);
			strcat(filename, "/SNP_dosage_");
			char individual1[20];
			StrToCharSeq(individual1, individual);
			strcat(filename, individual1);
			strcat(filename, ".txt");
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

				float dosage = stof(input);
				snp_dosage_rep[individual][i].push_back(dosage);
			}
			fclose(file_in);
		}
	}

}

