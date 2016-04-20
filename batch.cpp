// function: load the batch variables into memory, and initialize some relevant parameters

#include <vector>
#include <string>
#include <unordered_map>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "basic.h"
#include "batch.h"
#include "global.h"



using namespace std;



void batch_load()
{
	// fill in the following
	//unordered_map<string, vector<float>> batch_individual;
	//unordered_map<string, vector<float>> batch_sample;
	//============================ part#1: fill in individual batch rep ============================
	//unordered_map<string, vector<float>> batch_individual;
	char filename[100];
	filename[0] = '\0';
	strcat(filename, filename_data_source);
	strcat(filename, "batch_individuals.txt");
	//puts("the current file worked on is: ");
	//puts(filename);

	FILE * file_in = fopen(filename, "r");
	if(file_in == NULL)
	{
		fputs("File error\n", stderr); exit (1);
	}

	int input_length = 20000;
	char input[input_length];
	int count_line = 0;
	while(fgets(input, input_length, file_in) != NULL)
	{
		count_line++;
		if(count_line == 1)continue;	// label line

		trim(input);

		const char * sep = "\t";
		char * p;
		p = strtok(input, sep);
		string individual = p;
		vector<float> vec;
		batch_individual[individual] = vec;

		int count = 0;
		while(p)
		{
			count++;
			if(count == 1)  // this is the individual
			{
				p = strtok(NULL, sep);
				continue;
			}

			char temp[100];
			strcpy(temp, p);
			float value = stof(temp);
			batch_individual[individual].push_back(value);
			p = strtok(NULL, sep);
		}
	}
	fclose(file_in);


	//============================ part#2: fill in sample batch rep ============================
	//unordered_map<string, vector<float>> batch_sample;
	filename[0] = '\0';
	strcat(filename, filename_data_source);
	strcat(filename, "batch_samples.txt");
	//puts("the current file worked on is: ");
	//puts(filename);

	file_in = fopen(filename, "r");
	if(file_in == NULL)
	{
		fputs("File error\n", stderr); exit (1);
	}

	count_line = 0;
	while(fgets(input, input_length, file_in) != NULL)
	{
		count_line++;
		if(count_line == 1)continue;	// label line

		trim(input);

		const char * sep = "\t";
		char * p;
		p = strtok(input, sep);
		string sample = p;
		vector<float> vec;
		batch_sample[sample] = vec;

		int count = 0;
		while(p)
		{
			count++;
			if(count == 1)  // this is the sample
			{
				p = strtok(NULL, sep);
				continue;
			}

			char temp[100];
			strcpy(temp, p);
			float value = stof(temp);
			batch_sample[sample].push_back(value);
			p = strtok(NULL, sep);
		}
	}
	fclose(file_in);


	// fill in the following:
	//int num_batch;
	//========================= fill in the total number of batch variables =========================
	num_batch = 0;
	for( auto it = batch_individual.begin(); it != batch_individual.end(); ++it )
	{
		int temp = it->second.size();
		num_batch += temp;
		break;
	}
	for( auto it = batch_sample.begin(); it != batch_sample.end(); ++it )
	{
		int temp = it->second.size();
		num_batch += temp;
		break;
	}

}

