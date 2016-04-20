// expression data relevant operations, like loading the expressin matrix into the memory

#include "expression.h"
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



using namespace std;


// func: loading the training dataset
long int gene_train_load(char * filename1, char * filename2)  // fill in: eQTL_samples; gene_list; eQTL_tissue_rep
{
	//===================================== eQTL_samples ===========================================
	FILE * file_in = fopen(filename1, "r");
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
		string eTissue = p;
		unordered_map<string, vector<float>> rep;
		eQTL_tissue_rep.emplace(eTissue, rep);

		int count = 0;
		while(p)
		{
			count++;
			if(count == 1)  // this is the eTissue
			{
				p = strtok(NULL, sep);
				continue;
			}

			// append this sample, and iterate across all samples
			string sample = p;
			vector<float> list;
			eQTL_tissue_rep[eTissue].emplace(sample, list);
			eQTL_samples.emplace(sample, eTissue);

			p = strtok(NULL, sep);
		}

	}
	fclose (file_in);


	//===================================== gene_list; eQTL_tissue_rep ===========================================
	unordered_map<int, string> index_rep;
	file_in = fopen(filename2, "r");
	if(file_in == NULL)
	{
		fputs("File error\n", stderr); exit (1);
	}
	int count = 0;
	while(fgets(input, input_length, file_in) != NULL)
	{
		count++;
		switch(count)
		{
			case 1:
			{
				// // fill the index_rep, with eQTL_samples
				int index = 0;
				trim(input);

				const char * sep = "\t";
				char * p;
				p = strtok(input, sep);

				while(p)
				{
					index++;
					if(index == 1 || index == 2)
					{
						p = strtok(NULL, sep);
						continue;
					}

					string sample = p;
 					unordered_map<string, string>::const_iterator got = eQTL_samples.find(sample);
  					if ( got != eQTL_samples.end() )
  					{
  						index_rep.emplace(index, sample);
  					}

					p = strtok(NULL, sep);
				}
				break;
			}

			default:
			{
				// fill the gene_list, and eQTL_tissue_rep
				int index = 0;
				trim(input);

				const char * sep = "\t";
				char * p;
				p = strtok(input, sep);
				string gene = p;
				gene_list.push_back(gene);

				while(p)
				{
					index++;
					if(index == 1 || index == 2)
					{
						p = strtok(NULL, sep);
						continue;
					}

 					unordered_map<int, string>::const_iterator got = index_rep.find(index);
  					if ( got != index_rep.end() )
  					{
						char rpkm[100];
						strcpy(rpkm, p);
						float expression = stof(rpkm);
						string sample = index_rep[index];
						string eTissue = eQTL_samples[sample];
						eQTL_tissue_rep[eTissue][sample].push_back(expression);
  					}

					p = strtok(NULL, sep);
				}
				break;
			}
		}


	}
	fclose (file_in);


	//===================================== gene_index_map ===========================================
	//unordered_map<string, int> gene_index_map;  // re-map those genes into their order (reversed hashing of above)
	for(int i=0; i<gene_list.size(); i++)
	{
		string gene = gene_list[i];
		gene_index_map[gene] = i;
	}


	//===================================== etissue_list ===========================================
	for(auto it=eQTL_tissue_rep.begin(); it != eQTL_tissue_rep.end(); ++it)
	{
		string etissue = it->first;
		etissue_list.push_back(etissue);
	}


	//===================================== etissue_index_map ===========================================
	// unordered_map<string, int> etissue_index_map;  // re-map those etissues into their order (reversed hashing above)
	for(int i=0; i<etissue_list.size(); i++)
	{
		string etissue = etissue_list[i];
		etissue_index_map[etissue] = i;
	}


	//===================================== esample_tissue_rep ===========================================
	//unordered_map<string, vector<string>> esample_tissue_rep;  // esample lists of all etissues
	for(int i=0; i<etissue_list.size(); i++)
	{
		string etissue = etissue_list[i];
		vector<string> vec;
		esample_tissue_rep[etissue] = vec;
		for(auto it=eQTL_tissue_rep[etissue].begin(); it != eQTL_tissue_rep[etissue].end(); ++it)
		{
			string esample = it->first;
			esample_tissue_rep[etissue].push_back(esample);
		}
	}


	return gene_list.size();
}





// func: loading the testing dataset
void gene_test_load(char * filename1, char * filename2)  // fill in: eQTL_tissue_rep_test, eQTL_samples_test, esample_tissue_rep_test
{

	//===================================== eQTL_samples_test ===========================================
	FILE * file_in = fopen(filename1, "r");
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
		string eTissue = p;
		unordered_map<string, vector<float>> rep;
		eQTL_tissue_rep_test.emplace(eTissue, rep);

		int count = 0;
		while(p)
		{
			count++;
			if(count == 1)  // this is the eTissue
			{
				p = strtok(NULL, sep);
				continue;
			}

			// append this sample, and iterate across all samples
			string sample = p;
			vector<float> list;
			eQTL_tissue_rep_test[eTissue].emplace(sample, list);
			eQTL_samples_test.emplace(sample, eTissue);

			p = strtok(NULL, sep);
		}

	}
	fclose (file_in);


	//===================================== eQTL_tissue_rep ===========================================
	unordered_map<int, string> index_rep;
	file_in = fopen(filename2, "r");
	if(file_in == NULL)
	{
		fputs("File error\n", stderr); exit (1);
	}
	int count = 0;
	while(fgets(input, input_length, file_in) != NULL)
	{
		count++;
		switch(count)
		{
			case 1:
			{
				// // fill the index_rep, with eQTL_samples_test
				int index = 0;
				trim(input);

				const char * sep = "\t";
				char * p;
				p = strtok(input, sep);

				while(p)
				{
					index++;
					if(index == 1 || index == 2)
					{
						p = strtok(NULL, sep);
						continue;
					}

					string sample = p;
 					unordered_map<string, string>::const_iterator got = eQTL_samples_test.find(sample);
  					if ( got != eQTL_samples_test.end() )
  					{
  						index_rep.emplace(index, sample);
  					}

					p = strtok(NULL, sep);
				}
				break;
			}

			default:
			{
				// fill in the eQTL_tissue_rep_test
				int index = 0;
				trim(input);

				const char * sep = "\t";
				char * p;
				p = strtok(input, sep);
				string gene = p;
				//gene_list.push_back(gene);

				while(p)
				{
					index++;
					if(index == 1 || index == 2)
					{
						p = strtok(NULL, sep);
						continue;
					}

 					unordered_map<int, string>::const_iterator got = index_rep.find(index);
  					if ( got != index_rep.end() )
  					{
						char rpkm[100];
						strcpy(rpkm, p);
						float expression = stof(rpkm);
						string sample = index_rep[index];
						string eTissue = eQTL_samples_test[sample];
						eQTL_tissue_rep_test[eTissue][sample].push_back(expression);
  					}

					p = strtok(NULL, sep);
				}
				break;
			}
		}


	}
	fclose (file_in);


	//===================================== esample_tissue_rep_test ===========================================
	//unordered_map<string, vector<string>> esample_tissue_rep;  // esample lists of all etissues
	for(int i=0; i<etissue_list.size(); i++)
	{
		string etissue = etissue_list[i];
		vector<string> vec;
		esample_tissue_rep_test[etissue] = vec;
		for(auto it=eQTL_tissue_rep_test[etissue].begin(); it != eQTL_tissue_rep_test[etissue].end(); ++it)
		{
			string esample = it->first;
			esample_tissue_rep_test[etissue].push_back(esample);
		}
	}


	return;
}





void gene_tss_load()
{
	int index = 0;

	char filename[100];
	filename[0] = '\0';
	strcat(filename, filename_data_source);
	strcat(filename, "gene_tss.txt");
	//puts("the current file worked on is: ");
	//puts(filename);

	FILE * file_in = fopen(filename, "r");
	if(file_in == NULL)
	{
		fputs("File error\n", stderr); exit (1);
	}
	int input_length = 100000;
	char input[input_length];
	int count = 0;
	while(fgets(input, input_length, file_in) != NULL)
	{
		// fill in this: unordered_map<string, gene_pos> gene_tss
		trim(input);

		const char * sep = "\t";
		char * p;
		p = strtok(input, sep);
		string gene = p;
		gene_pos tuple;
		gene_tss[gene] = tuple;

		index = 0;
		while(p)
		{
			index++;
			if(index == 1)
			{
				p = strtok(NULL, sep);
				continue;
			}
			if(index == 2)
			{
				// chr
				//char temp[100];
				//strcpy(temp, p);
				long temp = strtol(p, NULL, 10);
				gene_tss[gene].chr = temp;
				p = strtok(NULL, sep);
				continue;
			}
			if(index == 3)
			{
				// tss
				//char temp[100];
				//strcpy(temp, p);
				long temp = strtol(p, NULL, 10);
				gene_tss[gene].tss = temp;
				break;
			}
		}


	}
	fclose(file_in);

}




void gene_xymt_load()
{
	char filename[100];
	filename[0] = '\0';
	strcat(filename, filename_data_source);
	strcat(filename, "gene_xymt.txt");

	FILE * file_in = fopen(filename, "r");
	if(file_in == NULL)
	{
		fputs("File error\n", stderr); exit (1);
	}
	int input_length = 100000;
	char input[input_length];
	int count = 0;
	while(fgets(input, input_length, file_in) != NULL)
	{
		// fill in this: unordered_map<string, int> gene_xymt_rep
		trim(input);
		string gene = input;
		gene_xymt_rep[gene] = 1;
	}
	fclose(file_in);

}

