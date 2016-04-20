// saving all the parameters

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <string>
#include <vector>
#include <utility>
#include "global.h"
#include "parameter_save.h"
#include <cstring>
#include "expression.h"




using namespace std;




void para_save()
{
	cout << "saving the parameters into file..." << endl;


	// write the etissue_list into a file
	//================================ vector<string> etissue_list ================================
	char filename[100] = "../result/etissue_list.txt";
	//puts("the current file worked on is: ");
	//puts(filename);

    FILE * file_out = fopen(filename, "w+");
    if(file_out == NULL)
    {
        fputs("File error\n", stderr); exit(1);
    }

    int count = 0;
    for(int i=0; i<etissue_list.size(); i++)
    {
    	count ++;
    	string etissue = etissue_list[i];
		char buf[100];
    	sprintf(buf, "%s\t%d\n", etissue.c_str(), count);
    	fwrite(buf, sizeof(char), strlen(buf), file_out);
    }
	fclose(file_out);


	//================================ vector<Matrix_imcomp> cube_para_cis_gene ================================
	// this is tissue specific
	for(int i=0; i<num_etissue; i++)
	{
		string etissue = etissue_list[i];
		int etissue_index = i;

		char filename[100] = "../result/para_cis_gene/";
		char temp[10];
		sprintf(temp, "%d", i+1);
		strcat(filename, "etissue");
		strcat(filename, temp);
		strcat(filename, ".txt");

		cube_para_cis_gene[i].save(filename);
		// leaving this etissue
	}


	//================================== Matrix matrix_para_snp_cellenv ===================================
	sprintf(filename, "%s", "../result/para_snp_cellenv.txt");
	matrix_para_snp_cellenv.save(filename);


	//============================== vector<Matrix> cube_para_cellenv_gene ==============================
	// this is tissue specific
	for(int i=0; i<num_etissue; i++)
	{
		string etissue = etissue_list[i];
		int etissue_index = i;

		char filename[100] = "../result/para_cellenv_gene/";
		char temp[10];
		sprintf(temp, "%d", i+1);
		strcat(filename, "etissue");
		strcat(filename, temp);
		strcat(filename, ".txt");

		cube_para_cellenv_gene[etissue_index].save(filename);
		// leaving this etissue
	}


	//=============================== Matrix matrix_para_batch_batch_hidden ===============================
	sprintf(filename, "%s", "../result/para_batch_batch_hidden.txt");
	matrix_para_batch_batch_hidden.save(filename);


	//=============================== Matrix matrix_para_batch_hidden_gene ================================
	sprintf(filename, "%s", "../result/para_batch_hidden_gene.txt");
	matrix_para_batch_hidden_gene.save(filename);





	cout << "all parameters have been saved into files..." << endl;
	return;
}


