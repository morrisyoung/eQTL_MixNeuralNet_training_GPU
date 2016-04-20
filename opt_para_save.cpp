// function: save all the parameters into file (after this iteration), and test them or interpret them later on
// this is mainly used for testing the training code

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <string>
#include <vector>
#include <utility>
#include "global.h"
#include <cstring>
#include "expression.h"
#include "opt_para_save.h"




using namespace std;




void para_inter_save(int iteration)
{
	cout << "saving the parameters into file (after this iteration)..." << endl;
	char temp[10];


	// write the etissue_list into a file
	//================================ vector<string> etissue_list ================================
	char filename[100] = "../result_inter/etissue_list.txt";
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

		char filename[100] = "../result_inter/para_cis_gene/";
		strcat(filename, "etissue");

		// eTissue #
		sprintf(temp, "%d", i+1);
		strcat(filename, temp);
		//

		strcat(filename, "_");

		// iteration #
		sprintf(temp, "%d", iteration);
		strcat(filename, temp);
		//

		strcat(filename, ".txt");
		//puts("the current file worked on is: ");
		//puts(filename);

		cube_para_cis_gene[etissue_index].save(filename);
		// leaving this etissue
	}


	//================================== Matrix matrix_para_snp_cellenv ===================================
	sprintf(filename, "%s", "../result_inter/para_snp_cellenv_");

	// iteration #
	sprintf(temp, "%d", iteration);
	strcat(filename, temp);
	//

	strcat(filename, ".txt");
	//puts("the current file worked on is: ");
	//puts(filename);

	matrix_para_snp_cellenv.save(filename);


	//============================== vector<Matrix> cube_para_cellenv_gene ==============================
	// this is tissue specific
	for(int i=0; i<num_etissue; i++)
	{
		string etissue = etissue_list[i];
		int etissue_index = i;

		char filename[100] = "../result_inter/para_cellenv_gene/";
		strcat(filename, "etissue");

		// eTissue #
		sprintf(temp, "%d", i+1);
		strcat(filename, temp);
		//

		strcat(filename, "_");

		// iteration #
		sprintf(temp, "%d", iteration);
		strcat(filename, temp);
		//

		strcat(filename, ".txt");
		//puts("the current file worked on is: ");
		//puts(filename);

		cube_para_cellenv_gene[etissue_index].save(filename);
		// leaving this etissue
	}


	//=============================== Matrix matrix_para_batch_batch_hidden ===============================
	sprintf(filename, "%s", "../result_inter/para_batch_batch_hidden_");

	// iteration #
	sprintf(temp, "%d", iteration);
	strcat(filename, temp);
	//
	
	strcat(filename, ".txt");
	//puts("the current file worked on is: ");
	//puts(filename);

	matrix_para_batch_batch_hidden.save(filename);


	//=============================== Matrix matrix_para_batch_hidden_gene ================================
	sprintf(filename, "%s", "../result_inter/para_batch_hidden_gene_");

	// iteration #
	sprintf(temp, "%d", iteration);
	strcat(filename, temp);
	//

	strcat(filename, ".txt");
	//puts("the current file worked on is: ");
	//puts(filename);

	matrix_para_batch_hidden_gene.save(filename);



	cout << "all parameters have been saved into files (after this iteration)..." << endl;
	return;
}


