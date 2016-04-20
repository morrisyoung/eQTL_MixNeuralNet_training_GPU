// lib_matrix.h
// function: the matrix class and some basic operations

#ifndef LIB_MATRIX_H
#define LIB_MATRIX_H


#include <iostream>
//#include <sys/types.h>
//#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
//#include <string.h>
#include <string>
#include <string.h>
#include <vector>
#include <math.h>       /* exp */




using namespace std;




// matrix class
class Matrix
{
	long int dimension1;
	long int dimension2;
	vector<float *> matrix;			// element as "float *" other than "array", as we don't pre-know the length

	public:
		//================================ constructor =======================================
		void init(long int value1, long int value2)
		{
			dimension1 = value1;
			dimension2 = value2;

			// the matrix will be initialized as 0 matrix, with "calloc"
			for(long int i=0; i<dimension1; i++)
			{
				float * pointer = (float *)calloc( dimension2, sizeof(float) );
				matrix.push_back(pointer);
			}
			return;
		}
		void init(long int value1, long int value2, vector<float *> vec)
		{
			dimension1 = value1;
			dimension2 = value2;

			// the matrix will be initialized as 0 matrix, with "calloc"
			for(long int i=0; i<dimension1; i++)
			{
				float * pointer = (float *)calloc( dimension2, sizeof(float) );
				matrix.push_back(pointer);
			}

			for(long int i=0; i<dimension1; i++)
			{
				for(long int j=0; j<dimension2; j++)
				{
					matrix[i][j] = vec[i][j];
				}
			}
			return;
		}


		//================================ operations =======================================
		// dimension evaluation
		long int get_dimension1()
		{
			return dimension1;
		}
		long int get_dimension2()
		{
			return dimension2;
		}

		// clean: set all the values in this matrix to 0
		void clean()
		{
			for(long int i=0; i<dimension1; i++)
			{
				for(long int j=0; j<dimension2; j++)
				{
					matrix[i][j] = 0;
				}
			}
			return;
		}

		// scale: scale the full matrix according to a given factor
		void scale(float factor)
		{
			for(long int i=0; i<dimension1; i++)
			{
				for(long int j=0; j<dimension2; j++)
				{
					matrix[i][j] = matrix[i][j] * factor;
				}
			}
			return;
		}

		// get
		float get(long int pos1, long int pos2)
		{
			return matrix[pos1][pos2];
		}

		// get element
		float * get_list(long int pos)
		{
			return matrix[pos];
		}


		// assign
		void assign(long int pos1, long int pos2, float value)
		{
			matrix[pos1][pos2] = value;
			return;
		}

		// add on
		void add_on(long int pos1, long int pos2, float value)
		{
			matrix[pos1][pos2] = matrix[pos1][pos2] + value;
			return;
		}


		// check nan: check whether there is Nan in this Matrix
		int check_nan()
		{
			int flag = 0;

			// NOTE: TODO
			// this is not able to compile in GPU-CPU code
			/*
			for(long int i=0; i<dimension1; i++)
			{
				for(long int j=0; j<dimension2; j++)
				{
					float value = matrix[i][j];
					// check nan
					int temp = isnan(value);
					if( temp )
					{
						flag = 1;
						break;
					}
				}
			}
			*/

			return flag;
		}


		// given a filename, try to save this matrix into a file
		void save(char * filename)
		{
			FILE * file_out = fopen(filename, "w+");
			if(file_out == NULL)
			{
			    fputs("File error\n", stderr); exit(1);
			}

			for(long int i=0; i<dimension1; i++)
			{
				for(long int j=0; j<dimension2; j++)
				{
					float parameter = matrix[i][j];
					char buf[1024];
					sprintf(buf, "%f\t", parameter);
					fwrite(buf, sizeof(char), strlen(buf), file_out);
				}
				fwrite("\n", sizeof(char), 1, file_out);
			}
			fclose(file_out);
			return;
		}


		//================================ destructor =======================================
		void release()
		{
			for(long int i=0; i<dimension1; i++)
			{
				free(matrix[i]);
			}
			return;
		}


};




// matrix (imcomplete, for cis- association parameters) class
class Matrix_imcomp
{
	long int dimension;
	vector<long int> list_length;
	vector<float *> matrix;				// element as "float *" other than "array", as we don't pre-know the length
	vector<int>	list_chr;				// the chr index of all these cis- parameter lists
	vector<long int> list_sst;			// the SNP start site of the cis- parameter list

	public:
		//================================ constructor =======================================
		void init(long int length)
		{
			dimension = length;
			for(long int i=0; i<dimension; i++)
			{
				list_length.push_back(0);
				float * pointer = NULL;
				matrix.push_back(pointer);
				list_chr.push_back(-1);
				list_sst.push_back(-1);
			}
			return;
		}

		// initialize each element (empty) with specified length
		void init_element(long int pos, long int length)
		{
			list_length[pos] = length;
			matrix[pos] = (float *)calloc( length, sizeof(float) );
			return;
		}

		// initialize each element (with values) with specified length
		void fill_element(long int pos, long int length, float * list)
		{
			list_length[pos] = length;
			matrix[pos] = (float *)calloc( length, sizeof(float) );
			for(long int i=0; i<length; i++)
			{
				matrix[pos][i] = list[i];
			}
			return;
		}

		// assign the chr value for one specific element
		void init_assign_chr(long int pos, int value)
		{
			list_chr[pos] = value;
			return;
		}

		// assign the SNP start site value for one specific element
		void init_assign_sst(long int pos, long int value)
		{
			list_sst[pos] = value;
			return;
		}


		//================================ operations =======================================
		// get_dimension1
		long int get_dimension1()
		{
			return dimension;
		}

		// get_dimension2, for one specific element
		long int get_dimension2(long int pos)
		{
			return list_length[pos];
		}

		// get chr
		int get_chr(long int pos)
		{
			return list_chr[pos];
		}

		// get tss
		long int get_sst(long int pos)
		{
			return list_sst[pos];
		}

		// clean: set the whole value space to 0
		void clean()
		{
			for(long int i=0; i<dimension; i++)
			{
				for(long int j=0; j<list_length[i]; j++)
				{
					matrix[i][j] = 0;
				}
			}
			return;
		}

		// scale: scale the full matrix_imcomp based on a given factor
		void scale(float factor)
		{
			for(long int i=0; i<dimension; i++)
			{
				for(long int j=0; j<list_length[i]; j++)
				{
					matrix[i][j] = matrix[i][j] * factor;
				}
			}
			return;
		}

		// get
		float get(long int pos1, long int pos2)
		{
			return matrix[pos1][pos2];
		}

		// assign
		void assign(long int pos1, long int pos2, float value)
		{
			matrix[pos1][pos2] = value;
			return;
		}

		// add on
		void add_on(long int pos1, long int pos2, float value)
		{
			matrix[pos1][pos2] = matrix[pos1][pos2] + value;
			return;
		}


		// check nan: check whether there is Nan in this Matrix_imcomp
		int check_nan()
		{
			int flag = 0;

			// NOTE: temporarily remove
			// this is not able to compile in GPU-CPU code
			/*
			for(long int i=0; i<dimension; i++)
			{
				for(long int j=0; j<list_length[i]; j++)
				{
					float value = matrix[i][j];
					// check nan
					int temp = isnan(value);
					if( temp )
					{
						flag = 1;
						break;
					}
				}
			}
			*/

			return flag;
		}


		// given a filename, try to save this matrix into a file
		void save(char * filename)
		{
			FILE * file_out = fopen(filename, "w+");
			if(file_out == NULL)
			{
			    fputs("File error\n", stderr); exit(1);
			}

			for(long int i=0; i<dimension; i++)
			{
				for(long int j=0; j<list_length[i]; j++)
				{
					float parameter = matrix[i][j];
					char buf[1024];
					sprintf(buf, "%f\t", parameter);
					fwrite(buf, sizeof(char), strlen(buf), file_out);
				}
				fwrite("\n", sizeof(char), 1, file_out);
			}
			fclose(file_out);
			return;
		}


		//================================ destructor =======================================
		void release()
		{
			for(long int i=0; i<dimension; i++)
			{
				free(matrix[i]);
			}
			return;
		}


};




#endif

// end of lib_matrix.h


