// the MLE (maximum likelihood estimate) program given fixed hierarchy
// comments: or you can treat this as expectation of the internal nodes, given the branch lengths; the thing is that we have close-form solution of expectation when it's a joint Gaussian

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
#include "genotype.h"
#include "expression.h"
#include "optimization.h"
#include "global.h"
#include "main.h"  // typedef struct tuple_long
#include <math.h>       /* exp */
#include "lib_matrix.h"
#include "opt_hierarchy.h"
#include <cmath>





using namespace std;




//======================================================================================================
//========================================== matrix inversion ==========================================
//======================================================================================================
/*
Matrix inversion:
copy right @:
	http://stanford.edu/~liszt90/acm/notebook.html#file14
	GaussJordan.cc 14/35

// Gauss-Jordan elimination with full pivoting.
//
// Uses:
//   (1) solving systems of linear equations (AX=B)
//   (2) inverting matrices (AX=I)
//   (3) computing determinants of square matrices
//
// Running time: O(n^3)
//
// INPUT:    a[][] = an nxn matrix
//           b[][] = an nxm matrix
//
// OUTPUT:   X      = an nxm matrix (stored in b[][])
//           A^{-1} = an nxn matrix (stored in a[][])
//           returns determinant of a[][]
*/

const double EPS = 1e-10;

typedef vector<int> VI;
typedef float T;
typedef vector<T> VT;
typedef vector<VT> VVT;


T GaussJordan(VVT &a, VVT &b) {
  const int n = a.size();
  const int m = b[0].size();
  VI irow(n), icol(n), ipiv(n);
  T det = 1;

  for (int i = 0; i < n; i++) {
    int pj = -1, pk = -1;
    for (int j = 0; j < n; j++) if (!ipiv[j])
      for (int k = 0; k < n; k++) if (!ipiv[k])
	if (pj == -1 || fabs(a[j][k]) > fabs(a[pj][pk])) { pj = j; pk = k; }
    if (fabs(a[pj][pk]) < EPS) { cerr << "Matrix is singular." << endl; exit(0); }
    ipiv[pk]++;
    swap(a[pj], a[pk]);
    swap(b[pj], b[pk]);
    if (pj != pk) det *= -1;
    irow[i] = pj;
    icol[i] = pk;

    T c = 1.0 / a[pk][pk];
    det *= a[pk][pk];
    a[pk][pk] = 1.0;
    for (int p = 0; p < n; p++) a[pk][p] *= c;
    for (int p = 0; p < m; p++) b[pk][p] *= c;
    for (int p = 0; p < n; p++) if (p != pk) {
      c = a[p][pk];
      a[p][pk] = 0;
      for (int q = 0; q < n; q++) a[p][q] -= a[pk][q] * c;
      for (int q = 0; q < m; q++) b[p][q] -= b[pk][q] * c;      
    }
  }

  for (int p = n-1; p >= 0; p--) if (irow[p] != icol[p]) {
    for (int k = 0; k < n; k++) swap(a[k][irow[p]], a[k][icol[p]]);
  }

  return det;
}
//======================================================================================================
//========================================== matrix inversion ==========================================
//======================================================================================================







string root = "root";



/*
//// input:
//		vector<Matrix_imcomp> cube_para_cis_gene;
//		vector<Matrix> cube_para_cellenv_gene;



//// output: two cubes with all parental variables filled
//		vector<Matrix_imcomp> cube_para_cis_gene_parent;
//		vector<Matrix> cube_para_cellenv_gene_parent;



// INPUT: the hierarchy (with branch length), should be able to fill hashtable2 and hashtable4 below
//		three files (or two practically):
//			1. all leaves hashing all parents (directly intermediate nodes) with the branch lengths;
//			2. all internal nodes hashing children and parents with the branch lengths;
//			3. the root hashing it's child (top node of the tree)

# this is the reference for tree file; detailed data structures see below



// what to build for the hierarchy computing (prepared and renewed):
//		[1]. (not any more) hashing all the leaves and the root ([0]^n) to their variable array
//		[2]. hashing all the leaves to their parents (in order to retrieve the parental variable array)
//		[3]. (not any more) hashing all the internal nodes to variable array to be filled
//		[4]. hashing all the internal nodes to their children and parent (with length to them, or the variance), in order to build the computational matrices
//		[5]. having a bi-directional list for the internal nodes (in order to build and fill in computtional matrices)
//		(6). building the tissue distance list (to its parent) based on etissue_list, to be used by the actual regularization

unordered_map<string, hierarchy_neighbor> hash_leaf_parent;		// --> [2]
unordered_map<string, vector< hierarchy_neighbor >> hash_internode_neighbor;
																// --> [4]
vector<string> internode_list;									// --> [5]
unordered_map<string, int> internode_index_map;					// --> [5]
int num_internode;
vector<float> etissue_dis_par_list;								// --> (6)

with:

typedef struct hierarchy_neighbor
{
	string node;
    float branch;
}hierarchy_neighbor;



// the computing matrices:
// assuming there are n internal nodes:

k1_p1	k1_p2	k1_p3	...	k1_pn		p1 		f1(leaves, root)
...										p2 		f2(leaves, root)
...										.		.
...									x 	.	=	.
...										.		.
...										.		.
kn_p1	k1_p2	k1_p3	...	k1_pn		pn 		fn(leaves, root)

-->
A x P = B
B = C x D (parameter matrix x data matrix)

-->
P = A^{-1} x (C x D)
(the only thing that's changed is the D; so we can prepare A^{-1} and C in advance)


// I will make the following variables local
vector<vector<float>> matrix_computation;						// --> A
vector<vector<float>> matrix_data_para;							// --> C
vector<float> array_data;										// --> D (to fill in each round)
vector<float> P;												// --> P



//pseudocode: (some of the parameters are re-usable)
do the following for both cis- regulator (G) and cellular regulator (C):
	for each gene i:
		retrieve [1], with input and INPUT; build [6] with INPUT
		fill in [2], with INPUT
		initialize [3], with INPUT
		fill in [4], with INPUT
		for each position j in len(G) or len(C):
			go over [4], fill in the A and B matrices above
			P = A^{-1} x B
		fill in cube_para_cis_gene_parent[][i][] or cube_para_cellenv_gene_parent[][i][] with [3] and etissue_list

return cube_para_cis_gene_parent, cube_para_cellenv_gene_parent and [6]
*/






// func: A^{-1} x (C x D)
// input involves the following:
//	vector<vector<float>> matrix_computation;						// --> A
//	vector<vector<float>> matrix_data_para;							// --> C
//	vector<float> array_data;										// --> D (has been fillin before entering)
//	vector<float> P;												// --> P
void matrix_multiply(
	vector<vector<float>> * matrix_computation_pointer,
	vector<vector<float>> * matrix_data_para_pointer,
	vector<float> * array_data_pointer,
	vector<float> * P_pointer)
{
	float * list_temp = (float *)calloc( num_internode, sizeof(float) );

	for(int i=0; i<num_internode; i++)
	{
		list_temp[i] = 0;
		for(int j=0; j<num_etissue + 1; j++)
		{
			list_temp[i] += (* matrix_data_para_pointer)[i][j] * (* array_data_pointer)[j];
		}
	}

	for(int i=0; i<num_internode; i++)
	{
		(* P_pointer)[i] = 0;
		for(int j=0; j<num_internode; j++)
		{
			(* P_pointer)[i] += (* matrix_computation_pointer)[i][j] * list_temp[j];
		}
	}

	free(list_temp);
	return;
}





// func: fill in A and C; inverse A
// input involves the following:
//	vector<vector<float>> matrix_computation;						// --> A
//	vector<vector<float>> matrix_data_para;							// --> C
// task: take derivative for each variable in the joint distribution
//	the solution should have the following format:
//	-(1/l1)*(x1-x)-(1/l2)*(x2-x)+(1/l3)*(x-x3)
//		where (x1, l1) is child 1, (x2, l2) is child 2, and (x3, l3) is the parent (with branch length)
//		child1 and child2 might be leaves, and parent might be the root
void hierarchy_matrix_prepare(vector<vector<float>> * matrix_computation_pointer, vector<vector<float>> * matrix_data_para_pointer)
{
	for(int i=0; i<num_internode; i++)
	{
		string internode = internode_list[i];

		string neighbor1 = hash_internode_neighbor[internode][0].node;		// child1
		float branch1 = hash_internode_neighbor[internode][0].branch;
		float branch1_inv = 1.0 / branch1;
		string neighbor2 = hash_internode_neighbor[internode][1].node;		// child2
		float branch2 = hash_internode_neighbor[internode][1].branch;
		float branch2_inv = 1.0 / branch2;
		string neighbor3 = hash_internode_neighbor[internode][2].node;		// parent
		float branch3 = hash_internode_neighbor[internode][2].branch;
		float branch3_inv = 1.0 / branch3;

		//==== check the three, fill in A and C
		(* matrix_computation_pointer)[i][i] = branch1_inv + branch2_inv + branch3_inv;

		// check neighbor1 (leaf or not)
		unordered_map<string, hierarchy_neighbor>::const_iterator got = hash_leaf_parent.find(neighbor1);
		if( got != hash_leaf_parent.end() )		// this is a leaf
		{
			int etissue_index = etissue_index_map[neighbor1];
			(* matrix_data_para_pointer)[i][etissue_index] += branch1_inv;
		}
		else									// this is an internal node
		{
			int internode_index = internode_index_map[neighbor1];
			(* matrix_computation_pointer)[i][internode_index] = -branch1_inv;
		}

		// check neighbor2 (leaf or not)
		got = hash_leaf_parent.find(neighbor2);
		if( got != hash_leaf_parent.end() )		// this is a leaf
		{
			int etissue_index = etissue_index_map[neighbor2];
			(* matrix_data_para_pointer)[i][etissue_index] += branch2_inv;
		}
		else									// this is an internal node
		{
			int internode_index = internode_index_map[neighbor2];
			(* matrix_computation_pointer)[i][internode_index] = -branch2_inv;
		}

		// check neighbor3 (root or not)
  		if(neighbor3.compare(root) == 0)
		{
			(* matrix_data_para_pointer)[i][num_etissue] += branch3_inv;		// the last element is the root
		}
		else
		{
			int internode_index = internode_index_map[neighbor3];
			(* matrix_computation_pointer)[i][internode_index] = -branch3_inv;
		}



		//==== inverse A
		vector<vector<float>> matrix ((* matrix_computation_pointer));
		// for(int count1=0; count1<num_internode; count1++)
		// {
		// 	vector<float> vec;
		// 	matrix.push_back(vec);
		// 	for(int count2=0; count2<num_internode; count2++)
		// 	{
		// 		matrix[count1].push_back((* matrix_computation_pointer)[count1][count2]);
		// 	}
		// }

		double det = GaussJordan((* matrix_computation_pointer), matrix);



	}

	return;
}






void hierarchy()
{
	//============= variable preparation =============
	// initialize as 0:
	for(int i=0; i<num_etissue; i++)
	{
		cube_para_cis_gene_parent[i].clean();
		cube_para_cellenv_gene_parent[i].clean();
	}

	//==== what we can utilize:
	//unordered_map<string, hierarchy_neighbor> hash_leaf_parent;		// --> [2]
	//unordered_map<string, vector< hierarchy_neighbor >> hash_internode_neighbor;
																		// --> [4]
	//vector<string> internode_list;									// --> [5]
	//unordered_map<string, int> internode_index_map;					// --> [5]
	//int num_internode;
	//vector<float> etissue_dis_par_list;								// --> (6)


	//==== and the following local ones:
	//vector<vector<float>> matrix_computation;						// --> A
	//vector<vector<float>> matrix_data_para;						// --> C
	//vector<float> array_data;										// --> D (to fill in each round)
	//vector<float> P;												// --> P
	vector<vector<float>> matrix_computation;						// --> A
	for(int i=0; i<num_internode; i++)
	{
		vector<float> vec;
		matrix_computation.push_back(vec);
		for(int j=0; j<num_internode; j++)
		{
			matrix_computation[i].push_back(0);
		}
	}
	vector<vector<float>> matrix_data_para;							// --> C
	for(int i=0; i<num_internode; i++)
	{
		vector<float> vec;
		matrix_data_para.push_back(vec);
		for(int j=0; j<num_etissue + 1; j++)	// having the coefficients for all leaves and the root
		{
			matrix_data_para[i].push_back(0);
		}
	}
	hierarchy_matrix_prepare(&matrix_computation, &matrix_data_para);

	vector<float> array_data;										// --> D
	for(int i=0; i<num_etissue + 1; i++)		// including the root
	{
		array_data.push_back(0);
	}

	vector<float> P;												// --> P
	for(int i=0; i<num_internode; i++)
	{
		P.push_back(0);
	}


	//=================================
	//==== for cis- regulator cube ====
	//=================================
	for(int i = 0; i < num_gene; i++)
	{
		int num_regulator = cube_para_cis_gene_parent[0].get_dimension2(i);		// can use any tissue to retrieve the dimension
		for(int j=0; j<num_regulator; j++)
		{
			//==== fill in array_data
			for(int count=0; count<num_etissue + 1; count++)		// including the root
			{
				if(count == num_etissue)							// this is the root
				{
					array_data[count] = 0;							// we set the root prior as 0-array
				}

				array_data[i] = cube_para_cis_gene[count].get(i, j);
			}

			//==== hierarchy calculation
			matrix_multiply(&matrix_computation, &matrix_data_para, &array_data, &P);


  			//==== fill in cube_para_cis_gene_parent[x][i][j], as the computation matrices will be re-used
  			// with: hash_leaf_parent, internode_index_map, P
  			for(int count=0; count<num_etissue; count++)
  			{
				string etissue = etissue_list[count];
				string parent = hash_leaf_parent[etissue].node;
				int parent_index = internode_index_map[parent];

				cube_para_cis_gene_parent[count].assign(i, j, P[parent_index]);
			}

		}// end j, the current regulator

	}//end i, the current gene




	//============================================
	//==== for cellular factor regulator cube ====
	//============================================
	for(int i = 0; i < num_gene; i++)
	{
		int num_regulator = cube_para_cellenv_gene_parent[0].get_dimension2();		// can use any tissue to retrieve the dimension
		for(int j=0; j<num_regulator; j++)
		{
			//==== fill in array_data
			for(int count=0; count<num_etissue + 1; count++)		// including the root
			{
				if(count == num_etissue)							// this is the root
				{
					array_data[count] = 0;
				}

				array_data[i] = cube_para_cellenv_gene[count].get(i, j);
			}

			//==== hierarchy calculation
			matrix_multiply(&matrix_computation, &matrix_data_para, &array_data, &P);


  			//==== fill in cube_para_cellenv_gene_parent[x][i][j], as the computation matrices will be re-used
  			// with: hash_leaf_parent, internode_index_map, P
  			for(int count=0; count<num_etissue; count++)
  			{
				string etissue = etissue_list[count];
				string parent = hash_leaf_parent[etissue].node;
				int parent_index = internode_index_map[parent];

				cube_para_cellenv_gene_parent[count].assign(i, j, P[parent_index]);
			}

		}// end j, the current regulator

	}//end i, the current gene




	return;
}


