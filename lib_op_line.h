// op_line.h
// function: some functions for line operation

#ifndef OP_LINE_H
#define OP_LINE_H


#include <iostream>
//#include <sys/types.h>
//#include <dirent.h>
#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>
//m#include <string>
#include <string.h>
#include <vector>



using namespace std;



// filehandle class
class line_class
{
	char * pointer;
	int length;
	vector<char *> list_str;		// the output of "split('\t')"; element as char *

	public:
		// constructor
		line_class(char * line)
		{
			pointer = line;
		}

		// split the line by "\t"
		void split_tab()
		{
			const char * sep = "\t";
			char * p;
			p = strtok(pointer, sep);

			while(p)
			{
				char * element = (char *)calloc(strlen(p), sizeof(char));
				strcpy(element, p);
				list_str.push_back(element);
				p = strtok(NULL, sep);
			}

			length = list_str.size();
			return;
		}

		// print the splitted list
		void print_splitted_line()
		{
			cout << "the elements in this line after splitting by tab:" << endl;
			for(unsigned i=0; i<list_str.size(); i++)
			{
				cout << i << " : ";
				cout << list_str[i] << endl;
			}
			return;
		}

		// release the heap memory used temporarily for this line
		void release()
		{
			for(unsigned i=0; i<list_str.size(); i++)
				free(list_str[i]);
			return;
		}

		// get the length of the current splitted line
		int size()
		{
			return length;
		}

		// element access of the splitted line
		char * at(int index)
		{
			return list_str[index];
		}

};



#endif

// end of op_line.h

