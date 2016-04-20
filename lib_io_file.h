// io_file.h
// function: some basic file operation (read and write) functions

#ifndef IO_FILE_H
#define IO_FILE_H


#include <iostream>
//#include <sys/types.h>
//#include <dirent.h>
#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>
//m#include <string>
#include <string.h>



using namespace std;



// filehandle class
class filehandle
{
	FILE * pointer;

	public:
		// constructor
		filehandle(char * filename, char * type)
		{
			pointer = fopen(filename, type);
			if(pointer == NULL)
			{
				fputs("File error\n", stderr); exit(1);
			}
		}

		// read next new line if there is
		int readline(char * line, long input_length)
		{
			int end = 0;
			if(fgets(line, input_length, pointer) == NULL)
				end = 1;

			trim(line);
			return end;
		}

		// close the current file handle
		void close()
		{
			fclose(pointer);
			return;
		}

		// utility: trim [' ', '\t', '\n'] from both right and left sides of a line
		void trim(char * line)
		{
			// right trim
			size_t n;
			n = strlen(line);
			while (n > 0 && (line[n-1] == ' ' || line[n-1] == '\t' || line[n-1] == '\n'))
			{
				n--;
			}
			line[n] = '\0';
			// left trim
			while (line[0] != '\0' && (line[0] == ' ' || line[0] == '\t'))
			{
				line++;
			}
			return;
		}

};




#endif

// end of io_file.h

