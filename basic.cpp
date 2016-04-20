// function: some basic functions that maybe used by some parts of the whole program

#include <string.h>
#include <ctype.h>
#include "basic.h"
#include <string>
#include <utility>
#include "global.h"



using namespace std;


void rtrim(char * str)
{
	size_t n;
	n = strlen(str);
	while (n > 0 && (str[n-1] == ' ' || str[n-1] == '\t' || str[n-1] == '\n'))
	{
		n--;
	}
	str[n] = '\0';
}

void ltrim(char * str)
{
	while (str[0] != '\0' && (str[0] == ' ' || str[0] == '\t'))
	{
		str++;
	}
}

void trim(char * str)
{
  rtrim(str);
  ltrim(str);
}

void pair_split(char * p, pair<string, float> * pointer)
{
	char p1[100];
	strcpy(p1, p);

	char * p2 = p1;
	int i = 0;
	while(p2[0] != ' ')
	{
		i++;
		p2++;
	}
	p2++;
	(* pointer).second = stof(p2);

	p1[i] = '\0';
	string snp = p1;
	(* pointer).first = snp;

}


void StrToCharSeq(char * str1, string str)
{
	int i = 0;
	for (i=0; i<str.length(); ++i)
	{
		str1[i] = str[i];
	}
	str1[i] = '\0';

}



// transform the sample ID into individual ID
// in real dataset (GTEx v.4): from "GTEX-R55E-0826-SM-2TC5M" to "GTEX-R55E"
// in simulated dataset: from "105-15" to "105"
string sample_to_individual(string sample)
{
	string individual;

	if(indicator_real == 1)			// real dataset
	{
		int count = 0;
		for(int i=0; i<sample.size(); i++)
		{
			//
			if(sample[i] == '-')
			{
				count++;
			}
			//
			if(count == 2)			// the sub-string before the second '-' is the individual ID
			{
				break;
			}

			individual.push_back(sample[i]);
		}
	}
	else							// simulated dataset
	{
		for(int i=0; i<sample.size(); i++)
		{
			if(sample[i] != '-')
			{
				individual.push_back(sample[i]);
			}
			else
			{
				break;
			}
		}
	}

	return individual;
}


