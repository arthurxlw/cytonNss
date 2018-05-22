/*
Copyright 2018 XIAOLIN WANG 

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "CorpusSi.h"
#include "Global.h"

namespace sentSeg
{

void CorpusSi::init(const string& corpusFile)
{
	vector<string> sns;
	XLLib::readFile(corpusFile, sns);
	int is=0;
	int n=0;
	int maxN=params.numSents;
	int numSents=sns.size();
	int sumWords=0;
	while(true)
	{
		if(is>=numSents)
		{
			break;
		}

		samples.push_back(SampleSi());
		SampleSi& s=samples.back();
		vector<string>& srcWords=s.srcWords;
		vector<int>& goldBounds=s.goldBounds;

		n=maxN;
		srcWords.clear();
		goldBounds.clear();
		for(int k=0; k<n; k++)
		{
			XLLib::str2list(sns.at(is), srcWords, false);
			goldBounds.push_back(srcWords.size());
			is++;
			if(is>=numSents)
			{
				break;
			}
		}

		s.init();
		sumWords +=srcWords.size();
	}

	printf("corpusSi.load %s , %d sents %d samples, avgLen %.1f words.\n", corpusFile.c_str(), is, samples.size(),
			((double)sumWords)/samples.size());
}

double CorpusSi::evalSegment(ostream* g)
{
	std::set<pair<int, int>> set;
	std::set<int> setB;

	int sumTotal=0;
	int sumRight=0;
	int sumPred=0;
	int sumRightB=0;
	double sumLatency=0;
	int sumWords=0;

	for(int is=0; is<samples.size(); is++)
	{
		SampleSi& s=samples.at(is);
		vector<int>& golds=s.goldBounds;
		vector<int>& preds=s.boundaries;

		set.clear();
		setB.clear();
		for(int k=0; k<golds.size(); k++)
		{
			int start=k>0?golds.at(k-1):0;
			set.insert(std::make_pair(start, golds.at(k)));

			setB.insert(golds.at(k));
		}
		int nTotal=set.size();

		int nRight=0;
		int nRightB=0;
		for(int k=0; k<preds.size(); k++)
		{
			int start=k>0?preds.at(k-1):0;
			pair<int,int> u=std::make_pair(start, preds.at(k));
			if(set.find(u)!=set.end())
			{
				nRight+=1;
			}

			if(setB.find(preds.at(k))!=setB.end())
			{
				nRightB+=1;
			}
		}
		int nPred=preds.size();

		set.clear();

		sumRight+=nRight;
		sumRightB+=nRightB;
		sumTotal+=nTotal;
		sumPred+=nPred;

		int nW=s.srcWords.size();
		sumLatency+=s.latency*nW;
		sumWords +=nW;

	}

	double prec=((double)sumRight)/sumPred;
	double recall=((double)sumRight)/sumTotal;
	double f1=prec*recall/(prec+recall)*2;

	double latency=sumLatency/sumWords;

	double score= f1-latency*params.factorLatency;

	XLLib::printfln(true, g, " %.4f & %.4f & %.4f & %.4f & %.4f",
				prec, recall, f1,  latency, score);

	return f1;
}

} /* namespace reinLearnSentSeg */
