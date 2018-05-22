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

#include "EvaluateSegment.h"
#include "Global.h"

namespace sentSeg
{

void EvaluateSegment::reset()
{
	sumTotal=0;
	sumRight=0;
	sumPred=0;
	sumRightB=0;
	sumWords=0;
	sumLatency=0;
}
void EvaluateSegment::apply(SampleSi& s, ostream* g)
{
	vector<int>& golds=s.goldBounds;
	vector<int>& preds=s.boundaries;
	double latency=s.latency;

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

	int nWords=s.srcWords.size();

	sumRight+=nRight;
	sumRightB+=nRightB;
	sumTotal+=nTotal;
	sumPred+=nPred;
	sumWords+=nWords;
	sumLatency+=latency*nWords;
}

Precision EvaluateSegment::getResult(ostream* os, vector<double>* pRes)
{
	double prec=((double)sumRight)/sumPred;
	double recall=((double)sumRight)/sumTotal;
	double f1=prec*recall/(prec+recall)*2;

	double latency=sumLatency/sumWords;

	double s0=f1;
	double s1=-latency*params.factorLatency;
	double s=s0+s1;

	string t=XLLib::stringFormat(" %.4f & %.4f & %.4f & %.4f & %.4f ", prec, recall, f1, latency,  s );
	std::cout<<t;
	std::cout.flush();
	if(os!=NULL) *os<<t;

	if(pRes!=NULL)
	{
		pRes->clear();
		pRes->push_back(f1);
		pRes->push_back(-latency);
	}

	return s;
}

} /* namespace reinLearnSentSeg */
