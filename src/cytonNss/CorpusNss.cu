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

#include "CorpusNss.h"
#include "ParamsNss.h"

namespace sentSeg
{

void CorpusNss::init(const string& corpusFile, Vocabulary* vocab_, int batchSize)
{
	vocab=vocab_;

	ifstream f;
	XLLib::fileReader(corpusFile, f);
	string line;
	vector<string> words;
	while (getline(f, line, '\n'))
	{
		XLLib::str2list(line, words);
		if(words.size()<params.maxSentLen && words.size()>0)
		{
			vector<int> sent;
			vocab->getIds(words, sent);
			sents.push_back(sent);
		}
	}
	f.close();

	indexes.clear();
	for(int i=0; i<sents.size(); i++)
	{
		indexes.push_back(i);
	}
	kIndex=0;

	fprintf(stderr, "corpus %d sents %d batches .\n", sents.size(), sents.size()/batchSize);
}

void CorpusNss::reset()
{
	kIndex=0;
}

void CorpusNss::shuffleReset()
{
	std::random_shuffle(indexes.begin(), indexes.end());
	reset();
}

bool CorpusNss::read(BatchNss& batch, bool training)
{
	int nf=params.numFutureWords;

	if(kIndex>=indexes.size())
	{
		return false;
	}
	int batchSize=batch.x.ni;
	int ns=sents.size()-1;
	assert(ns<RAND_MAX);
	vector<int> idxs;
	int maxSentLen=0;
	int kStart=kIndex;
	for(int i=0; i<batchSize; i++)
	{
		if(kIndex>=indexes.size())
		{
			kIndex=kStart;
		}
		int is=indexes.at(kIndex++);
		idxs.push_back(is);
		int len=sents.at(is).size();
		maxSentLen=std::max(len, maxSentLen);
	}

	int len=maxSentLen+nf;

	HostMatInt& x=batch.x;
	HostMatInt& y=batch.y;
	x.resize(batchSize, len);
	y.resize(batchSize, len);

	batch.lens.clear();
	for(int i=0; i<batchSize; i++)
	{
		int is=idxs.at(i);
		vector<int>& sent=sents.at(is);

		int ix=0;
		for(int k=0; k<sent.size(); k++)
		{
			x.at(i, ix) = sent.at(k);
			ix+=1;
		}

		int ixEnd=ix+nf;
		for(int ks=1; ;ks++)
		{
			int is1 = (is+ks) %sents.size();
			if(training)
			{
				is1=rand()%sents.size();
			}
			vector<int>& sent1=sents.at(is1);
			for(int kw=0; kw<sent1.size(); kw++)
			{
				x.at(i, ix) = sent1.at(kw);
				ix+=1;
				if(ix>=ixEnd)
				{
					break;
				}
			}
			if(ix>=ixEnd)
			{
				break;
			}
		}
		assert(ix==ixEnd);

		for(; ix<len; ix++)
		{
			x.at(i, ix)=vocab->empty;
		}

		{
			int nf0=1;
			int tLen=sent.size();
			int tStart=tLen-1+nf0;
			int tEnd=tStart+nf-nf0+1;
			batch.lens.push_back(tLen);
			for(int iy=0; iy<len; iy++)
			{
				int tv=-1;
				if(iy<nf0)
				{
					tv=-1;
				}
				else if(iy<tStart)
				{
					tv=0;
				}
				else if(iy>=tStart && iy<tEnd)
				{
					tv=iy-tStart+1;
				}
				y.at(i, iy)=tv;
			}
		}

	}
	return true;
}



} /* namespace sentSeg */
