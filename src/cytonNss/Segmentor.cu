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

#include "Segmentor.h"

namespace sentSeg {

void Segmentor::init(NetworkNss* network_, Vocabulary* vocab_, vector<double>* thresholds_)
{
	network=network_;
	vocab=vocab_;
	thresholds=thresholds_;
}


void Segmentor::putWords(const vector<string>& words)
{
	//update nodes;
	int startPos=nodes.size();

	for(vector<string>::const_iterator it=words.begin(); it!=words.end(); it++)
	{
		WordNode node(*it);
		node.wid=vocab->getId(node.word);
		nodes.push_back(node);
	}

	updateScores(startPos);

}

void Segmentor::updateScores(int start)
{
	int len=std::min(params.maxSentLen-start, (int)nodes.size()-start);
	if(len>0)
	{
		if(start==0)
		{
			network->setContext(true);
		}
		else
		{
			network->setContext(false);
		}

		vector<int> input;
		for(int i=start; i<start+len; i++)
		{
			input.push_back(nodes.at(i).wid);
		}
		network->apply(&input[0], 1, len);
		HostMatPrec& probs=network->predY;

		for(unsigned int i=start;i<start+len;i+=1)
		{
			vector<double>& score=nodes.at(i).score;
			score.clear();
			for(int k=0; k<probs.ni; k++)
			{
				double tProb=probs.at(k, i-start);
				score.push_back(tProb);
			}
		}
	}
}

int Segmentor::findBoundary(bool final)
{
	int boundary=-1;
	int nf0=1;
	int maxLen=params.maxSentLen;
	int len=std::min((int)nodes.size(), maxLen);

	if(boundary<0 && len>0 )
	{
		int segLen=-1;
		for(int j=nf0; j<len; j++)
		{
			WordNode& n=nodes.at(j);
			for(int i=n.score.size()-1; i>=1; i--)
			{
				Precision tProb=n.score.at(i);
				if(tProb>=thresholds->at(i-1))
				{
					int tSegLen=j+2-i-nf0;
					if(tSegLen>0)
					{
						segLen=tSegLen;
						break;
					}
				}
			}
			if(segLen>0)
			{
				break;
			}
		}

		if(segLen<0)
		{
			if(len>=maxLen)
			{
				Precision maxY=-1;
				int maxJ=-1;
				int maxI=-1;
				for(int j=nf0; j<len; j++)
				{
					vector<double>& score=nodes.at(j).score;
					for(int i=1; i<score.size(); i++)
					{
						Precision ty=score.at(i)-thresholds->at(i-1);
						if(ty>maxY)
						{
							int tSegLen=j+2-i-nf0;
							if(tSegLen>0)
							{
								maxY=ty;
								maxJ=j;
								maxI=i;
							}
						}
					}
				}
				segLen=maxJ+2-maxI-nf0;
			}
		}

		if(segLen>0)
		{
			boundary=segLen-1;
		}
	}

	if(boundary<0 && !nodes.empty() && final)
	{
		boundary=nodes.size()-1;
	}

	return boundary;

}

bool Segmentor::getSegment(bool final, vector<string>& words)
{
	int position=findBoundary(final);

	bool found=false;
	if(position>=0)
	{
		found=true;

		words.clear();
		int endPos=position+1; //break after this word;
		for(int i=0;i<endPos;i++)
		{
			words.push_back(nodes.at(i).word);
		}
		nodes.erase(nodes.begin(),nodes.begin()+endPos);

		this->updateScores(0);
	}
	return found;
}



} /* namespace simSeg */
