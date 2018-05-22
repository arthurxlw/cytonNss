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

#include "NetworkNss.h"
#include "WeightFactory.h"
#include "ParamsNss.h"

namespace sentSeg
{

extern Vocabulary vocab;



void NetworkNss::init(BatchNss& batch_)
{
	batch=&batch_;
	hx=&batch_.x;
	hy=&batch_.y;

	int nf=params.numFutureWords;
	int nf0=1;
	assert(nf>=nf0);
	targetDim=nf-nf0+2;

	x.copyFrom(*hx);
	Variable* tx=embedding.init("embedding", &x, hx, params.vocabSize, params.hiddenSize);

	tx = lstm.init("lstm", tx, false, params.hiddenSize, params.numLayers, params.dropout);
	lstm.hy.grad.setZero();
	lstm.cy.grad.setZero();

	tx = reshape.init("", tx, tx->length()/params.hiddenSize, params.hiddenSize, 1, 1 );

	tx = dropOut.init("dropout", tx, params.dropout);

	tx=linear.init("linear", tx, targetDim, true);

	tx = softmax.init("softmax", tx);

	assert(tx->n == batch_.y.length());

	weightFactory.init("SGD");
	weightFactory.alloc(5.0);
}


void NetworkNss::forward()
{
	x.copyFrom(*hx);
	embedding.forward();

	lstm.forward();
	{
		Variable*tx =&lstm.y;
		reshape.forward(tx->length()/params.hiddenSize, params.hiddenSize, 1, 1);
	}

	dropOut.forward();

	linear.forward();

	softmax.forward();

	predY.copyFrom(softmax.y);

}

Precision NetworkNss::setTarget()
{
	assert(predY.ni==targetDim);
	int& numTargets=batch->numTargets;
	grad.resize(predY.ni, predY.nj);
	Precision score=0;
	Precision scale=1.0/params.batchSize;

	HostMatInt& targets=*hy;
	//set grad
	assert(grad.nj == targets.length());
	numTargets=0;
	for(int j=0; j<targets.length(); j++)
	{
		int target=targets.at(j);
		for(int i=0; i<predY.ni; i++)
		{
			Precision tPred=predY.at(i,j);
			Precision& tGrad=grad.at(i,j);
			if(target>=0 )
			{
				assert(target<predY.ni);
				if(target==i)
				{
					tGrad=1-tPred;
					score += log(tPred+1e-9);
					numTargets+=1;
				}
				else
				{
					tGrad=-tPred;
				}
				tGrad *=scale;
			}
			else
			{
				tGrad=0;
			}
		}
	}
	assert(softmax.x->length()==grad.length());
	softmax.x->grad.copyFrom(grad.data, grad.length());
	return score;
}

Precision NetworkNss::backward4learn()
{
	Precision score=setTarget();

	linear.backward();

	dropOut.backward();

	lstm.backward();

	embedding.backward();

	return score;
}

void NetworkNss::calculateGradient()
{
	embedding.calculateGradient();

	lstm.calculateGradient();

	linear.calculateGradient();
}


Precision NetworkNss::learn()
{
	assert(cytonLib::testMode==false);

	forward();

	Precision score=backward4learn();

	weightFactory.whole.grad.setZero();
	calculateGradient();

	weightFactory.update(params.learningRate);

	return score;

}

Precision NetworkNss::getScore()
{
	forward();
	Precision score=setTarget();
	return score;
}


void NetworkNss::apply(int* wids, int ni_, int nj_)
{
	hx->copyFrom(wids, ni_, nj_);
	this->forward();
}

void NetworkNss::setContext(bool reset)
{
	if(reset)
	{
		lstm.hx.setZero();
		lstm.cx.setZero();
	}
	else
	{
		lstm.hx.copyFrom(lstm.hy);
		lstm.cx.copyFrom(lstm.cy);
	}

}

void NetworkNss::apply(SampleSi& s, vector<double>& thresholds, ostream* os)
{
	int nf=params.numFutureWords;
	int nf0=1;
	assert(thresholds.size()==nf-nf0+1);

	int maxLen=params.maxSentLen;
	int start=0;
	vector<int>& input=s.srcWids;
	vector<int>& boundaries=s.boundaries;
	vector<bool>& segByMax=s.segByMax;
	double latency=0;
	boundaries.clear();
	segByMax.clear();
	HostMatPrec& probs=predY;
	while(start<input.size())
	{
		int len=std::min(maxLen, (int)input.size()-start);
		int segLen=-1;
		double tLatency=0;
		bool tSegByMax=false;

		{
			this->apply(&input[start], 1, len);
			assert(probs.ni==nf-nf0+2 && probs.nj==len);
			for(int j=nf0; j<len; j++)
			{
				for(int i=probs.ni-1; i>=1; i--)
				{
					Precision tProb=probs.at(i, j);
					if(tProb>=thresholds.at(i-1))
					{
						int tSegLen=j+2-i-nf0;
						if(tSegLen>0)
						{
							segLen=tSegLen;
							tLatency=(segLen-1)*segLen/2+(i-1+nf0)*segLen;
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
						for(int i=1; i<probs.ni; i++)
						{
							Precision ty=probs.at(i, j)-thresholds.at(i-1);
							if(ty>maxY)
							{
								int tSegLen=j+2-i-nf0;
								if(tSegLen>=0)
								{
									maxY=ty;
									maxJ=j;
									maxI=i;
								}
							}
						}
					}
					segLen=maxJ+2-maxI-nf0;
					tLatency=(segLen-1)*segLen/2+(len-segLen)*segLen;
					tSegByMax=true;
				}
				else
				{
					segLen=len;
					tLatency= (segLen-1)*segLen/2;
				}
			}
		}
		start+=segLen;
		boundaries.push_back(start);
		segByMax.push_back(tSegByMax);
		latency+=tLatency;
	}

	latency /=input.size();

	s.latency=latency;
}

} /* namespace sentSeg */
