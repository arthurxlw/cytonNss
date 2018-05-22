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

#include "BatchNss.h"
#include "Vocabulary.h"

namespace sentSeg
{
extern Vocabulary vocab;

void BatchNss::init()
{
	int batchSize=params.batchSize;
	int maxSentLen=params.maxSentLen;
	int nf=params.numFutureWords;

	int maxLen=maxSentLen+nf;

	x.resize(batchSize, maxLen);
	x.setValue(vocab.empty);
	x.enlarge=false;
	y.resize(batchSize, maxLen );
	y.enlarge=false;
}

int BatchNss::getNumTarget()
{
	int res=0;
	for(int i=0; i<y.length(); i++)
	{
		if(y.at(i)>=0)
		{
			res+=1;
		}
	}
	return res;
}

} /* namespace sentSeg */
