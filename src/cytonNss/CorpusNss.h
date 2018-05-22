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

#ifndef CORPUSNSS_H_
#define CORPUSNSS_H_

#include "basicHeadsNss.h"
#include "BatchNss.h"
#include "Vocabulary.h"
namespace sentSeg
{

class CorpusNss
{
public:
	Vocabulary* vocab;
	vector<vector<int>> sents;
	vector<int> indexes;
	int kIndex;

	void init(const string& corpusFile, Vocabulary* vocab_, int batchSize);

	void reset();

	void shuffleReset();

	bool read(BatchNss& batch, bool training);
};

} /* namespace sentSeg */

#endif /* CORPUSNSS_H_ */