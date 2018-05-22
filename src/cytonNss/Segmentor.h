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

#ifndef _CYTONNSS_SEGMENTOR_H_
#define _CYTONNSS_SEGMENTOR_H_

#include "basicHeadsNss.h"
#include "WordNode.h"
#include "NetworkNss.h"
#include "Vocabulary.h"

namespace sentSeg {

class Segmentor
{
	NetworkNss* network;
	Vocabulary* vocab;
	vector<double>* thresholds;

	vector<WordNode> nodes;


public:

	void init(NetworkNss* network_, Vocabulary* vocab_, vector<double>* thresholds_);

	void putWords(const vector<string>& word);

	bool getSegment(bool final, vector<string>& words);

protected:

	void updateScores(int startPos);

	int findBoundary(bool final);

	vector<WordNode>& getNodes();
};

} /* namespace simSeg */

#endif /* SRC_SIMINTERPRETSEGMENT_SEGMENTORLOGLINEAR_H_ */
