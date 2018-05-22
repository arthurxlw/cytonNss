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

#ifndef PARAMSNSS_H_
#define PARAMSNSS_H_

#include "ParametersBase.h"
#include <string.h>

using std::string;

namespace sentSeg
{

class ParamsNss: public xllib::ParametersBase
{
public:
	string mode;
	string saveModel;
	string loadModel;
	int vocabSize;
	int hiddenSize;
	int numLayers;
	float dropout;
	int batchSize;
	float learningRate;
	float learnRateDecay;
	int maxSentLen;
	int numFutureWords;
	string vocabFile;
	string trainCorpus;
	string devCorpus;
	string input;
	string output;
	std::vector<double> thresholds;
	float scoreTolerance;
	int tuneSteps;
	float factorLatency;
	int numSents;

	ParamsNss();

	void init_members();

	void saveModelSetting(const string& fileName);

	void loadModelSetting(const string& fileName);
};

extern ParamsNss params;

} /* namespace sentSeg */

#endif /* PARAMSNSS_H_ */
