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

#include "ParamsNss.h"
#include "Global.h"

namespace sentSeg
{

ParamsNss::ParamsNss()
{
	const Option options[] = {
			{"mode", "train", "Running mode: train/tune/apply"},
			{"loadModel", "", "Load model from a model directory"},
			{"saveModel", "", "Save model to a model directory"},
			{"vocabSize", "0", "Size of the vocabulary. 0 means unlimited."},
			{"hiddenSize", "512", "Size of hidden states"},
			{"numLayers", "3", "Number of layers"},
			{"optimizer",  "SGD",	"Optimizer"},
			{"learningRate",  "1.0",	"Learning rate"},
			{"learnRateDecay",  "0.5",	"Decay rate of learning rate"},
			{"dropout",  "0.5",	"Dropout rate"},
			{"maxSentLen",  "40",	"Maximun length of sentence"},
			{"numFutureWords", "6", "Maximun number of future words"},
			{"vocabFile", "trainFile.vocab", "Vocabulary File"},
			{"train", "trainFile", "Training file. One sentence per line."},
			{"dev", "dev", "Ddevelopment file. One sentence per line."},
			{"batchSize", "64", "Batch size"},
			{"input", "", "Input File of the apply mode. stdin means standard input"},
			{"output", "", "Output File of the apply mode or the tune mode. stdin means standard output"},
			{"thresholds", "0.9:0.8:0.7:0.6:0.5:0.4", "Thresholds"},
			{"scoreTolerance", "0.04", "Tolerance of scores in the tune mode"},
			{"tuneSteps", "0", "Maximum number of tune steps. O means infinite."},
			{"factorLatency", "0.01", "Factory of latency in evaluation: score=f1-factorLatency*latency."},
			{"numSents", "10", "Number of sentences as one input in the tune mode"},
			{"","",""}
	};

	addOptions(options);
}

void ParamsNss::init_members()
{
	fprintf(stderr, "Parameters:\n");
	mode=get("mode");
	saveModel=get("saveModel");
	loadModel=get("loadModel");
	vocabSize=geti("vocabSize");
	hiddenSize=geti("hiddenSize");
	numLayers=geti("numLayers");
	learningRate=getf("learningRate");
	learnRateDecay=getf("learnRateDecay");
	dropout=getf("dropout");
	maxSentLen=geti("maxSentLen");
	numFutureWords=geti("numFutureWords");
	batchSize=geti("batchSize");
	vocabFile=get("vocabFile");
	trainCorpus=get("train");
	devCorpus=get("dev");

	input=get("input");
	output=get("output");

	string tThresholds=get("thresholds");
	thresholds.clear();
	XLLib::str2doubles(tThresholds, ":", thresholds);

	scoreTolerance=getf("scoreTolerance");
	tuneSteps=geti("tuneSteps");
	factorLatency=getf("factorLatency");
	numSents=geti("numSents");

	if(loadModel!="")
	{
		loadModelSetting(loadModel);
	}
}

void ParamsNss::saveModelSetting(const string& fileName)
{
	std::ofstream f(fileName+"/setting");
	f<<hiddenSize<<" "<<numLayers<<" "<<vocabSize<<"\n";
	f<<numFutureWords<<"\n";
	f.close();
}

void ParamsNss::loadModelSetting(const string& fileName)
{
	string modelDir=XLLib::dirName(fileName);
	std::ifstream f(modelDir+"/setting");
	f>>hiddenSize;
	f>>numLayers;
	f>>vocabSize;
	f>>numFutureWords;
	f.close();

	fprintf(stderr, "loadModelSetting hiddenSize %d numLayers %d  vocabSize %d numFutureWords %d\n",
			hiddenSize, numLayers, vocabSize,
			numFutureWords);
}


ParamsNss params;
} /* namespace sentSeg */
