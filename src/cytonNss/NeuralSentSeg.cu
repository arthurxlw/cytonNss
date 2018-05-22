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

#include "NeuralSentSeg.h"
#include "EvaluateSegment.h"
#include "stdlib.h"
#include <list>
#include "Segmentor.h"
using std::list;

namespace sentSeg
{
extern Vocabulary vocab;

void NeuralSentSeg::work()
{
	if(params.mode=="train")
	{
		workTrain();
	}
	else if(params.mode=="apply")
	{
		workApply();
	}
	else if(params.mode=="tune")
	{
		workTune();
	}
	else
	{
		fprintf(stderr, "unknown params.mode %s", params.mode.c_str());
		assert(false);
	}

}

void NeuralSentSeg::workTrain()
{
	XLLibTime startTime=XLLib::startTime();

	vocab.load(params.vocabFile, params.vocabSize);
	if(params.vocabSize!=vocab.size())
	{
		fprintf(stderr, "change vocobSize %d => %d\n", params.vocabSize, vocab.size());
		params.vocabSize=vocab.size();
	}
	batch.init();
	network.init(batch);

	CorpusNss corpus;
	corpus.init(params.trainCorpus, &vocab, params.batchSize);

	CorpusNss  devCorpora;
	devCorpora.init(params.devCorpus, &vocab, params.batchSize);

	int epoch=0;
	Precision trainScore=0;
	int trainNum=0;
	cytonLib::testMode=false;
	Precision bestDevScore=-1000;
	string bestModel;
	corpus.shuffleReset();
	bool corpusTraining=false;
	int numFails=0;

	if(params.loadModel!="")
	{
		bestModel=params.loadModel;
		this->loadModel(bestModel);
		XLLib::printf("loadModel %s", bestModel.c_str());
		Precision tScore=testCorpus(devCorpora);
		bestDevScore=std::max(tScore, bestDevScore);
	}
	if(params.saveModel!="")
	{
		saveModelSetting(params.saveModel);
	}
	int iTest=0;
	bool decay=false;
	XLLib::printf("#Epoch Update Time LearningRate, likelihood Train Dev\n");
	for(int iBatch=0; ; iBatch++)
	{
		assert(testMode==false);

		bool epochEnd;
		bool tRead=corpus.read(batch, corpusTraining);
		if(!tRead)
		{
			epochEnd=true;
			epoch +=1;
			corpus.shuffleReset();
			tRead=corpus.read(batch, corpusTraining);
		}
		else
		{
			epochEnd=false;
		}

		Precision tScore=network.learn();
		trainScore +=tScore;
		trainNum+=batch.getNumTarget();

		if(epochEnd)
		{
			XLLib::printf("#%d %d %s %.2e, likelihood %.3e", epoch, iBatch, XLLib::endTime(startTime).c_str(),
					 params.learningRate, trainScore/trainNum);
			iTest +=1;
			bool success=false;
			string tModel;
			Precision tScore=testCorpus(devCorpora);
			tModel=XLLib::stringFormat("%s/model_e%d_s%d_%.3f", params.saveModel.c_str(), epoch, iBatch, tScore);
			success=(tScore >= bestDevScore+1e-3);
			bool successRelax=(tScore >= bestDevScore+1e-3);
			bestDevScore=std::max(tScore, bestDevScore);

			if(successRelax)
			{
				weightFactory.save(tModel);
				numFails=0;
				XLLib::fileLink(XLLib::fileName(tModel), XLLib::stringFormat("%s/model", params.saveModel.c_str()));
				bestModel=tModel;
				XLLib::printf(" s");
			}

			if(!success)
			{
				decay=true;
				numFails+=1;
				XLLib::printf(" f%d", numFails);
				if(numFails>=3)
				{
					printf("\n");
					break;
				}
				if(!successRelax)
				{
					weightFactory.load(bestModel);
					printf(" load %s\n", XLLib::fileName(bestModel).c_str());
				}
			}

			if(decay)
			{
				params.learningRate *=params.learnRateDecay;
			}

			printf("\n");
			trainScore=0;
			trainNum=0;
		}
	}

	printf("\nbestModel %s\n", bestModel.c_str());

}

void NeuralSentSeg::workApply()
{
	cytonLib::testMode=true;
	cytonLib::batchSize=1;
	params.batchSize=1;
	vector<double>& thresholds=params.thresholds;
	int nf=params.numFutureWords;
	int nf0=1;
	assert(thresholds.size()==nf-nf0+1);

	loadModel(params.loadModel);

	bool gbTestMode=true;
	std::swap(gbTestMode, cytonLib::testMode);
	EvaluateSegment evalSeg;
	XLLibTime startTime=XLLib::startTime();

	Segmentor segmentor;
	segmentor.init(&network, &vocab, &thresholds);

	const string& inputFile=params.input;
	const string& outputFile=params.output;

	FILE* f;
	if(inputFile!="stdin")
	{
		f=fopen(inputFile.c_str(),"r");
	}
	else
	{
		f=stdin;
	}
	if (f == NULL)
	{
		fprintf(stderr, "cannot open the input file %s", inputFile.c_str());
		exit(1);
	}

	FILE* g;
	if(outputFile!="stdout")
	{
		g=fopen(outputFile.c_str(),"w");
	}
	else
	{
		g=stdout;
	}

	char * line = NULL;
	size_t len = 0;
	vector<string> words;
	vector<string> segment;
	while(true)
	{
		ssize_t read = getline(&line, &len, f);
		int tLen=strlen(line);
		while(tLen>0 && line[tLen-1]=='\n' )
		{
			line[tLen-1]=0;
			tLen-=1;
		}

		bool isEnd=false;
		if(read!=-1)
		{
			XLLib::str2list(line, words);
			segmentor.putWords(words);
			isEnd=words.empty();
		}
		else
		{
			isEnd=true;
		}

		while(segmentor.getSegment(isEnd, segment))
		{
			for(unsigned int i=0; i<segment.size(); i++)
			{
				if(i!=0)
				{
					fprintf(g, " ");
				}
				fprintf(g, "%s", segment.at(i).c_str());
			}
			fprintf(g,"\n");
		}
		if(isEnd)
		{
			fprintf(g, "\n");
		}

		if(read==-1)
		{
			break;
		}
	}
	fclose(f);
	fclose(g);
	std::swap(gbTestMode, cytonLib::testMode);
}


void NeuralSentSeg::workTune()
{
	XLLibTime startTime=XLLib::startTime();
	cytonLib::testMode=true;
	cytonLib::batchSize=1;
	params.batchSize=1;

	vector<double>& thresholds=params.thresholds;
	int nf=params.numFutureWords;
	int nf0=1;
	assert(thresholds.size()==nf-nf0+1);

	loadModel(params.loadModel);

	CorpusSi corpora;
	corpora.init(params.devCorpus);

	int nTh=thresholds.size();
	double dTh=0.1;
	double minTh=0.2;
	std::multimap<double, vector<double>> que;
	que.insert(std::pair<double, vector<double>>(0.0, thresholds));
	std::set<vector<double>> dict;
	vector<double> scores;
	vector<string> details;
	double bestScore=-1000;
	string bestLog;
	int tuneSteps=params.tuneSteps;
	int testPeriod=100;
	for(int is=0; ; is++)
	{
		bool searchEnd= (tuneSteps>0 && is>=tuneSteps);

		vector<double> ths;
		if(!searchEnd)
		{
			searchEnd=true;
			while(!que.empty())
			{
				std::multimap<double,vector<double>>::iterator it=que.begin();
				ths.assign(it->second.begin(), it->second.end());
				que.erase(it);

				if(dict.find(ths)==dict.end())
				{
					searchEnd=false;
					break;
				}
			}
		}

		if( (is!=0 && is%testPeriod==0) || searchEnd)
		{
			printf("\n# %2d %d %d\n bestScore %.4f bestLog %s\n", is,
					que.size(), dict.size(),
					bestScore, bestLog.c_str());
			string tFile=XLLib::stringFormat("%s/resTune.%d", params.output.c_str(), is);
			printf("\nwriting %s\n", tFile.c_str());
			vector<string> tDetails(details);
			XLLib::sort(scores, tDetails);
			XLLib::dirPrepare4file(tFile);
			XLLib::writeFile(tFile, tDetails);
		}
		if(searchEnd)
		{
			break;
		}

		double tScore=tuneApply(corpora, ths,  bestScore, bestLog, scores, details);
		assert(scores.size()==details.size());
		dict.insert(ths);

		if(tScore>=bestScore-params.scoreTolerance)
		{
			XLLib::printf(" *");
			for(int k=nTh-1; k>=0; k--)
			{
				for(int k1=0; k1<2; k1++)
				{
					double& th=ths.at(k);
					double dTh1= (k1==0?-1:1)*dTh;
					double th1=th+dTh1;
					if(th1>=minTh && th1<=1.0)
					{
						th+=dTh1;
						bool valid=true;
						for(int it=0; it<nTh-1; it++)
						{
							if(ths.at(it)<ths.at(it+1))
							{
								valid=false;
								break;
							}
						}

						if(valid && dict.find(ths)==dict.end())
						{
							que.insert(std::pair<double, vector<double> >(-tScore, ths));
						}
						th-=dTh1;
					}
				}
			}
		}
		printf("\n");


	}
	printf("\nEnd bestScore %.4f %s\n", bestScore, bestLog.c_str());
}

double NeuralSentSeg::tuneApply(CorpusSi& corpus, vector<double>& thresholds,
		double& bestScore, string& bestLog, vector<double>& scores, vector<string>& details)
{
	int nf=params.numFutureWords;
	int nf0=1;
	assert(thresholds.size()==nf-nf0+1);

	bool gbTestMode=true;
	std::swap(gbTestMode, cytonLib::testMode);
	EvaluateSegment evalSeg;
	vector<string> res;
	ostringstream os;

	XLLib::printf(true, &os, "%s", XLLib::toString_vec(thresholds, "%.2f",":").c_str());
	XLLib::printf(true, &os, " ");

	XLLibTime startTime=XLLib::startTime();
	ofstream* ptrG=NULL;
	evalSeg.reset();
	for(int is=0; is<corpus.samples.size(); is++)
	{
		if(is%10==0)XLLib::printf(".");
		SampleSi& s=corpus.samples.at(is);
		network.apply(s, thresholds, ptrG);
		evalSeg.apply(s, ptrG);
	}
	vector<double> tScores;
	double score=evalSeg.getResult(&os, &tScores);
	scores.push_back(score);
	details.push_back(os.str());
	if(score>bestScore)
	{
		bestScore=score;
		bestLog=os.str();
	}
	std::swap(gbTestMode, cytonLib::testMode);
	return score;
}

void updateDict(std::map<string,int>& dict, const string& key)
{
	std::map<string, int>::iterator it=dict.find(key);
	if(it==dict.end())
	{
		dict[key]=1;
	}
	else
	{
		it->second +=1;
	}
}

void printDict(std::map<string,int>& dict)
{
	vector<string> keys;
	vector<int> vals;
	for(std::map<string, int>::iterator it=dict.begin(); it!=dict.end(); it++)
	{
		keys.push_back(it->first);
		vals.push_back(it->second);
	}

	vector<int> idxs;
	XLLib::sortIndex(vals, idxs);
	for(int k=0; k<idxs.size(); k++)
	{
		int idx=idxs.at(k);
		printf("%s %d\n", keys.at(idx).c_str(), vals.at(idx));
	}
}

Precision NeuralSentSeg::testCorpus(CorpusNss& corpus)
{
	bool gbTestMode=true;
	std::swap(gbTestMode, cytonLib::testMode);
	corpus.kIndex=0;
	int ns=0;
	Precision score=0;
	while((corpus.read(batch, false)))
	{
		score += network.getScore();
		ns+=batch.getNumTarget();
	}
	score /=ns;

	XLLib::printf(" %.3e", score);
	std::swap(gbTestMode, cytonLib::testMode);
	return score;
}


void NeuralSentSeg::saveModelSetting(const string& modelDir)
{
	string tVocab=modelDir+"/vocab";
	XLLib::dirPrepare4file(tVocab);
	vocab.save(tVocab);
	params.saveModelSetting(modelDir);
}

void NeuralSentSeg::loadModel(const string& modelName)
{
	if(params.loadModel=="")
	{
		fprintf(stderr, "loadModel is empty %s\n", params.loadModel.c_str());
		exit(1);
	}
	else
	{
		int i=params.loadModel.rfind('/');
		string modelDir=params.loadModel.substr(0,i);

		string tFile=modelDir+"/vocab";
		vocab.load(tFile, 0);
		if(params.vocabSize!=vocab.size())
		{
			params.vocabSize=vocab.size();
		}
		batch.init();
		network.init(batch);
		weightFactory.load(params.loadModel);
		fprintf(stderr, "loadWeights %s\n", modelName.c_str());
	}

}

} /* namespace sentSeg */
