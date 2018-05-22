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

#ifndef NETWORKNSS_H_
#define NETWORKNSS_H_

#include "basicHeadsNss.h"
#include "Network.h"
#include "EmbeddingLayer.h"
#include "LstmLayer.h"
#include "ReshapeLayer.h"
#include "CuDropoutLayer.h"
#include "LinearLayer.h"
#include "SoftmaxLayer.h"
#include "BatchNss.h"
#include "HostMatReal.h"
#include "Vocabulary.h"
#include "SampleSi.h"

namespace sentSeg
{

class NetworkNss: public cytonLib::Network
{
public:
	BatchNss* batch;
	HostMatInt* hx;
	DevMatInt x;
	HostMatInt* hy;
	int targetDim;
	HostMatPrec predY;
	HostMatPrec grad;
	EmbeddingLayer embedding;
	LstmLayer lstm;
	ReshapeLayer reshape;
	CuDropoutLayer dropOut;
	LinearLayer linear;
	SoftmaxLayer softmax;

	void init(BatchNss& batch);

	void forward();

	Precision learn();

	virtual Precision getScore();

	void setContext(bool reset);

	void apply(int* wids, int ni, int nj);

	void apply(SampleSi& s, vector<double>& thresholds, ostream* os);

protected:
	Precision setTarget();

	Precision setTarget_labelSmooth();

	Precision backward4learn();

	void calculateGradient();

};

} /* namespace sentSeg */

#endif /* NETWORKNSS_H_ */
