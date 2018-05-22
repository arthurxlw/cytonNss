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

#include "WeightFactory.h"
#include "Global.h"
#include "cublasWrapper.h"
#include "utils.h"

namespace cytonLib {

WeightFactory weightFactory;

void WeightFactory::init(const string& method)
{
	if(method=="adam")
	{
		optAdam=true;
		adamGamma=0.9;
		adamGamma2=0.999999;
		adamEpsilon=1e-9;
	}
	else if(method=="SGD")
	{
		optSgd=true;
	}
	else
	{
		assert(false);
	}

}

void WeightFactory::create(Weight& weight, string tag, int ni, int nj)
{
	weight.create(tag, ni, nj);
	weights.push_back(&weight);
}

void WeightFactory::alloc(Precision clipGradient)
{
	int length=0;
	for(int i=0;i<weights.size();i++)
	{
		Weight& w=*weights.at(i);
		fprintf(stderr, "weight%d %s %d*%d\n", i, w.tag.c_str(), w.ni, w.nj);
		length+=w.length();
	}
	whole.resize(length, 1);
	whole.clipGrad=clipGradient;
	fprintf(stderr, "totalWeight %d\n",length);

	int offset=0;
	for(vector<Weight*>::iterator iw=weights.begin();iw!=weights.end();iw++)
	{
		Weight& w=*(*iw);
		w.set(w.ni, w.ni, w.nj, whole.data+offset, whole.grad.data+offset);
		offset+=w.length();
	}


	whole.initRandom(-global.initFactor, global.initFactor);
	if(optAdam)
	{
		momentum.resize(whole.ni, whole.nj);
		momentum.setZero();
		gradientVariance.resize(whole.ni, whole.nj);
		gradientVariance.setZero();
		dWeight.resize(whole.ni, whole.nj);
	}
	else if(optSgd)
	{
	}
	else
	{
		assert(false);
	}

}

void WeightFactory::clearGrad()
{
	whole.grad.setZero();
}


__global__
void weightFactory_update_adam(Precision* grad, Precision* gradMomentum, Precision* gradVar,  Precision* weight, Precision* dWeight, int len,
		Precision gamma,Precision gamma2, Precision epsilon, Precision lambda )
{
	int i=blockDim.x*blockIdx.x+threadIdx.x;
	if(i<len)
	{
		Precision& g=grad[i];
		Precision& gm=gradMomentum[i];
		Precision& gv=gradVar[i];
		Precision& w=weight[i];
		Precision& dw=dWeight[i];

		gm=(1-gamma)*g+gamma*gm;
		gv=(1-gamma2)*g*g+gamma2*gv;

		dw= gm/(sqrt(gv)+epsilon)*lambda;
		w += dw;
	}
}


void WeightFactory::update(Precision lambda)
{
	int len=whole.length();
	Precision pnFactor=sqrt(1.0/whole.length());

	if(whole.clipGrad>0)
	{
		whole.grad.clip(whole.clipGrad);
	}
	if(optAdam)
	{
		Precision step=global.batch;
		Precision tf=sqrt(1.0-std::pow(adamGamma2, step)) / (1.0-std::pow(adamGamma, step));
		weightFactory_update_adam<<<ceil(len, blockSize), blockSize>>>(whole.grad.data, momentum.data, gradientVariance.data, whole.data, dWeight.data, len,
				adamGamma, adamGamma2, adamEpsilon, lambda*tf);
	}
	else if(optSgd)
	{
		checkError(cublasXaxpy(global.cublasHandle, whole.length(), &lambda, whole.grad.data, 1, whole.data, 1));
	}
	else
	{
		assert(false);
	}
}

void WeightFactory::save(const string& fileName)
{
	XLLib::dirPrepare4file(fileName);
	std::ofstream f(fileName.c_str());

	f<<"##"<<"WeightFactory"<<"\n";
	whole.save(f);
	f.close();
}

void WeightFactory::load(const string& fileName)
{
	if(!XLLib::fileExists(fileName))
	{
		fprintf(stderr, "Error: model file %s does not exist.\n", fileName.c_str());
		assert(false);
	}

	ifstream f(fileName.c_str());
	string tTag=string("##WeightFactory");
	checkFile(f,tTag);
	whole.load(f);
	f.close();
}

} /* namespace cytonLib */
