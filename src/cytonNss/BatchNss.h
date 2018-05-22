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

#ifndef BATCHNSS_H_
#define BATCHNSS_H_

#include "basicHeadsNss.h"
#include "HostMatrix.h"
using namespace cytonLib;

namespace sentSeg
{

class BatchNss
{
public:
	HostMatInt x;
	HostMatInt y;
	vector<int> lens;
	int numTargets;

	void init();

	int getNumTarget();
};

} /* namespace sentSeg */

#endif /* BATCHNSS_H_ */
