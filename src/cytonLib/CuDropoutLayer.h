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

#ifndef _CYTONLIB_CUDROPOUTLAYER_H_
#define _CYTONLIB_CUDROPOUTLAYER_H_

#include "Global.h"
#include "Variable.h"
#include "Layer.h"


namespace cytonLib
{

class CuDropoutLayer: public Layer
{
public:
	cudnnDropoutDescriptor_t dropoutDesc;
	void* stateSpace;
	size_t stateSize;
	void* reserveSpace;
	size_t reserveSize;
	bool localTestMode;

	Variable* init(string tag_, Variable* x, Precision dropout);
	void forward();
	void backward();
};

} /* namespace reinLearnSentSeg */

#endif /* CUDROPOUTLAYER_H_ */
