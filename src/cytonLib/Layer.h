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

#ifndef _CYTONLIB_LAYER_H_
#define _CYTONLIB_LAYER_H_

#include "Variable.h"

namespace cytonLib
{

class Layer
{
public:
	string tag;
	bool addGrad;
	Variable* x;
	Variable y;

	Layer();

	virtual void forward()
	{
	};

	virtual void backward()
	{
	};

	virtual void calculateGradient()
	{
	};

};

} /* namespace cytonLib */

#endif /* LAYER_H_ */
