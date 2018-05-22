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

#ifndef _CYTONLIB_NETWORK_H_
#define _CYTONLIB_NETWORK_H_

#include "basicHeads.h"
#include "Layer.h"
#include "Weight.h"

namespace cytonLib
{

class Network: public Layer
{
public:

	void init();

	void forward();

	void backward();

	void calculateGradient();

	virtual Precision getScore();

protected:
	vector<Layer*> layers;
	HostMatPrec hY;
};

} /* namespace cytonLib */

#endif /* NETWORK_H_ */
