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

#include "ReshapeLayer.h"
#include "Global.h"

namespace cytonLib
{

Variable* ReshapeLayer::init(const string& tag_, Variable* x_,
		int n, int c, int h, int w)
{
	tag=tag_;
	x=x_;

	this->forward(n, c, h, w);
	return &y;
}

void ReshapeLayer::forward(int n, int c, int h, int w)
{
	assert(x->length() == n*c*h*w);
	y.set(n, c, h, w, x->data, x->grad.data);
}

void ReshapeLayer::forward()
{
}


} /* namespace cythonLib */
