/*! \file Peridigm_Compute_Angular_Momentum.cpp */

//@HEADER
// ************************************************************************
//
//                             Peridigm
//                 Copyright (2011) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions?
// David J. Littlewood   djlittl@sandia.gov
// John A. Mitchell      jamitch@sandia.gov
// Michael L. Parks      parks@sandia.gov
// Stewart A. Silling    sasilli@sandia.gov
//
// ************************************************************************
//@HEADER

#include <vector>

#include "Peridigm_Compute_Angular_Momentum.hpp"
#include "../core/Peridigm.hpp"

//! Standard constructor.
PeridigmNS::Compute_Angular_Momentum::Compute_Angular_Momentum(PeridigmNS::Peridigm *peridigm_ ){peridigm = peridigm_;}

//! Destructor.
PeridigmNS::Compute_Angular_Momentum::~Compute_Angular_Momentum(){}


//! Returns the fieldspecs computed by this class
std::vector<Field_NS::FieldSpec> PeridigmNS::Compute_Angular_Momentum::getFieldSpecs() const 
{
  	std::vector<Field_NS::FieldSpec> myFieldSpecs;
  	myFieldSpecs.push_back(Field_NS::FORCE3D);

  	return myFieldSpecs;
}



//! Fill the angular momentum vector
int PeridigmNS::Compute_Angular_Momentum::compute(const int numOwnedPoints,
                                                  const int* ownedIDs,
                                                  const int* neighborhoodList,
                                                  PeridigmNS::DataManager& dataManager) const 
{

	int retval;

  	Teuchos::RCP<Epetra_Vector> velocity,  arm, volume;
  	velocity = dataManager.getData(Field_NS::VELOC3D, Field_ENUM::STEP_NP1);
  	arm      = dataManager.getData(Field_NS::CURCOORD3D, Field_ENUM::STEP_NP1);
  	volume   = dataManager.getData(Field_NS::VOLUME, Field_ENUM::STEP_NONE);

 	// Sanity check
    	if ( (velocity->Map().NumMyElements() != volume->Map().NumMyElements()) ||  (arm->Map().NumMyElements() != volume->Map().NumMyElements()) )
 	{
        	retval = 1;
            	return(retval);
	}
 	
	// Collect values
  	double *volume_values = volume->Values();
  	double *velocity_values = velocity->Values();
  	double *arm_values  = arm->Values();

	// Initialize angular momentum values
  	double angular_momentum_x,  angular_momentum_y, angular_momentum_z;
  	angular_momentum_x = angular_momentum_y = angular_momentum_z = 0.0;

  	// volume is a scalar and force a vector, so maps are different; must do multiplication on per-element basis
  	int numElements = volume->Map().NumMyElements();
  	double vol;
  	for (int i=0;i<numElements;i++) 
  	{
		vol = volume_values[i];
		double v1 = velocity_values[3*i];
    		double v2 = velocity_values[3*i+1];
    		double v3 = velocity_values[3*i+2];
    		double r1 = arm_values[3*i];
    		double r2 = arm_values[3*i+1];
    		double r3 = arm_values[3*i+2];
    		angular_momentum_x = angular_momentum_x + vol*(v2*r3 - v3*r2);
   		angular_momentum_y = angular_momentum_y + vol*(v3*r1 - v1*r3); 
    		angular_momentum_z = angular_momentum_z + vol*(v1*r2 - v2*r1);
  	}

 	// \todo Generalize this for multiple materials
 	double density = peridigm->getMaterialModels()->operator[](0)->Density();

  	//angular_momentum_x = angular_momentum_x*density;
 	//angular_momentum_y = angular_momentum_y*density;
  	//angular_momentum_z = angular_momentum_z*density;

	std::cout << "Hello!" << std::endl;

	return(0);

}