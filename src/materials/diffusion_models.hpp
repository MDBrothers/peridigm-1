//! \file diffusion_models.hpp

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
// Michael L. Parks      mlparks@sandia.gov
// Stewart A. Silling    sasilli@sandia.gov
//
// ************************************************************************
//@HEADER

#ifndef DIFUSSION
#define DIFFUSION

#include "Peridigm_InfluenceFunction.hpp"

namespace MATERIAL_EVALUATION {

  template<typename ScalarT>
  void computePhaseOneDensityInPores
  (
    ScalarT* phaseOneDensityInPores,
    const ScalarT* porePressureYOverlap,
    const double* deltaTemperature,
    int numOwnedPoints
  );

  template<typename ScalarT>
  void computePhaseOneDensityInFracture
  (
    ScalarT* phaseOneDensityInFracture,
    const ScalarT* fracturePressureYOverlap,
    const double* deltaTemperature,
    int numOwnedPoints
  );

  namespace ONE_PHASE{

    typedef PeridigmNS::InfluenceFunction::functionPointer FunctionPointer;

    //! Computes contributions to the internal flow resulting from owned points.
    template<typename ScalarT>
    void computeInternalFlow
    (
      const ScalarT* yOverlap,
      const ScalarT* porePressureYOverlap,
      const double* porePressureVOverlap,
      const ScalarT* fracturePressureYOverlap,
      const double* fracturePressureVOverlap,
      const double* volumeOverlap,
      const double* damage,
      const ScalarT* matrixPorosityNP1,
      const double* matrixPorosityN,
      const ScalarT* fracturePorosityNP1,
      const double* fracturePorosityN,
      const ScalarT* phaseOneDensityInPoresNP1,
      const double* phaseOneDensityInPoresN,
      const ScalarT* phaseOneDensityInFractureNP1,
      const double* phaseOneDensityInFractureN,
      const ScalarT* breaklessDilatationOwned,
      ScalarT* phaseOnePoreFlowOverlap,
      ScalarT* phaseOneFracFlowOverlap,
      const int* localNeighborList,
      const int numOwnedPoints,
      const double m_matrixPermeabilityXX,
      const double m_matrixPermeabilityYY,
      const double m_matrixPermeabilityZZ,
      const double m_phaseOneViscosity,
      const double m_horizon,
      const double m_horizon_fracture,
      const double deltaTime,
      const double* deltaTemperature = 0
    );

    //! Computes contributions to the internal flow resulting from owned points.
    // This is not an explicit template sepcialization because with finite difference
    // velocities are perturbed. Meaning the template cannot match the reguirements.
    void computeInternalFlowComplex
    (
      const std::complex<double> * yOverlap,
      const std::complex<double> * porePressureYOverlap,
      const std::complex<double> * porePressureVOverlap,
      const std::complex<double> * fracturePressureYOverlap,
      const std::complex<double> * fracturePressureVOverlap,
      const double* volumeOverlap,
      const double* damage,
      const double* principleDamageDirection,
      const std::complex<double> * matrixPorosityNP1,
      const double * matrixPorosityN,
      const std::complex<double> * fracturePorosityNP1,
      const double * fracturePorosityN,
    	const std::complex<double> * phaseOneDensityInPoresNP1,
    	const double * phaseOneDensityInPoresN,
    	const std::complex<double> * phaseOneDensityInFractureNP1,
    	const double * phaseOneDensityInFractureN,
      const std::complex<double>* breaklessDilatation,
      std::complex<double> * phaseOnePoreFlowOverlap,
      std::complex<double> * phaseOneFracFlowOverlap,
      const int*  localNeighborList,
      const int numOwnedPoints,
      const double m_matrixPermeabilityXX,
      const double m_matrixPermeabilityYY,
      const double m_matrixPermeabilityZZ,
      const double m_phaseOneViscosity,
      const double m_horizon,
      const double m_horizon_fracture,
      const double deltaTime,
      const double* deltaTemperature
    );

  }

  namespace TWO_PHASE{

    typedef PeridigmNS::InfluenceFunction::functionPointer FunctionPointer;

    //! Computes contributions to the internal flow resulting from owned points.
    template<typename ScalarT>
    void computeInternalFlow
    (
      const ScalarT* yOverlap,
      const ScalarT* porePressureYOverlap,
      const double* porePressureVOverlap,
      const ScalarT* fracturePressureYOverlap,
      const double* fracturePressureVOverlap,
      const ScalarT* phaseOneSaturationPoresYOverlap,
      const double* phaseOneSaturationPoresVOverlap,
      const ScalarT* phaseOneSaturationFracYOverlap,
      const double* phaseOneSaturationFracVOverlap,
      const double* volumeOverlap,
      const double* damage,
      const ScalarT* matrixPorosityNP1,
      const double* matrixPorosityN,
      const ScalarT* fracturePorosityNP1,
      const double* fracturePorosityN,
      const ScalarT* phaseOneDensityInPoresNP1,
      const double* phaseOneDensityInPoresN,
      const ScalarT* phaseOneDensityInFractureNP1,
      const double* phaseOneDensityInFractureN,
      const ScalarT* phaseTwoDensityInPoresNP1,
      const double* phaseTwoDensityInPoresN,
      const ScalarT* phaseTwoDensityInFractureNP1,
      const double* phaseTwoDensityInFractureN,
      const ScalarT* breaklessDilatationOwned,
      ScalarT* phaseOnePoreFlowOverlap,
      ScalarT* phaseOneFracFlowOverlap,
      ScalarT* phaseTwoPoreFlowOverlap,
      ScalarT* phaseTwoFracFlowOverlap,
      const int*  localNeighborList,
      const int numOwnedPoints,
      const double m_matrixPermeabilityXX,
      const double m_matrixPermeabilityYY,
      const double m_matrixPermeabilityZZ,
      const double m_phaseOneViscosity,
      const double m_phaseTwoViscosity,
      const double m_horizon,
      const double m_horizon_fracture,
      const double deltaTime,
      const double* deltaTemperature = 0
    );

    //! Computes contributions to the internal flow resulting from owned points.
    void computeInternalFlowComplex
    (
      const std::complex<double>* yOverlap,
      const std::complex<double>* porePressureYOverlap,
      const std::complex<double>* porePressureVOverlap,
      const std::complex<double>* fracturePressureYOverlap,
      const std::complex<double>* fracturePressureVOverlap,
      const std::complex<double>* phaseOneSaturationPoresYOverlap,
      const std::complex<double>* phaseOneSaturationPoresVOverlap,
      const std::complex<double>* phaseOneSaturationFracYOverlap,
      const std::complex<double>* phaseOneSaturationFracVOverlap,
      const double* volumeOverlap,
      const double* damage,
      const std::complex<double>* matrixPorosityNP1,
      const double* matrixPorosityN,
      const std::complex<double>* fracturePorosityNP1,
      const double* fracturePorosityN,
      const std::complex<double>* phaseOneDensityInPoresNP1,
      const double* phaseOneDensityInPoresN,
      const std::complex<double>* phaseOneDensityInFractureNP1,
      const double* phaseOneDensityInFractureN,
      const std::complex<double>* phaseTwoDensityInPoresNP1,
      const double* phaseTwoDensityInPoresN,
      const std::complex<double>* phaseTwoDensityInFractureNP1,
      const double* phaseTwoDensityInFractureN,
      const std::complex<double>* breaklessDilatationOwned,
      std::complex<double>* phaseOnePoreFlowOverlap,
      std::complex<double>* phaseOneFracFlowOverlap,
      std::complex<double>* phaseTwoPoreFlowOverlap,
      std::complex<double>* phaseTwoFracFlowOverlap,
      const int*  localNeighborList,
      const int numOwnedPoints,
      const double m_matrixPermeabilityXX,
      const double m_matrixPermeabilityYY,
      const double m_matrixPermeabilityZZ,
      const double m_phaseOneViscosity,
      const double m_phaseTwoViscosity,
      const double m_horizon,
      const double m_horizon_fracture,
      const double deltaTime,
      const double* deltaTemperature
    );

  }

}
#endif // DIFFUSION
