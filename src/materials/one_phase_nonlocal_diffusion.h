//! \file one_phase_nonlocal_diffusion.h

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
#ifndef ONE_PHASE_DIFFUSION_H
#define ONE_PHASE_DIFFUSION_H

#include "Peridigm_InfluenceFunction.hpp"

namespace MATERIAL_EVALUATION {

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
  const double* principleDamageDirection,
  const double* criticalDilatationOwned,
  const ScalarT* breaklessDilatationOwned,
  ScalarT* phaseOnePoreFlowOverlap,
  ScalarT* phaseOneFracFlowOverlap,
  const int*  localNeighborList,
  const int numOwnedPoints,
  const double m_permeabilityScalar,
  const double m_permeabilityCurveInflectionDamage,
  const double m_permeabilityAlpha,
  const double m_maxPermeability,
  const double m_phaseOneBasePerm,
  const double m_phaseOneDensity,
  const double m_phaseOneCompressibility,
  const double m_phaseOneViscosity,
  const double m_horizon,
  const double m_horizon_fracture,
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
  const double* volumeOverlap,
  const double* damage,
  const double* principleDamageDirection,
  const double* criticalDilatationOwned,
  const std::complex<double>* breaklessDilatationOwned,
  std::complex<double>* phaseOnePoreFlowOverlap,
  std::complex<double>* phaseOneFracFlowOverlap,
  const int*  localNeighborList,
  const int numOwnedPoints,
  const double m_permeabilityScalar,
  const double m_permeabilityCurveInflectionDamage,
  const double m_permeabilityAlpha,
  const double m_maxPermeability,
  const double m_phaseOneBasePerm,
  const double m_phaseOneDensity,
  const double m_phaseOneCompressibility,
  const double m_phaseOneViscosity,
  const double m_horizon,
  const double m_horizon_fracture,
  const double* deltaTemperature
);

//! Compute the new matrix Porosity
void computeMatrixPorosityComplex
(
  std::complex<double>* matrixPorosityNP1,
  const std::complex<double>* matrixPorosityN,
  const std::complex<double>* porePressureYOverlapNP1,
  const std::complex<double>* porePressureYOverlapN,
  const std::complex<double>* dilatationNP1,
  const std::complex<double>* dilatationN,
  const double compressibilityRock,
  const double alpha,
  const int* localNeighborList,
  int numOwnedPoints
);

void computeMatrixPorosity
(
  double* matrixPorosityNP1,
  const double* matrixPorosityN,
  const double* porePressureYOverlapNP1,
  const double* porePressureYOverlapN,
  const double* dilatationNP1,
  const double* dilatationN,
  const double compressibilityRock,
  const double alpha,
  const int* localNeighborList,
  int numOwnedPoints
);

//! Compute the new fracture Porosity
void computeFracturePorosityComplex
(
  std::complex<double>* fracturePorosityNP1,
  const std::complex<double>* breaklessDilatationOwnedNP1,
  const double* criticalDilatationOwned,
  const int* localNeighborList,
  int numOwnedPoints
);

void computeFracturePorosity
(
  double* fracturePorosityNP1,
  const double* breaklessDilatationOwnedNP1,
  const double* criticalDilatationOwned,
  const int* localNeighborList,
  int numOwnedPoints
);


}

#endif // ONE_PHASE_DIFFUSION_H
