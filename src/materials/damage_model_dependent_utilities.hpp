//! \file damage_model_dependent_utilities.hpp

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

#ifndef DMODL_DEP_UTILS
#define DMODL_DEP_UTILS

#include "Peridigm_InfluenceFunction.hpp"

namespace MATERIAL_EVALUATION {

  typedef PeridigmNS::InfluenceFunction::functionPointer FunctionPointer;

  //! Computes the approximate dilatation for the most extreme isotropic deformation allowed before bonds must fail
  // -- used in part to help compute fracture volume
  void computeCriticalDilatation
  (
    const double* xOverlap,
    const double *mOwned,
    const double* volumeOverlap,
    double* criticalDilatationOwned,
    const int* localNeighborList,
    int numOwnedPoints,
    double horizon,
    const double m_criticalStretch,
    const FunctionPointer OMEGA=PeridigmNS::InfluenceFunction::self().getInfluenceFunction()
  );

  //! Computes dilatation as if no bonds were broken in part to help estimate fracture volume
  template<typename ScalarT>
  void computeBreaklessDilatation
  (
    const double* xOverlap,
    const ScalarT* yOverlap,
    const double *mOwned,
    const double* volumeOverlap,
    ScalarT* breaklessDilatationOwned,
    const int* localNeighborList,
    int numOwnedPoints,
    double horizon,
    const FunctionPointer OMEGA=PeridigmNS::InfluenceFunction::self().getInfluenceFunction(),
    double thermalExpansionCoefficient = 0,
    const double* deltaTemperature = 0
  );

  //! Compute the new matrix Porosity
  template<typename ScalarT>
  void computeMatrixPorosity
  (
    ScalarT* matrixPorosityNP1,
    const double* matrixPorosityN,
    const ScalarT* porePressureYOverlapNP1,
    const double* porePressureYOverlapN,
    const ScalarT* dilatationNP1,
    const double* dilatationN,
    const double compressibilityRock,
    const double alpha,
    int numOwnedPoints
  );

  //! Compute the new fracture porosity
  template<typename ScalarT>
  void computeFracturePorosity
  (
    ScalarT* fracturePorosityNP1,
    const ScalarT* breaklessDilatationOwnedNP1,
    const double* criticalDilatationOwned,
    int numOwnedPoints
  );

}

#endif // DMODL_DEP_UTILS
