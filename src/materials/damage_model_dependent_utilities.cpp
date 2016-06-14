//! \file damage_model_dependent_utilities.cpp

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
#include <complex>
#include <cmath>
#include <Sacado.hpp>
#include "damage_model_dependent_utilities.hpp"
#include "material_utilities.h"
#include <boost/math/special_functions/fpclassify.hpp>

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
    const double m_criticalStretch, //NOTE MOVED m-critical stretch
    const FunctionPointer OMEGA
  )
  {
    const double *xOwned = xOverlap;
    const double *m = mOwned;
    const double *v = volumeOverlap;
    double *theta = criticalDilatationOwned;
    double cellVolume;
    const int *neighPtr = localNeighborList;
    for(int p=0; p<numOwnedPoints;p++, xOwned+=3, m++, theta++){
      int numNeigh = *neighPtr; neighPtr++;
      const double *X = xOwned;
      *theta = double(0.0);
      for(int n=0;n<numNeigh;n++,neighPtr++){
        int localId = *neighPtr;
        cellVolume = v[localId];
        const double *XP = &xOverlap[3*localId];
        double X_dx = XP[0]-X[0]; //TODO use indirect addressing
        double X_dy = XP[1]-X[1];
        double X_dz = XP[2]-X[2];
        double zetaSquared = X_dx*X_dx+X_dy*X_dy+X_dz*X_dz;
        double d = sqrt(zetaSquared);
        double e = sqrt(zetaSquared*(1.0+m_criticalStretch));
        e -= d;
        double omega = OMEGA(d,horizon);
        *theta += 3.0*omega*d*e*cellVolume/(*m);
      }
    }
  }

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
    const FunctionPointer OMEGA,
    double thermalExpansionCoefficient,
    const double* deltaTemperature
  )
  {
    const double *xOwned = xOverlap;
    const ScalarT *yOwned = yOverlap;
    const double *deltaT = deltaTemperature;
    const double *m = mOwned;
    const double *v = volumeOverlap;
    ScalarT *theta = breaklessDilatationOwned;
    double cellVolume;
    const int *neighPtr = localNeighborList;

    for(int p=0; p<numOwnedPoints;p++, xOwned+=3, yOwned+=3, deltaT++, m++, theta++){
      int numNeigh = *neighPtr; neighPtr++;
      const double *X = xOwned;
      const ScalarT *Y = yOwned;
      *theta = ScalarT(0.0);

      for(int n=0;n<numNeigh;n++,neighPtr++){
        int localId = *neighPtr;
        cellVolume = v[localId];
        const double *XP = &xOverlap[3*localId];
        const ScalarT *YP = &yOverlap[3*localId];

        double X_dx = XP[0]-X[0];
        double X_dy = XP[1]-X[1];
        double X_dz = XP[2]-X[2];
        double zetaSquared = X_dx*X_dx+X_dy*X_dy+X_dz*X_dz;
        ScalarT Y_dx = YP[0]-Y[0];
        ScalarT Y_dy = YP[1]-Y[1];
        ScalarT Y_dz = YP[2]-Y[2];
        ScalarT dY = Y_dx*Y_dx+Y_dy*Y_dy+Y_dz*Y_dz;
        double d = sqrt(zetaSquared);
        ScalarT e = sqrt(dY);
        e -= d;

        if(deltaTemperature)
          e -= thermalExpansionCoefficient*(*deltaT)*d;
        double omega = OMEGA(d,horizon);
        *theta += 3.0*omega*d*e*cellVolume/(*m);
      }
    }
  }

  template void computeBreaklessDilatation<double>
  (
    const double* xOverlap,
    const double* yOverlap,
    const double *mOwned,
    const double* volumeOverlap,
    double* breaklessDilatationOwned,
    const int* localNeighborList,
    int numOwnedPoints,
    double horizon,
    const FunctionPointer OMEGA,
    double thermalExpansionCoefficient,
    const double* deltaTemperature
  );

  template void computeBreaklessDilatation<std::complex<double> >
  (
    const double* xOverlap,
    const std::complex<double>* yOverlap,
    const double *mOwned,
    const double* volumeOverlap,
    std::complex<double>* breaklessDilatationOwned,
    const int* localNeighborList,
    int numOwnedPoints,
    double horizon,
    const FunctionPointer OMEGA,
    double thermalExpansionCoefficient,
    const double* deltaTemperature
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
    const double m_compressibilityRock,
    const double m_alpha,
    int numOwnedPoints
  ){
    ScalarT* phiMatrixNP1 = matrixPorosityNP1;
    const double* phiMatrixN = matrixPorosityN;
    const ScalarT* porePressureYOwnedNP1 = porePressureYOverlapNP1;
    const double* porePressureYOwnedN = porePressureYOverlapN;
    const ScalarT* thetaMatrixLocalNP1 = dilatationNP1; //The definition of matrix dilatation from Hisanao's formulation matches the standard definition of dilatation with a critical stretch damage model.
    const double* thetaMatrixLocalN = dilatationN;

    for(int p=0; p<numOwnedPoints; p++,  phiMatrixNP1++, phiMatrixN++, porePressureYOwnedNP1++, porePressureYOwnedN++, thetaMatrixLocalNP1++, thetaMatrixLocalN++){
      *phiMatrixNP1 = (*phiMatrixN)*(1.0 - m_compressibilityRock*(*porePressureYOwnedNP1 - *porePressureYOwnedN)) +
                      m_alpha*(1.0 + *thetaMatrixLocalN)*(m_compressibilityRock*(*porePressureYOwnedNP1 - *porePressureYOwnedN) + (*thetaMatrixLocalNP1 - *thetaMatrixLocalN));
    }
  }

  //! Explicit template instantiation for method to compute the new matrix Porosity
  template void computeMatrixPorosity<std::complex<double> >
  (
    std::complex<double>* matrixPorosityNP1,
    const double* matrixPorosityN,
    const std::complex<double>* porePressureYOverlapNP1,
    const double porePressureYOverlapN,
    const std::complex<double>* dilatationNP1,
    const double* dilatationN,
    const double m_compressibilityRock,
    const double m_alpha,
    int numOwnedPoints
  );

  //! Explicit template intstantiation for method to compute the new matrix Porosity
  template void computeMatrixPorosity<double>
  (
    double* matrixPorosityNP1,
    const double* matrixPorosityN,
    const double* porePressureYOverlapNP1,
    const double* porePressureYOverlapN,
    const double* dilatationNP1,
    const double* dilatationN,
    const double m_compressibilityRock,
    const double m_alpha,
    const int* localNeighborList,
    int numOwnedPoints
  );

  //! Explicit template specialization for to compute the new fracture Porosity
  template void computeFracturePorosity<std::complex<double> >
  (
    std::complex<double>* fracturePorosityNP1,
    const std::complex<double>* breaklessDilatationOwnedNP1,
    const double* criticalDilatationOwned,
    int numOwnedPoints
  ){
    const std::complex<double>* thetaLocal = breaklessDilatationOwnedNP1; //The definition of local dilatation from Hisanao's formulation matches the standard definition of dilatation without a damage model.
    const double* thetaCritical = criticalDilatationOwned;
    std::complex<double>* fracturePorosity = fracturePorosityNP1;

    for(int p=0; p<numOwnedPoints;p++, thetaLocal++, thetaCritical++, fracturePorosity++){
      *fracturePorosity = *thetaLocal - *thetaCritical;
      if(std::real(*fracturePorosity) < 0.0) *fracturePorosity = std::complex(0.0, std::imag(*fracturePorosity)); //No negative porosities, but preserve imaginary component.
    }
  }

  template<typename ScalarT>
  void computeFracturePorosity
  (
    ScalarT* fracturePorosityNP1,
    const ScalarT* breaklessDilatationOwnedNP1,
    const double* criticalDilatationOwned,
    int numOwnedPoints
  ){
    const ScalarT* thetaLocal = breaklessDilatationOwnedNP1; //The definition of local dilatation from Hisanao's formulation matches the standard definition of dilatation without a damage model.
    const double* thetaCritical = criticalDilatationOwned;
    ScalarT* fracturePorosity = fracturePorosityNP1;

    for(int p=0; p<numOwnedPoints;p++, thetaLocal++, thetaCritical++, fracturePorosity++){
      *fracturePorosity = *thetaLocal - *thetaCritical;
      if(*fracturePorosity < 0.0) *fracturePorosity = 0.0; //No negative porosities.
    }
  }

  //! Explcit template instantiation for to compute the fracture Porosity
  template void computeFracturePorosity<double>
  (
    double* fracturePorosityNP1,
    const double* breaklessDilatationOwnedNP1,
    const double* criticalDilatationOwned,
    int numOwnedPoints
  );


}}
