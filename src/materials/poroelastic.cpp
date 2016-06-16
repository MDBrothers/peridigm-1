//! \file poroelastic.cpp

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

#include <cmath>
#include <Sacado.hpp>
#include "poroelastic.hpp"
#include "material_utilities.h"
#include <boost/math/special_functions/fpclassify.hpp>

namespace MATERIAL_EVALUATION{

  //! Computes contributions to the internal force resulting from owned points.
  template<typename ScalarT>
  void computeInternalForceLinearElasticCoupled
  (
    const double* xOverlap,
    const ScalarT* yOverlap,
    const ScalarT* porePressureYOverlap,
  	const ScalarT* fracturePressureYOverlap,
    const double* mOwned,
    const double* volumeOverlap,
    const ScalarT* dilatationOwned,
    const double* damage,
    const double* bondDamage,
    const double* dsfOwned,
    ScalarT* fInternalOverlap,
    const int*  localNeighborList,
    const int numOwnedPoints,
    const double BULK_MODULUS,
    const double SHEAR_MODULUS,
    const double horizon,
    const double m_alpha,
    const double thermalExpansionCoefficient,
    const double* deltaTemperature
  )
  {
    /*
     * Compute processor local contribution to internal force
     */
    double K = BULK_MODULUS;
    double MU = SHEAR_MODULUS;

    const double *xOwned = xOverlap;
    const ScalarT *yOwned = yOverlap;
    const ScalarT *porePressureYOwned = porePressureYOverlap;
  	const ScalarT *fracturePressureYOwned = fracturePressureYOverlap;
    const double *deltaT = deltaTemperature;
    const double *m = mOwned;
    const double *v = volumeOverlap;
    const ScalarT *theta = dilatationOwned;
    ScalarT *fOwned = fInternalOverlap;
    const double * damageOwned = damage;

    const int *neighPtr = localNeighborList;
    double cellVolume, alpha, X_dx, X_dy, X_dz, zeta, omega, harmonicAverageDamage;
    ScalarT Y_dx, Y_dy, Y_dz, dY, t, fx, fy, fz, e, c1;
    for(int p=0;p<numOwnedPoints;p++, porePressureYOwned++, fracturePressureYOwned++, xOwned +=3, yOwned +=3, fOwned+=3, deltaT++, m++, theta++, damageOwned++){

      int numNeigh = *neighPtr; neighPtr++;
      const double *X = xOwned;
      const ScalarT *Y = yOwned;
      alpha = 15.0*MU/(*m);
      double selfCellVolume = v[p];
      for(int n=0;n<numNeigh;n++,neighPtr++,bondDamage++){
        int localId = *neighPtr;
        cellVolume = v[localId];
        const double *XP = &xOverlap[3*localId];
        const ScalarT *YP = &yOverlap[3*localId];
        const double *damageNeighbor = &damage[localId];
        X_dx = *(XP+0) - *(X+0); //Use indirect addressing, just like below for fOwned
        X_dy = *(XP+1) - *(X+1);
        X_dz = *(XP+2) - *(X+2);
        zeta = sqrt(X_dx*X_dx+X_dy*X_dy+X_dz*X_dz);
        Y_dx = *(YP+0) - *(Y+0);
        Y_dy = *(YP+1) - *(Y+1);
        Y_dz = *(YP+2) - *(Y+2);
        dY = sqrt(Y_dx*Y_dx+Y_dy*Y_dy+Y_dz*Y_dz);
        e = dY - zeta;

       if(deltaTemperature)
           e -= thermalExpansionCoefficient*(*deltaT)*zeta;

        omega = scalarInfluenceFunction(zeta,horizon);
  			//m_alpha is biot coefficient
  			c1 = omega*(*theta)*(3.0*K/(*m)-alpha/3.0) -3.0*omega/(*m)*m_alpha*(*porePressureYOwned);
  			t = -3.0*(*bondDamage)*(*fracturePressureYOwned)*omega/(*m)*zeta + (1.0-*bondDamage)*(c1 * zeta + (1.0-*bondDamage) * omega * alpha * e);

        fx = t * Y_dx / dY;
        fy = t * Y_dy / dY;
        fz = t * Y_dz / dY;

        *(fOwned+0) += fx*cellVolume;
        *(fOwned+1) += fy*cellVolume;
        *(fOwned+2) += fz*cellVolume;
        fInternalOverlap[3*localId+0] -= fx*selfCellVolume;
        fInternalOverlap[3*localId+1] -= fy*selfCellVolume;
        fInternalOverlap[3*localId+2] -= fz*selfCellVolume;
      }
    }
  }

  /** Explicit template instantiation for double. */
  template void computeInternalForceLinearElasticCoupled<double>
  (
    const double* xOverlap,
    const double* yOverlap,
    const double* porePressureYOverlap,
  	const double* fracturePressureYOverlap,
    const double* mOwned,
    const double* volumeOverlap,
    const double* dilatationOwned,
    const double* damage,
    const double* bondDamage,
    const double* dsfOwned,
    double* fInternalOverlap,
    const int*  localNeighborList,
    const int numOwnedPoints,
    const double BULK_MODULUS,
    const double SHEAR_MODULUS,
    const double horizon,
    const double m_alpha,
    const double thermalExpansionCoefficient,
    const double* deltaTemperature
  );

  /** Explicit template instantiation for std::complex<double>. */
  template void computeInternalForceLinearElasticCoupled<std::complex<double> >
  (
    const double* xOverlap,
    const std::complex<double>* yOverlap,
    const std::complex<double>* porePressureYOverlap,
  	const std::complex<double>* fracturePressureYOverlap,
    const double* mOwned,
    const double* volumeOverlap,
    const std::complex<double>* dilatationOwned,
    const double* damage,
    const double* bondDamage,
    const double* dsfOwned,
    std::complex<double>* fInternalOverlap,
    const int*  localNeighborList,
    int numOwnedPoints,
    const double BULK_MODULUS,
    const double SHEAR_MODULUS,
    const double horizon,
    const double thermalExpansionCoefficient,
  	const double m_alpha,
    const double* deltaTemperature
  );

}
