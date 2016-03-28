//! \file two_phase_nonlocal_diffusion.cxx

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
#include "two_phase_nonlocal_diffusion.h"
#include "material_utilities.h"
#include <boost/math/special_functions/fpclassify.hpp>


namespace MATERIAL_EVALUATION {

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

template void computeBreaklessDilatation<Sacado::Fad::DFad<double> >
(
  const double* xOverlap,
  const Sacado::Fad::DFad<double>* yOverlap,
  const double *mOwned,
  const double* volumeOverlap,
  Sacado::Fad::DFad<double>* breaklessDilatationOwned,
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
  const double* principleDamageDirection,
  const double* criticalDilatationOwned,
  const ScalarT* breaklessDilatationOwned,
  ScalarT* phaseOnePoreFlowOverlap,
  ScalarT* phaseOneFracFlowOverlap,
  ScalarT* phaseTwoPoreFlowOverlap,
  ScalarT* phaseTwoFracFlowOverlap,
  const int*  localNeighborList,
  const int numOwnedPoints,
  const double m_permeabilityScalar,
  const double m_permeabilityCurveInflectionDamage,
  const double m_permeabilityAlpha,
  const double m_maxPermeability,
  const double m_phaseOneBasePerm,
  const double m_phaseTwoBasePerm,
  const double m_phaseOneDensity,
  const double m_phaseTwoDensity,
  const double m_phaseOneCompressibility,
  const double m_phaseTwoCompressibility,
  const double m_phaseOneViscosity,
  const double m_phaseTwoViscosity,
  const double m_horizon,
  const double m_horizon_fracture,
  const double* deltaTemperature
)
{
  /*
   * Compute processor local contribution to internal fluid flow
   */

  const ScalarT *yOwned = yOverlap;
  const double *porePressureVOwned = porePressureVOverlap;
  const ScalarT *porePressureYOwned = porePressureYOverlap;
  const double *fracturePressureVOwned = fracturePressureVOverlap;
  const ScalarT *fracturePressureYOwned = fracturePressureYOverlap;
  const ScalarT *phaseOneSaturationPoresYOwned = phaseOneSaturationPoresYOverlap;
  const double *phaseOneSaturationPoresVOwned = phaseOneSaturationPoresVOverlap;
  const ScalarT *phaseOneSaturationFracYOwned = phaseOneSaturationFracYOverlap;
  const double *phaseOneSaturationFracVOwned = phaseOneSaturationFracVOverlap;

  const double *v = volumeOverlap;
  const double *deltaT = deltaTemperature;
  const double *damageOwned = damage;
  const double *principleDamageDirectionOwned = principleDamageDirection;
  const double *thetaCritical = criticalDilatationOwned;
  const ScalarT *thetaBreakless = breaklessDilatationOwned;

  ScalarT *phaseOnePoreFlowOwned = phaseOnePoreFlowOverlap;
  ScalarT *phaseOneFracFlowOwned = phaseOneFracFlowOverlap;
  ScalarT *phaseTwoPoreFlowOwned = phaseTwoPoreFlowOverlap;
  ScalarT *phaseTwoFracFlowOwned = phaseTwoFracFlowOverlap;

  const int *neighPtr = localNeighborList;
  double cellVolume, harmonicAverageDamage;
  ScalarT phaseOnePorePerm, phaseTwoPorePerm;
  ScalarT dPorePressure, dFracPressure, dLocalPoreFracPressure, Y_dx, Y_dy, Y_dz, dY, fractureWidthFactor;
  ScalarT fractureDirectionFactor, phaseOneFracPerm, phaseTwoFracPerm, phaseOneRelPermPores;
  ScalarT phaseOneRelPermFrac, phaseTwoRelPermPores, phaseTwoRelPermFrac, satStarPores, satStarFrac;
  ScalarT scalarPhaseOnePoreFlow, scalarPhaseOneFracFlow, scalarPhaseTwoPoreFlow, scalarPhaseTwoFracFlow;
  ScalarT scalarPhaseOneFracToPoreFlow, scalarPhaseTwoFracToPoreFlow, omegaPores, omegaFrac;

  for(int p=0;p<numOwnedPoints;p++, porePressureYOwned++, fracturePressureYOwned++,
                                    phaseOneSaturationPoresYOwned++, phaseOneSaturationPoresVOwned++,
                                    phaseOneSaturationFracYOwned++, phaseOneSaturationFracVOwned++,
                                    yOwned +=3, phaseOnePoreFlowOwned++, phaseOneFracFlowOwned++,
                                    phaseTwoPoreFlowOwned++, phaseTwoFracFlowOwned++, deltaT++, damageOwned++,
                                    principleDamageDirectionOwned +=3, thetaCritical++, thetaBreakless++,
                                    porePressureVOwned++, fracturePressureVOwned++){
    int numNeigh = *neighPtr; neighPtr++;
    double selfCellVolume = v[p];
    const ScalarT *Y = yOwned;
    const ScalarT *porePressureY = porePressureYOwned;
    const double *porePressureV = porePressureVOwned;
    const ScalarT *fracturePressureY = fracturePressureYOwned;
    const double *fracturePressureV = fracturePressureVOwned;
    const ScalarT *phaseOneSaturationPoresY = phaseOneSaturationPoresYOwned;
    const ScalarT *phaseOneSaturationFracY = phaseOneSaturationFracYOwned;
    const double *principleDamageDirection = principleDamageDirectionOwned;

    // compute relative permeabilities assuming no damage effect
    satStarPores = (*phaseOneSaturationPoresY - 0.2)/0.6; // means spec one is water
    satStarFrac = (*phaseOneSaturationFracY - 0.2)/0.6;
    phaseOneRelPermPores = m_phaseOneBasePerm*pow(satStarPores, 2.0); //Empirical model, exponent is related to the material
    phaseOneRelPermFrac = m_phaseOneBasePerm*pow(satStarFrac, 2.0);
    phaseTwoRelPermPores = m_phaseTwoBasePerm*pow((-satStarPores+1.0),2.0);
    phaseTwoRelPermFrac = m_phaseTwoBasePerm*pow((-satStarFrac+1.0),2.0);

    // for to calculate Leakoff
    dLocalPoreFracPressure = *fracturePressureY - *porePressureY;

    //compute fracture width based on a two sphere diameter difference. An ad-hoc relation.
    if(*thetaBreakless > 0.0){
      fractureWidthFactor = pow(6.0/M_PI*selfCellVolume*(*thetaBreakless),1.0/3.0) - pow(6.0/M_PI*selfCellVolume*(*thetaCritical),1.0/3.0);
      if(fractureWidthFactor < 0.0)
        fractureWidthFactor = 0.0; //Closed fractures have no flow
    }
    else
      fractureWidthFactor = 0.0;

    fractureWidthFactor *= fractureWidthFactor/12.0; //Empirical relation for permeability

    for(int n=0;n<numNeigh;n++,neighPtr++){
      int localId = *neighPtr;
      cellVolume = v[localId];
      const ScalarT *porePressureYP = &porePressureYOverlap[localId];
      const ScalarT *fracturePressureYP = &fracturePressureYOverlap[localId];
      const ScalarT *YP = &yOverlap[3*localId];
      const double *damageNeighbor = &damage[localId]; //TODO synchronize neighbor damage before force evaluation (this is aparently expensive though)

      Y_dx = *(YP+0) - *(Y+0);
      Y_dy = *(YP+1) - *(Y+1);
      Y_dz = *(YP+2) - *(Y+2);
      dY = sqrt(Y_dx*Y_dx+Y_dy*Y_dy+Y_dz*Y_dz);
      //NOTE I want to use DFad<double>, which is why I circumvent the standard influence function code.
      omegaPores = exp(-dY*dY/(m_horizon*m_horizon));// scalarInfluenceFunction(dY,m_horizon);
      //Frac diffusion is a more local process than pore diffusion.
      omegaFrac = exp(-dY*dY/(m_horizon_fracture*m_horizon_fracture));// scalarInfluenceFunction(dY,m_horizon_fracture);

      // Pressure potential
      dPorePressure = *porePressureYP - *porePressureY;
      dFracPressure = *fracturePressureYP - *fracturePressureY;

      // compute permeabilities
      // Frac permeability in directions other than orthogonal to the principle damage direction is strongly attenuated.
      fractureDirectionFactor = pow(cos(Y_dx*(*(principleDamageDirection+0)) + Y_dy*(*(principleDamageDirection+1)) + Y_dz*(*(principleDamageDirection+2))),2.0); //Frac flow allowed in direction perpendicular to damage
      harmonicAverageDamage = 1.0 / (1.0 / *damageOwned + 1.0 / *damageNeighbor);
      if(harmonicAverageDamage != harmonicAverageDamage) harmonicAverageDamage=0.0; //test for nan which occurs when a damage is zero.
      // Pore permeability is affected by an ad-hoc S-curve relation to damage.
      phaseOnePorePerm = m_permeabilityScalar*phaseOneRelPermPores + m_maxPermeability/(exp(-m_permeabilityAlpha*(harmonicAverageDamage - m_permeabilityCurveInflectionDamage))+1.0);
      phaseTwoPorePerm = m_permeabilityScalar*phaseTwoRelPermPores + m_maxPermeability/(exp(-m_permeabilityAlpha*(harmonicAverageDamage - m_permeabilityCurveInflectionDamage))+1.0);
      // Frac permeability is affected by bond allignment with fracture plane, width, and saturation
      phaseOneFracPerm = fractureWidthFactor*fractureDirectionFactor*phaseOneRelPermFrac;
      phaseTwoFracPerm = fractureWidthFactor*fractureDirectionFactor*phaseTwoRelPermFrac;

      // compute flow density
      // flow entering cell is positive
      scalarPhaseOnePoreFlow = omegaPores * m_phaseOneDensity / m_phaseOneViscosity * (4.0 / (M_PI*m_horizon*m_horizon)) * (phaseOnePorePerm / pow(dY, 4.0)) * dPorePressure;
      scalarPhaseOneFracFlow = omegaFrac * m_phaseOneDensity / m_phaseOneViscosity * (4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture)) * (phaseOneFracPerm / pow(dY, 4.0)) * dFracPressure;
      scalarPhaseTwoPoreFlow = omegaPores * m_phaseTwoDensity / m_phaseTwoViscosity * (4.0 / (M_PI*m_horizon*m_horizon)) * (phaseTwoPorePerm / pow(dY, 4.0)) * dPorePressure;
      scalarPhaseTwoFracFlow = omegaFrac * m_phaseTwoDensity / m_phaseTwoViscosity * (4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture)) * (phaseTwoFracPerm / pow(dY, 4.0)) * dFracPressure;

      // convert flow density to flow and account for reactions
      *phaseOnePoreFlowOwned += scalarPhaseOnePoreFlow*cellVolume;
      *phaseOneFracFlowOwned += scalarPhaseOneFracFlow*cellVolume;
      *phaseTwoPoreFlowOwned += scalarPhaseTwoPoreFlow*cellVolume;
      *phaseTwoFracFlowOwned += scalarPhaseTwoFracFlow*cellVolume;
      phaseOnePoreFlowOverlap[localId] -= scalarPhaseOnePoreFlow*selfCellVolume;
      phaseOneFracFlowOverlap[localId] -= scalarPhaseOneFracFlow*selfCellVolume;
      phaseTwoPoreFlowOverlap[localId] -= scalarPhaseTwoPoreFlow*selfCellVolume;
      phaseTwoFracFlowOverlap[localId] -= scalarPhaseTwoFracFlow*selfCellVolume;
    }

    //Leakoff calculation, self to self
    *phaseOnePoreFlowOwned += m_phaseOneDensity / m_phaseOneViscosity * 4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture) * (m_permeabilityScalar*phaseOneRelPermPores + m_maxPermeability/(1.0 + exp(-m_permeabilityAlpha*(*damageOwned - m_permeabilityCurveInflectionDamage))) / pow(m_horizon_fracture, 4.0)) * dLocalPoreFracPressure;
    *phaseOneFracFlowOwned -= m_phaseOneDensity / m_phaseOneViscosity * 4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture) * (m_permeabilityScalar*phaseOneRelPermPores + m_maxPermeability/(1.0 + exp(-m_permeabilityAlpha*(*damageOwned - m_permeabilityCurveInflectionDamage))) / pow(m_horizon_fracture, 4.0)) * dLocalPoreFracPressure;
    *phaseTwoPoreFlowOwned += m_phaseTwoDensity / m_phaseTwoViscosity * 4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture) * (m_permeabilityScalar*phaseTwoRelPermPores + m_maxPermeability/(1.0 + exp(-m_permeabilityAlpha*(*damageOwned - m_permeabilityCurveInflectionDamage))) / pow(m_horizon_fracture, 4.0)) * dLocalPoreFracPressure;
    *phaseTwoFracFlowOwned -= m_phaseTwoDensity / m_phaseTwoViscosity * 4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture) * (m_permeabilityScalar*phaseTwoRelPermPores + m_maxPermeability/(1.0 + exp(-m_permeabilityAlpha*(*damageOwned - m_permeabilityCurveInflectionDamage))) / pow(m_horizon_fracture, 4.0)) * dLocalPoreFracPressure;

    //Viscous terms
    double Porosity= 0.2;
    double B_formation_vol_factor_water = 1.0;
    double B_formation_vol_factor_oil = 1.0;
    double Compressibility_rock = 1.0;
    *phaseOnePoreFlowOwned = *phaseOnePoreFlowOwned - *porePressureV*m_phaseOneDensity*(Compressibility_rock + m_phaseOneCompressibility)*Porosity*(*phaseOneSaturationPoresYOwned)/B_formation_vol_factor_water + Porosity/B_formation_vol_factor_water*(*phaseOneSaturationPoresVOwned);
    *phaseOneFracFlowOwned = *phaseOneFracFlowOwned - *fracturePressureV*m_phaseOneDensity*(Compressibility_rock + m_phaseOneCompressibility)*Porosity*(*phaseOneSaturationFracYOwned)/B_formation_vol_factor_water + Porosity/B_formation_vol_factor_water*(*phaseOneSaturationFracVOwned);
    *phaseTwoPoreFlowOwned = *phaseTwoPoreFlowOwned - *porePressureV*m_phaseTwoDensity*(Compressibility_rock + m_phaseTwoCompressibility)*Porosity*(1.0 - *phaseOneSaturationPoresYOwned)/B_formation_vol_factor_oil + Porosity/B_formation_vol_factor_oil*(*phaseOneSaturationPoresVOwned);
    *phaseTwoFracFlowOwned = *phaseTwoFracFlowOwned - *fracturePressureV*m_phaseTwoDensity*(Compressibility_rock + m_phaseTwoCompressibility)*Porosity*(1.0 - *phaseOneSaturationFracYOwned)/B_formation_vol_factor_oil + Porosity/B_formation_vol_factor_oil*(*phaseOneSaturationFracVOwned);
  }
}

/** Explicit template instantiation for double. */
template void computeInternalFlow<double>
(
	const double* yOverlap,
  const double* porePressureYOverlap,
  const double* porePressureVOverlap,
  const double* fracturePressureYOverlap,
  const double* fracturePressureVOverlap,
  const double* phaseOneSaturationPoresYOverlap,
  const double* phaseOneSaturationPoresVOverlap,
  const double* phaseOneSaturationFracYOverlap,
  const double* phaseOneSaturationFracVOverlap,
  const double* volumeOverlap,
  const double* damage,
  const double* principleDamageDirection,
  const double* criticalDilatationOwned,
  const double* breaklessDilatationOwned,
  double* phaseOnePoreFlowOverlap,
  double* phaseOneFracFlowOverlap,
  double* phaseTwoPoreFlowOverlap,
  double* phaseTwoFracFlowOverlap,
  const int*  localNeighborList,
  const int numOwnedPoints,
  const double m_permeabilityScalar,
  const double m_permeabilityCurveInflectionDamage,
  const double m_permeabilityAlpha,
  const double m_maxPermeability,
  const double m_phaseOneBasePerm,
  const double m_phaseTwoBasePerm,
  const double m_phaseOneDensity,
  const double m_phaseTwoDensity,
  const double m_phaseOneCompressibility,
  const double m_phaseTwoCompressibility,
  const double m_phaseOneViscosity,
  const double m_phaseTwoViscosity,
  const double m_horizon,
  const double m_horizon_fracture,
  const double* deltaTemperature
);

/** Explicit template instantiation for Sacado::Fad::DFad<double>. */
template void computeInternalFlow<Sacado::Fad::DFad<double> >
(
	const Sacado::Fad::DFad<double>* yOverlap,
  const Sacado::Fad::DFad<double>* porePressureYOverlap,
  const double* porePressureVOverlap,
  const Sacado::Fad::DFad<double>* fracturePressureYOverlap,
  const double* fracturePressureVOverlap,
  const Sacado::Fad::DFad<double>* phaseOneSaturationPoresYOverlap,
  const double* phaseOneSaturationPoresVOverlap,
  const Sacado::Fad::DFad<double>* phaseOneSaturationFracYOverlap,
  const double* phaseOneSaturationFracVOverlap,
  const double* volumeOverlap,
  const double* damage,
  const double* principleDamageDirection,
  const double* criticalDilatationOwned,
  const Sacado::Fad::DFad<double>* breaklessDilatationOwned,
  Sacado::Fad::DFad<double>* phaseOnePoreFlowOverlap,
  Sacado::Fad::DFad<double>* phaseOneFracFlowOverlap,
  Sacado::Fad::DFad<double>* phaseTwoPoreFlowOverlap,
  Sacado::Fad::DFad<double>* phaseTwoFracFlowOverlap,
  const int*  localNeighborList,
  const int numOwnedPoints,
  const double m_permeabilityScalar,
  const double m_permeabilityCurveInflectionDamage,
  const double m_permeabilityAlpha,
  const double m_maxPermeability,
  const double m_phaseOneBasePerm,
  const double m_phaseTwoBasePerm,
  const double m_phaseOneDensity,
  const double m_phaseTwoDensity,
  const double m_phaseOneCompressibility,
  const double m_phaseTwoCompressibility,
  const double m_phaseOneViscosity,
  const double m_phaseTwoViscosity,
  const double m_horizon,
  const double m_horizon_fracture,
  const double* deltaTemperature
);

/*Because the complex case needs a difference argument pattern we need something
different than templates. */
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
  const double* principleDamageDirection,
  const double* criticalDilatationOwned,
  const std::complex<double>* breaklessDilatationOwned,
  std::complex<double>* phaseOnePoreFlowOverlap,
  std::complex<double>* phaseOneFracFlowOverlap,
  std::complex<double>* phaseTwoPoreFlowOverlap,
  std::complex<double>* phaseTwoFracFlowOverlap,
  const int*  localNeighborList,
  const int numOwnedPoints,
  const double m_permeabilityScalar,
  const double m_permeabilityCurveInflectionDamage,
  const double m_permeabilityAlpha,
  const double m_maxPermeability,
  const double m_phaseOneBasePerm,
  const double m_phaseTwoBasePerm,
  const double m_phaseOneDensity,
  const double m_phaseTwoDensity,
  const double m_phaseOneCompressibility,
  const double m_phaseTwoCompressibility,
  const double m_phaseOneViscosity,
  const double m_phaseTwoViscosity,
  const double m_horizon,
  const double m_horizon_fracture,
  const double* deltaTemperature
)
{
  /*
   * Compute processor local contribution to internal fluid flow
   */

  const std::complex<double> *yOwned = yOverlap;
  const std::complex<double> *porePressureVOwned = porePressureVOverlap;
  const std::complex<double> *porePressureYOwned = porePressureYOverlap;
  const std::complex<double> *fracturePressureVOwned = fracturePressureVOverlap;
  const std::complex<double> *fracturePressureYOwned = fracturePressureYOverlap;
  const std::complex<double> *phaseOneSaturationPoresYOwned = phaseOneSaturationPoresYOverlap;
  const std::complex<double> *phaseOneSaturationPoresVOwned = phaseOneSaturationPoresVOverlap;
  const std::complex<double> *phaseOneSaturationFracYOwned = phaseOneSaturationFracYOverlap;
  const std::complex<double> *phaseOneSaturationFracVOwned = phaseOneSaturationFracVOverlap;

  const double *v = volumeOverlap;
  const double *deltaT = deltaTemperature;
  const double *damageOwned = damage;
  const double *principleDamageDirectionOwned = principleDamageDirection;
  const double *thetaCritical = criticalDilatationOwned;
  const std::complex<double> *thetaBreakless = breaklessDilatationOwned;

  std::complex<double> *phaseOnePoreFlowOwned = phaseOnePoreFlowOverlap;
  std::complex<double> *phaseOneFracFlowOwned = phaseOneFracFlowOverlap;
  std::complex<double> *phaseTwoPoreFlowOwned = phaseTwoPoreFlowOverlap;
  std::complex<double> *phaseTwoFracFlowOwned = phaseTwoFracFlowOverlap;

  const int *neighPtr = localNeighborList;
  double cellVolume, harmonicAverageDamage;
  std::complex<double> phaseOnePorePerm, phaseTwoPorePerm;
  std::complex<double> dPorePressure, dFracPressure, dLocalPoreFracPressure, Y_dx, Y_dy, Y_dz, dY, fractureWidthFactor;
  std::complex<double> fractureDirectionFactor, phaseOneFracPerm, phaseTwoFracPerm, phaseOneRelPermPores;
  std::complex<double> phaseOneRelPermFrac, phaseTwoRelPermPores, phaseTwoRelPermFrac, satStarPores, satStarFrac;
  std::complex<double> scalarPhaseOnePoreFlow, scalarPhaseOneFracFlow, scalarPhaseTwoPoreFlow, scalarPhaseTwoFracFlow;
  std::complex<double> scalarPhaseOneFracToPoreFlow, scalarPhaseTwoFracToPoreFlow, omegaPores, omegaFrac;

  for(int p=0;p<numOwnedPoints;p++, porePressureYOwned++, fracturePressureYOwned++,
                                    phaseOneSaturationPoresYOwned++, phaseOneSaturationPoresVOwned++,
                                    phaseOneSaturationFracYOwned++, phaseOneSaturationFracVOwned++,
                                    yOwned +=3, phaseOnePoreFlowOwned++, phaseOneFracFlowOwned++,
                                    phaseTwoPoreFlowOwned++, phaseTwoFracFlowOwned++, deltaT++, damageOwned++,
                                    principleDamageDirectionOwned +=3, thetaCritical++, thetaBreakless++,
                                    porePressureVOwned++, fracturePressureVOwned++){
    int numNeigh = *neighPtr; neighPtr++;
    double selfCellVolume = v[p];
    const std::complex<double> *Y = yOwned;
    const std::complex<double> *porePressureY = porePressureYOwned;
    const std::complex<double> *porePressureV = porePressureVOwned;
    const std::complex<double> *fracturePressureY = fracturePressureYOwned;
    const std::complex<double> *fracturePressureV = fracturePressureVOwned;
    const std::complex<double> *phaseOneSaturationPoresY = phaseOneSaturationPoresYOwned;
    const std::complex<double> *phaseOneSaturationFracY = phaseOneSaturationFracYOwned;
    const double *principleDamageDirection = principleDamageDirectionOwned;

    // compute relative permeabilities assuming no damage effect
    satStarPores = (*phaseOneSaturationPoresY - 0.2)/0.6; // means spec one is water
    satStarFrac = (*phaseOneSaturationFracY - 0.2)/0.6;
    phaseOneRelPermPores = m_phaseOneBasePerm*pow(satStarPores, 2.0); //Empirical model, exponent is related to the material
    phaseOneRelPermFrac = m_phaseOneBasePerm*pow(satStarFrac, 2.0);
    phaseTwoRelPermPores = m_phaseTwoBasePerm*pow((-satStarPores+1.0),2.0);
    phaseTwoRelPermFrac = m_phaseTwoBasePerm*pow((-satStarFrac+1.0),2.0);

    // for to calculate Leakoff
    dLocalPoreFracPressure = *fracturePressureY - *porePressureY;

    //compute fracture width based on a two sphere diameter difference. An ad-hoc relation.
    if(std::real(*thetaBreakless) > 0.0){
      fractureWidthFactor = pow(6.0/M_PI*selfCellVolume*(*thetaBreakless),1.0/3.0) - pow(6.0/M_PI*selfCellVolume*(*thetaCritical),1.0/3.0);
      if(std::real(fractureWidthFactor) < 0.0)
        fractureWidthFactor = std::complex<double>(0.0, std::imag(fractureWidthFactor)); //Closed fractures have no real flow
    }
    else
      fractureWidthFactor = std::complex<double>(0.0, std::imag(fractureWidthFactor));

    fractureWidthFactor *= fractureWidthFactor/12.0; //Empirical relation for permeability

    for(int n=0;n<numNeigh;n++,neighPtr++){
      int localId = *neighPtr;
      cellVolume = v[localId];
      const std::complex<double> *porePressureYP = &porePressureYOverlap[localId];
      const std::complex<double> *fracturePressureYP = &fracturePressureYOverlap[localId];
      const std::complex<double> *YP = &yOverlap[3*localId];
      const double *damageNeighbor = &damage[localId]; //TODO synchronize neighbor damage before force evaluation (this is aparently expensive though)

      Y_dx = *(YP+0) - *(Y+0);
      Y_dy = *(YP+1) - *(Y+1);
      Y_dz = *(YP+2) - *(Y+2);
      dY = sqrt(Y_dx*Y_dx+Y_dy*Y_dy+Y_dz*Y_dz);
      //NOTE I want to use std::complex<double>, which is why I circumvent the standard influence function code.
      omegaPores = exp(-dY*dY/(m_horizon*m_horizon));// scalarInfluenceFunction(dY,m_horizon);
      //Frac diffusion is a more local process than pore diffusion.
      omegaFrac = exp(-dY*dY/(m_horizon_fracture*m_horizon_fracture));// scalarInfluenceFunction(dY,m_horizon_fracture);

      // Pressure potential
      dPorePressure = *porePressureYP - *porePressureY;
      dFracPressure = *fracturePressureYP - *fracturePressureY;

      // compute permeabilities
      // Frac permeability in directions other than orthogonal to the principle damage direction is strongly attenuated.
      fractureDirectionFactor = pow(cos(Y_dx*(*(principleDamageDirection+0)) + Y_dy*(*(principleDamageDirection+1)) + Y_dz*(*(principleDamageDirection+2))),2.0); //Frac flow allowed in direction perpendicular to damage
      harmonicAverageDamage = 1.0 / (1.0 / *damageOwned + 1.0 / *damageNeighbor);
      if(harmonicAverageDamage != harmonicAverageDamage) harmonicAverageDamage=0.0; //test for nan which occurs when a damage is zero.
      // Pore permeability is affected by an ad-hoc S-curve relation to damage.
      phaseOnePorePerm = m_permeabilityScalar*phaseOneRelPermPores + m_maxPermeability/(exp(-m_permeabilityAlpha*(harmonicAverageDamage - m_permeabilityCurveInflectionDamage))+1.0);
      phaseTwoPorePerm = m_permeabilityScalar*phaseTwoRelPermPores + m_maxPermeability/(exp(-m_permeabilityAlpha*(harmonicAverageDamage - m_permeabilityCurveInflectionDamage))+1.0);

      // Frac permeability is affected by bond allignment with fracture plane, width, and saturation
      phaseOneFracPerm = fractureWidthFactor*fractureDirectionFactor*phaseOneRelPermFrac;
      phaseTwoFracPerm = fractureWidthFactor*fractureDirectionFactor*phaseTwoRelPermFrac;

      // compute flow density
      // flow entering cell is positive
      scalarPhaseOnePoreFlow = omegaPores * m_phaseOneDensity / m_phaseOneViscosity * (4.0 / (M_PI*m_horizon*m_horizon)) * (phaseOnePorePerm / pow(dY, 4.0)) * dPorePressure;
      scalarPhaseOneFracFlow = omegaFrac * m_phaseOneDensity / m_phaseOneViscosity * (4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture)) * (phaseOneFracPerm / pow(dY, 4.0)) * dFracPressure;
      scalarPhaseTwoPoreFlow = omegaPores * m_phaseTwoDensity / m_phaseTwoViscosity * (4.0 / (M_PI*m_horizon*m_horizon)) * (phaseTwoPorePerm / pow(dY, 4.0)) * dPorePressure;
      scalarPhaseTwoFracFlow = omegaFrac * m_phaseTwoDensity / m_phaseTwoViscosity * (4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture)) * (phaseTwoFracPerm / pow(dY, 4.0)) * dFracPressure;

      // convert flow density to flow and account for reactions
      *phaseOnePoreFlowOwned += scalarPhaseOnePoreFlow*cellVolume;
      *phaseOneFracFlowOwned += scalarPhaseOneFracFlow*cellVolume;
      *phaseTwoPoreFlowOwned += scalarPhaseTwoPoreFlow*cellVolume;
      *phaseTwoFracFlowOwned += scalarPhaseTwoFracFlow*cellVolume;
      phaseOnePoreFlowOverlap[localId] -= scalarPhaseOnePoreFlow*selfCellVolume;
      phaseOneFracFlowOverlap[localId] -= scalarPhaseOneFracFlow*selfCellVolume;
      phaseTwoPoreFlowOverlap[localId] -= scalarPhaseTwoPoreFlow*selfCellVolume;
      phaseTwoFracFlowOverlap[localId] -= scalarPhaseTwoFracFlow*selfCellVolume;
    }

    //Leakoff calculation, self to self
    *phaseOnePoreFlowOwned += m_phaseOneDensity / m_phaseOneViscosity * 4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture) * (m_permeabilityScalar*phaseOneRelPermPores + m_maxPermeability/(1.0 + exp(-m_permeabilityAlpha*(*damageOwned - m_permeabilityCurveInflectionDamage))) / pow(m_horizon_fracture, 4.0)) * dLocalPoreFracPressure;
    *phaseOneFracFlowOwned -= m_phaseOneDensity / m_phaseOneViscosity * 4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture) * (m_permeabilityScalar*phaseOneRelPermPores + m_maxPermeability/(1.0 + exp(-m_permeabilityAlpha*(*damageOwned - m_permeabilityCurveInflectionDamage))) / pow(m_horizon_fracture, 4.0)) * dLocalPoreFracPressure;
    *phaseTwoPoreFlowOwned += m_phaseTwoDensity / m_phaseTwoViscosity * 4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture) * (m_permeabilityScalar*phaseTwoRelPermPores + m_maxPermeability/(1.0 + exp(-m_permeabilityAlpha*(*damageOwned - m_permeabilityCurveInflectionDamage))) / pow(m_horizon_fracture, 4.0)) * dLocalPoreFracPressure;
    *phaseTwoFracFlowOwned -= m_phaseTwoDensity / m_phaseTwoViscosity * 4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture) * (m_permeabilityScalar*phaseTwoRelPermPores + m_maxPermeability/(1.0 + exp(-m_permeabilityAlpha*(*damageOwned - m_permeabilityCurveInflectionDamage))) / pow(m_horizon_fracture, 4.0)) * dLocalPoreFracPressure;

    //Viscous terms
    double Porosity= 0.2;
    double B_formation_vol_factor_water = 1.0;
    double B_formation_vol_factor_oil = 1.0;
    double Compressibility_rock = 1.0;
    *phaseOnePoreFlowOwned = *phaseOnePoreFlowOwned - *porePressureV*m_phaseOneDensity*(Compressibility_rock + m_phaseOneCompressibility)*Porosity*(*phaseOneSaturationPoresYOwned)/B_formation_vol_factor_water + Porosity/B_formation_vol_factor_water*(*phaseOneSaturationPoresVOwned);
    *phaseOneFracFlowOwned = *phaseOneFracFlowOwned - *fracturePressureV*m_phaseOneDensity*(Compressibility_rock + m_phaseOneCompressibility)*Porosity*(*phaseOneSaturationFracYOwned)/B_formation_vol_factor_water + Porosity/B_formation_vol_factor_water*(*phaseOneSaturationFracVOwned);
    *phaseTwoPoreFlowOwned = *phaseTwoPoreFlowOwned - *porePressureV*m_phaseTwoDensity*(Compressibility_rock + m_phaseTwoCompressibility)*Porosity*(1.0 - *phaseOneSaturationPoresYOwned)/B_formation_vol_factor_oil + Porosity/B_formation_vol_factor_oil*(*phaseOneSaturationPoresVOwned);
    *phaseTwoFracFlowOwned = *phaseTwoFracFlowOwned - *fracturePressureV*m_phaseTwoDensity*(Compressibility_rock + m_phaseTwoCompressibility)*Porosity*(1.0 - *phaseOneSaturationFracYOwned)/B_formation_vol_factor_oil + Porosity/B_formation_vol_factor_oil*(*phaseOneSaturationFracVOwned);
  }
}


//! Computes contributions to the internal force resulting from owned points.
template<typename ScalarT>
void computeInternalForceLinearElasticCoupled
(
  const double* xOverlap,
  const ScalarT* yOverlap,
  const ScalarT* porePressureYOverlap,
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
  const double *deltaT = deltaTemperature;
  const double *m = mOwned;
  const double *v = volumeOverlap;
  const ScalarT *theta = dilatationOwned;
  ScalarT *fOwned = fInternalOverlap;
  const double * damageOwned = damage;

  const int *neighPtr = localNeighborList;
  double cellVolume, alpha, X_dx, X_dy, X_dz, zeta, omega, harmonicAverageDamage;
  ScalarT Y_dx, Y_dy, Y_dz, dY, t, fx, fy, fz, e, c1;
  for(int p=0;p<numOwnedPoints;p++, porePressureYOwned++, xOwned +=3, yOwned +=3, fOwned+=3, deltaT++, m++, theta++, damageOwned++){

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
      //harmonicAverageDamage = 1.0 / (1.0 / *damageOwned + 1.0 / *damageNeighbor);
      //if(harmonicAverageDamage != harmonicAverageDamage) harmonicAverageDamage=0.0; //test for nan
      //c1 = omega*(*theta)*(3.0*K/(*m)-alpha/3.0) -3.0*omega/(*m)*(1.0+harmonicAverageDamage)*(*porePressureYOwned);
      c1 = omega*(*theta)*(3.0*K/(*m)-alpha/3.0) -3.0*omega/(*m)*(*porePressureYOwned);
      t = (1.0-*bondDamage)*(c1 * zeta + (1.0-*bondDamage) * omega * alpha * e);

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
  const double thermalExpansionCoefficient,
  const double* deltaTemperature
);

/** Explicit template instantiation for Sacado::Fad::DFad<double>. */
template void computeInternalForceLinearElasticCoupled<Sacado::Fad::DFad<double> >
(
  const double* xOverlap,
  const Sacado::Fad::DFad<double>* yOverlap,
  const Sacado::Fad::DFad<double>* porePressureYOverlap,
  const double* mOwned,
  const double* volumeOverlap,
  const Sacado::Fad::DFad<double>* dilatationOwned,
  const double* damage,
  const double* bondDamage,
  const double* dsfOwned,
  Sacado::Fad::DFad<double>* fInternalOverlap,
  const int*  localNeighborList,
  int numOwnedPoints,
  const double BULK_MODULUS,
  const double SHEAR_MODULUS,
  const double horizon,
  const double thermalExpansionCoefficient,
  const double* deltaTemperature
);

/** Explicit template instantiation for std::complex<double>. */
template void computeInternalForceLinearElasticCoupled<std::complex<double> >
(
  const double* xOverlap,
  const std::complex<double>* yOverlap,
  const std::complex<double>* porePressureYOverlap,
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
  const double* deltaTemperature
);



}
