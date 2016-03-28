//! \file one_phase_nonlocal_diffusion.cxx

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
#include "one_phase_nonlocal_diffusion.h"
#include "material_utilities.h"
#include <boost/math/special_functions/fpclassify.hpp>


namespace MATERIAL_EVALUATION {

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

  const double *v = volumeOverlap;
  const double *deltaT = deltaTemperature;
  const double *damageOwned = damage;
  const double *principleDamageDirectionOwned = principleDamageDirection;
  const double *thetaCritical = criticalDilatationOwned;
  const ScalarT *thetaBreakless = breaklessDilatationOwned;

  ScalarT *phaseOnePoreFlowOwned = phaseOnePoreFlowOverlap;
  ScalarT *phaseOneFracFlowOwned = phaseOneFracFlowOverlap;

  const int *neighPtr = localNeighborList;
  double cellVolume, harmonicAverageDamage;
  ScalarT phaseOnePorePerm;
  ScalarT dPorePressure, dFracPressure, dLocalPoreFracPressure, Y_dx, Y_dy, Y_dz, dY, fractureWidthFactor;
  ScalarT fractureDirectionFactor, phaseOneFracPerm, phaseOneRelPermPores;
  ScalarT phaseOneRelPermFrac;
  ScalarT scalarPhaseOnePoreFlow, scalarPhaseOneFracFlow;
  ScalarT scalarPhaseOneFracToPoreFlow, omegaPores, omegaFrac;

  for(int p=0;p<numOwnedPoints;p++, porePressureYOwned++, fracturePressureYOwned++,
                                    yOwned +=3, phaseOnePoreFlowOwned++, phaseOneFracFlowOwned++,
                                    deltaT++, damageOwned++,
                                    principleDamageDirectionOwned +=3, thetaCritical++, thetaBreakless++,
                                    porePressureVOwned++, fracturePressureVOwned++){
    int numNeigh = *neighPtr; neighPtr++;
    double selfCellVolume = v[p];
    const ScalarT *Y = yOwned;
    const ScalarT *porePressureY = porePressureYOwned;
    const double *porePressureV = porePressureVOwned;
    const ScalarT *fracturePressureY = fracturePressureYOwned;
    const double *fracturePressureV = fracturePressureVOwned;
    const double *principleDamageDirection = principleDamageDirectionOwned;

    phaseOneRelPermPores = m_phaseOneBasePerm; //Empirical model, exponent is related to the material
    phaseOneRelPermFrac = m_phaseOneBasePerm;

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
      // Frac permeability is affected by bond allignment with fracture plane, width, and saturation
      phaseOneFracPerm = fractureWidthFactor*fractureDirectionFactor*phaseOneRelPermFrac;

      // compute flow density
      // flow entering cell is positive
      scalarPhaseOnePoreFlow = omegaPores * m_phaseOneDensity / m_phaseOneViscosity * (4.0 / (M_PI*m_horizon*m_horizon)) * (phaseOnePorePerm / pow(dY, 4.0)) * dPorePressure;
      scalarPhaseOneFracFlow = omegaFrac * m_phaseOneDensity / m_phaseOneViscosity * (4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture)) * (phaseOneFracPerm / pow(dY, 4.0)) * dFracPressure;

      // convert flow density to flow and account for reactions
      *phaseOnePoreFlowOwned += scalarPhaseOnePoreFlow*cellVolume;
      *phaseOneFracFlowOwned += scalarPhaseOneFracFlow*cellVolume;
      phaseOnePoreFlowOverlap[localId] -= scalarPhaseOnePoreFlow*selfCellVolume;
      phaseOneFracFlowOverlap[localId] -= scalarPhaseOneFracFlow*selfCellVolume;
    }

    //Leakoff calculation, self to self
    *phaseOnePoreFlowOwned += m_phaseOneDensity / m_phaseOneViscosity * 4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture) * (m_permeabilityScalar*phaseOneRelPermPores + m_maxPermeability/(1.0 + exp(-m_permeabilityAlpha*(*damageOwned - m_permeabilityCurveInflectionDamage))) / pow(m_horizon_fracture, 4.0)) * dLocalPoreFracPressure;
    *phaseOneFracFlowOwned -= m_phaseOneDensity / m_phaseOneViscosity * 4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture) * (m_permeabilityScalar*phaseOneRelPermPores + m_maxPermeability/(1.0 + exp(-m_permeabilityAlpha*(*damageOwned - m_permeabilityCurveInflectionDamage))) / pow(m_horizon_fracture, 4.0)) * dLocalPoreFracPressure;

    //Viscous terms
    double Porosity= 0.2;
    double B_formation_vol_factor_water = 1.0;
    double B_formation_vol_factor_oil = 1.0;
    double Compressibility_rock = 1.0;
    *phaseOnePoreFlowOwned = *phaseOnePoreFlowOwned - *porePressureV*m_phaseOneDensity*(Compressibility_rock + m_phaseOneCompressibility)*Porosity/B_formation_vol_factor_water;
    *phaseOneFracFlowOwned = *phaseOneFracFlowOwned - *fracturePressureV*m_phaseOneDensity*(Compressibility_rock + m_phaseOneCompressibility)*Porosity/B_formation_vol_factor_water;
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
  const double* volumeOverlap,
  const double* damage,
  const double* principleDamageDirection,
  const double* criticalDilatationOwned,
  const double* breaklessDilatationOwned,
  double* phaseOnePoreFlowOverlap,
  double* phaseOneFracFlowOverlap,
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

/** Explicit template instantiation for Sacado::Fad::DFad<double>. */
template void computeInternalFlow<Sacado::Fad::DFad<double> >
(
	const Sacado::Fad::DFad<double>* yOverlap,
  const Sacado::Fad::DFad<double>* porePressureYOverlap,
  const double* porePressureVOverlap,
  const Sacado::Fad::DFad<double>* fracturePressureYOverlap,
  const double* fracturePressureVOverlap,
  const double* volumeOverlap,
  const double* damage,
  const double* principleDamageDirection,
  const double* criticalDilatationOwned,
  const Sacado::Fad::DFad<double>* breaklessDilatationOwned,
  Sacado::Fad::DFad<double>* phaseOnePoreFlowOverlap,
  Sacado::Fad::DFad<double>* phaseOneFracFlowOverlap,
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

/*Because the complex case needs a difference argument pattern we need something
different than templates. */
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

  const double *v = volumeOverlap;
  const double *deltaT = deltaTemperature;
  const double *damageOwned = damage;
  const double *principleDamageDirectionOwned = principleDamageDirection;
  const double *thetaCritical = criticalDilatationOwned;
  const std::complex<double> *thetaBreakless = breaklessDilatationOwned;

  std::complex<double> *phaseOnePoreFlowOwned = phaseOnePoreFlowOverlap;
  std::complex<double> *phaseOneFracFlowOwned = phaseOneFracFlowOverlap;

  const int *neighPtr = localNeighborList;
  double cellVolume, harmonicAverageDamage;
  std::complex<double> phaseOnePorePerm;
  std::complex<double> dPorePressure, dFracPressure, dLocalPoreFracPressure, Y_dx, Y_dy, Y_dz, dY, fractureWidthFactor;
  std::complex<double> fractureDirectionFactor, phaseOneFracPerm, phaseOneRelPermPores;
  std::complex<double> phaseOneRelPermFrac, satStarPores, satStarFrac;
  std::complex<double> scalarPhaseOnePoreFlow, scalarPhaseOneFracFlow;
  std::complex<double> scalarPhaseOneFracToPoreFlow, omegaPores, omegaFrac;

  for(int p=0;p<numOwnedPoints;p++, porePressureYOwned++, fracturePressureYOwned++,
                                    yOwned +=3, phaseOnePoreFlowOwned++, phaseOneFracFlowOwned++,
                                    deltaT++, damageOwned++,
                                    principleDamageDirectionOwned +=3, thetaCritical++, thetaBreakless++,
                                    porePressureVOwned++, fracturePressureVOwned++){
    int numNeigh = *neighPtr; neighPtr++;
    double selfCellVolume = v[p];
    const std::complex<double> *Y = yOwned;
    const std::complex<double> *porePressureY = porePressureYOwned;
    const std::complex<double> *porePressureV = porePressureVOwned;
    const std::complex<double> *fracturePressureY = fracturePressureYOwned;
    const std::complex<double> *fracturePressureV = fracturePressureVOwned;
    const double *principleDamageDirection = principleDamageDirectionOwned;

    // compute relative permeabilities assuming no damage effect
    phaseOneRelPermPores = m_phaseOneBasePerm; //Empirical model, exponent is related to the material
    phaseOneRelPermFrac = m_phaseOneBasePerm;

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

      // Frac permeability is affected by bond allignment with fracture plane, width, and saturation
      phaseOneFracPerm = fractureWidthFactor*fractureDirectionFactor*phaseOneRelPermFrac;

      // compute flow density
      // flow entering cell is positive
      scalarPhaseOnePoreFlow = omegaPores * m_phaseOneDensity / m_phaseOneViscosity * (4.0 / (M_PI*m_horizon*m_horizon)) * (phaseOnePorePerm / pow(dY, 4.0)) * dPorePressure;
      scalarPhaseOneFracFlow = omegaFrac * m_phaseOneDensity / m_phaseOneViscosity * (4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture)) * (phaseOneFracPerm / pow(dY, 4.0)) * dFracPressure;

      // convert flow density to flow and account for reactions
      *phaseOnePoreFlowOwned += scalarPhaseOnePoreFlow*cellVolume;
      *phaseOneFracFlowOwned += scalarPhaseOneFracFlow*cellVolume;
      phaseOnePoreFlowOverlap[localId] -= scalarPhaseOnePoreFlow*selfCellVolume;
      phaseOneFracFlowOverlap[localId] -= scalarPhaseOneFracFlow*selfCellVolume;
    }

    //Leakoff calculation, self to self
    *phaseOnePoreFlowOwned += m_phaseOneDensity / m_phaseOneViscosity * 4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture) * (m_permeabilityScalar*phaseOneRelPermPores + m_maxPermeability/(1.0 + exp(-m_permeabilityAlpha*(*damageOwned - m_permeabilityCurveInflectionDamage))) / pow(m_horizon_fracture, 4.0)) * dLocalPoreFracPressure;
    *phaseOneFracFlowOwned -= m_phaseOneDensity / m_phaseOneViscosity * 4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture) * (m_permeabilityScalar*phaseOneRelPermPores + m_maxPermeability/(1.0 + exp(-m_permeabilityAlpha*(*damageOwned - m_permeabilityCurveInflectionDamage))) / pow(m_horizon_fracture, 4.0)) * dLocalPoreFracPressure;

    //Viscous terms
    double Porosity= 0.2;
    double B_formation_vol_factor_water = 1.0;
    double B_formation_vol_factor_oil = 1.0;
    double Compressibility_rock = 1.0;
    *phaseOnePoreFlowOwned = *phaseOnePoreFlowOwned - *porePressureV*m_phaseOneDensity*(Compressibility_rock + m_phaseOneCompressibility)*Porosity/B_formation_vol_factor_water;
    *phaseOneFracFlowOwned = *phaseOneFracFlowOwned - *fracturePressureV*m_phaseOneDensity*(Compressibility_rock + m_phaseOneCompressibility)*Porosity/B_formation_vol_factor_water;
  }
}

}
